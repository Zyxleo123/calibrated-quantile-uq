import pickle
import os, sys
import argparse
from argparse import Namespace
from copy import deepcopy
import numpy as np
import pickle as pkl
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from data.fetch_data import get_uci_data, get_toy_data
from utils.misc_utils import (
    test_uq,
    set_seeds,
    discretize_domain,
    EceSharpFrontier,
    get_save_file_name,
    compute_marginal_sharpness
)
from recal import iso_recal
from utils.q_model_ens import QModelEns, EnhancedMLP
from losses import (
    cali_loss,
    batch_cali_loss,
    qr_loss,
    batch_qr_loss,
    interval_loss,
    batch_interval_loss,
    mse_loss_fn
)
from quantile_models import average_calibration, bag_nll, crps_score, mpiw, interval_score, check_loss, mean_variance
from uci_model_agn import main as maqr_main
from calipso import main as calipso_main
import time

def get_loss_fn(loss_name):
    if loss_name == "qr":
        fn = qr_loss
    elif loss_name == "batch_qr":
        fn = batch_qr_loss
    elif loss_name in [
        "cal",
        "scaled_cal",
        "cal_penalty",
        "scaled_cal_penalty",
    ]:
        fn = cali_loss
    elif loss_name in [
        "batch_cal",
        "scaled_batch_cal",
        "batch_cal_penalty",
        "scaled_batch_cal_penalty",
    ]:
        fn = batch_cali_loss
    elif loss_name == "int":
        fn = interval_loss
    elif loss_name == "batch_int":
        fn = batch_interval_loss
    elif loss_name == "maqr":
        fn = mse_loss_fn
    elif loss_name == "calipso":
        fn = None
    else:
        raise ValueError("loss arg not valid")

    return fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models_path", type=str, default="", help="path to trained models"
    )
    parser.add_argument(
        "--min_thres", type=float, default=0.001, help="Minimum ECE threshold"
    )
    parser.add_argument(
        "--max_thres", type=float, default=0.15, help="Maximum ECE threshold"
    )
    parser.add_argument(
        "--num_thres", type=int, default=150, help="Number of ECE thresholds"
    )
    parser.add_argument(
        "--num_ens", type=int, default=1, help="number of members in ensemble"
    )
    parser.add_argument(
        "--boot", type=int, default=0, help="1 to bootstrap samples"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/UCI_Datasets",
        help="parent directory of datasets",
    )
    parser.add_argument(
        "--data", type=str, default="boston", help="dataset to use"
    )
    parser.add_argument(
        "--num_q",
        type=int,
        default=30,
        help="number of quantiles you want to sample each step",
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu num to use")

    parser.add_argument(
        "--num_ep", type=int, default=1000, help="number of epochs"
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=200,
        help="number of epochs to wait before early stopping",
    )
    parser.add_argument("--nl", type=int, default=1, help="number of layers")
    parser.add_argument("--hs", type=int, default=32, help="hidden size")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="weight decay")
    parser.add_argument("--bs", type=int, default=64, help="batch size")

    parser.add_argument("--loss", type=str, default='scaled_batch_cal',
                        help="specify type of loss")

    # New model architecture / regularization flags
    parser.add_argument(
        "--residual",
        type=int,
        default=0,
        help="1 to enable residual connections between hidden layers",
    )
    parser.add_argument(
        "--batch_norm",
        type=int,
        default=0,
        help="1 to enable BatchNorm on hidden layers",
    )
    parser.add_argument(
        "--layer_norm",
        type=int,
        default=0,
        help="1 to enable LayerNorm on hidden layers (ignored if batch_norm enabled)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="dropout probability to apply on each hidden layer",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="activation to use in MLP (relu, elu, tanh, leaky_relu)",
    )

    # only for cali losses
    parser.add_argument(
        "--penalty",
        dest="sharp_penalty",
        type=float,
        help="coefficient for sharpness penalty; 0 for none",
    )
    parser.add_argument(
        "--rand_ref",
        type=int,
        help="1 to use rand reference idxs for cali loss",
    )
    parser.add_argument(
        "--sharp_all",
        type=int,
        default=0,
        help="1 to penalize only widths that are over covered",
    )

    # draw a sorted group batch every
    parser.add_argument(
        "--gdp",
        dest="draw_group_every",
        type=int,
        help="draw a group batch every # epochs",
    )
    parser.add_argument(
        "--recal", type=int, default=1, help="1 to recalibrate after training"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./",
        help="dir to save results",
    )
    parser.add_argument(
        "--skip_existing", type=int, default=1, help="1 to skip existing results"
    )
    parser.add_argument("--debug", type=int, default=0, help="1 to debug")
    
    # MAQR-specific arguments
    parser.add_argument("--dist_type", type=str, default="kernel", help="distance type for MAQR")
    parser.add_argument("--num_in_bin", type=int, default=40, help="number of points in bin for MAQR")
    args = parser.parse_args()

    if "penalty" in args.loss:
        assert isinstance(args.sharp_penalty, float)
        assert 0.0 <= args.sharp_penalty <= 1.0

        if args.sharp_all is not None:
            args.sharp_all = bool(args.sharp_all)
    else:
        args.sharp_penalty = None
        args.sharp_all = None

    if args.rand_ref is not None:
        args.rand_ref = bool(args.rand_ref)

    if args.draw_group_every is None:
        args.draw_group_every = args.num_ep + 1

    args.boot = bool(args.boot)
    args.recal = bool(args.recal)
    args.debug = bool(args.debug)

    # normalize new flags to booleans consistent with other flags
    args.residual = bool(args.residual)
    args.batch_norm = bool(args.batch_norm)
    args.layer_norm = bool(args.layer_norm)
    # dropout and activation left as-is (float and str)

    if args.boot:
        if not args.num_ens > 1:
            raise RuntimeError("num_ens must be above > 1 for bootstrap")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    args.device = device

    return args


if __name__ == "__main__":
    # DATA_NAMES = ['wine', 'naval', 'kin8nm', 'energy', 'yacht', 'concrete', 'power', 'boston']
    print("Running new version")

    args = parse_args()

    # print("DEVICE: {}".format(args.device))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    per_seed_cali = []
    per_seed_sharp = []
    per_seed_gcali = []
    per_seed_crps = []
    per_seed_nll = []
    per_seed_check = []
    per_seed_int = []
    per_seed_int_cali = []
    per_seed_model = []

    va_bag_nll_list = []
    va_crps_list = []
    va_mpiw_list = []
    va_interval_list = []
    va_check_list = []

    # print(
    #     "Drawing group batches every {}, penalty {}".format(
    #         args.draw_group_every, args.sharp_penalty
    #     )
    # )

    # Save file name
    if "penalty" not in args.loss:
        save_file_name = os.path.join(args.save_dir, args.models_path.replace('_models.pkl', '.pkl'))
    else:
        # penalizing sharpness
        if args.sharp_all is not None and args.sharp_all:
            save_file_name = "{}/{}_loss{}_pen{}_sharpall_ens{}_boot{}_seed{}_thres{}.pkl".format(
                args.save_dir,
                args.data,
                args.loss,
                args.sharp_penalty,
                args.num_ens,
                args.boot,
                args.seed,
                args.max_thres
            )
        elif args.sharp_all is not None and not args.sharp_all:
            save_file_name = "{}/{}_loss{}_pen{}_wideonly_ens{}_boot{}_seed{}_thres{}.pkl".format(
                args.save_dir,
                args.data,
                args.loss,
                args.sharp_penalty,
                args.num_ens,
                args.boot,
                args.seed,
                args.max_thres
            )

    if os.path.exists(save_file_name) and args.skip_existing and not args.debug:
        print("skipping {}".format(save_file_name), flush=True)
        sys.exit()

    # Set seeds
    set_seeds(args.seed)

    # --- NEW: redirect prints and tqdm to a logfile under ./<data>/<basename_of_savefile>.log ---
    log_dir = os.path.join("log", args.data)
    os.makedirs(log_dir, exist_ok=True)
    log_basename = os.path.basename(save_file_name).replace(".pkl", ".log")
    log_path = os.path.join(log_dir, log_basename)
    # open in append mode with line buffering
    log_f = open(log_path, "a", buffering=1)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = log_f
    sys.stderr = log_f

    # Fetching data
    data_args = Namespace(
        data_dir=args.data_dir, dataset=args.data, seed=args.seed
    )

    if "uci" in args.data_dir.lower():
        data_out = get_uci_data(args)
    elif "toy" in args.data_dir.lower():
        data_out = get_toy_data(args)

    x_tr, x_va, x_te, y_tr, y_va, y_te, y_al = (
        data_out.x_tr,
        data_out.x_va,
        data_out.x_te,
        data_out.y_tr,
        data_out.y_va,
        data_out.y_te,
        data_out.y_al,
    )
    y_range = (y_al.max() - y_al.min()).item()


    # print(f"Training size {len(x_tr)}")
    print(f"Training size {len(x_tr)}, Validation size {len(x_va)}, Test size {len(x_te)}") 
    # Check if using MAQR approach
    if args.loss == 'maqr':
        # Use MAQR training procedure from uci_model_agn.py
        dim_y = y_tr.shape[1]
        args.mean_model = EnhancedMLP
        args.mean_model_args = {
            "output_size": dim_y,
            "hidden_size": args.hs,
            "num_layers": args.nl,
            "residual": args.residual,
            "batch_norm": args.batch_norm,
            "layer_norm": args.layer_norm,
            "dropout": args.dropout,
            "activation": args.activation,
        }
        model_ens, loader, cdf_x_va_tensor, cdf_y_va_tensor, pred_mean_va, pred_mean_te = maqr_main(
            from_main_py=True, args=args
        )
        
        # For MAQR, we need to center the targets for evaluation
        y_va_centered = y_va - torch.from_numpy(pred_mean_va).to(y_va.device)
        y_te_centered = y_te - torch.from_numpy(pred_mean_te).to(y_te.device)

    # Loss function
    args.scale = True if "scale" in args.loss else False
    batch_loss = True if "batch" in args.loss else False

    testing_device = torch.device('cuda:0')
    x_tr, y_tr, x_va, y_va, x_te, y_te = (
        x_tr.to(testing_device),
        y_tr.to(testing_device),
        x_va.to(testing_device),
        y_va.to(testing_device),
        x_te.to(testing_device),
        y_te.to(testing_device),
    )
    if args.loss == 'maqr':
        y_va = y_va_centered.to(testing_device)
        y_te = y_te_centered.to(testing_device)


    metrics_controlled = []
    # --- PRE-LOADING PATCH: Define the missing classes ---
    import __main__
    class VanillaModel(nn.Module):
        def __init__(self, nfeatures):
            super().__init__()
            self.net = None
        def forward(self, x):
            return self.net(x)
    __main__.VanillaModel = VanillaModel
    __main__.EnhancedMLP = EnhancedMLP
    __main__.QModelEns = QModelEns

    models_controlled = pickle.load(open(args.models_path, 'rb'))
    # with open(tqdm_out_path, 'a', buffering=1) as tqdm_out:
    import time
    i = 0
    # replace tqdm.tqdm(models_controlled) with file-targeted tqdm
    for controlled_model_ens in tqdm.tqdm(models_controlled, file=log_f):
        current_metrics_tmp = {}
        current_metrics_tmp['model_controlled'] = controlled_model_ens
        controlled_model_ens.use_device(testing_device)
        print(f"Evaluating controlled model {i+1}/{len(models_controlled)}")
        # Test UQ on val with controlled model
        start_time = time.time()
        print("start test_uq val for recal time")
        va_exp_props_controlled_recal = torch.linspace(-2.0, 3.0, 501, device=testing_device)
        _, va_obs_props_controlled_recal = test_uq(
            controlled_model_ens, x_va, y_va, va_exp_props_controlled_recal, y_range, recal_model=None, recal_type=None, output_sharp_score_only=True
        )
        print(f"test_uq val for recal time: {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        print("start test_uq val time")
        va_exp_props_controlled_tmp = torch.linspace(0.01, 0.99, 99, device=testing_device)
        current_metrics_tmp['va_sharp_score_controlled'], current_metrics_tmp['va_obs_props_controlled'] = test_uq(
            controlled_model_ens, x_va, y_va, va_exp_props_controlled_tmp, y_range, recal_model=None, recal_type=None, output_sharp_score_only=True
        )
        print(f"test_uq val time: {time.time() - start_time:.2f} seconds")
        start_time = time.time()
        print("start average_calibration val time")
        current_metrics_tmp['va_ece_controlled'] = average_calibration(
            controlled_model_ens, x_va, y_va, args=Namespace(exp_props=va_exp_props_controlled_tmp, device=testing_device, metric="cal_q", calipso=args.loss == 'calipso')
        )
        print(f"average_calibration val time: {time.time() - start_time:.2f} seconds")

        # Test UQ on test with controlled model
        start_time = time.time()
        print("start test_uq test time")
        te_exp_props_controlled_tmp = torch.linspace(0.01, 0.99, 99, device=testing_device)
        current_metrics_tmp['te_sharp_score_controlled'], current_metrics_tmp['te_obs_props_controlled'] = test_uq(
            controlled_model_ens, x_te, y_te, te_exp_props_controlled_tmp, y_range, recal_model=None, recal_type=None, output_sharp_score_only=True
        )
        print(f"test_uq test time: {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        print("start average_calibration test time")
        current_metrics_tmp['te_ece_controlled'] = average_calibration(
            controlled_model_ens, x_te, y_te, args=Namespace(exp_props=te_exp_props_controlled_tmp, device=testing_device, metric="cal_q", calipso=args.loss == 'calipso')
        )
        print(f"average_calibration test time: {time.time() - start_time:.2f} seconds")

        # Other scoring rules on test
        args_for_score = Namespace(device=testing_device, q_list=torch.linspace(0.01, 0.99, 99), alpha_list=torch.linspace(0.01, 0.20, 20), loss=args.loss)
        # start_time = time.time()
        # print("start bag_nll test time")
        # try:
        #     current_metrics_tmp['te_bag_nll_controlled'] = float(bag_nll(controlled_model_ens, x_te, y_te, args_for_score))
        # except Exception as e:
        #     current_metrics_tmp['te_bag_nll_controlled'] = float('nan')
        #     raise e
        # print(f"bag_nll test time: {time.time() - start_time:.2f} seconds")
        start_time = time.time()
        print("start crps_score test time")
        try:
            current_metrics_tmp['te_crps_controlled'] = float(crps_score(controlled_model_ens, x_te, y_te, args_for_score))
        except Exception as e:
            current_metrics_tmp['te_crps_controlled'] = float('nan')
            raise e
        print(f"crps_score test time: {time.time() - start_time:.2f} seconds")
        start_time = time.time()
        print("start mpiw test time")
        try:
            current_metrics_tmp['te_mpiw_controlled'] = float(torch.mean(mpiw(controlled_model_ens, x_te, y_te, args_for_score)))
        except Exception as e:
            current_metrics_tmp['te_mpiw_controlled'] = float('nan')
            raise e
        print(f"mpiw test time: {time.time() - start_time:.2f} seconds")
        start_time = time.time()
        print("start interval_score test time")
        try:
            current_metrics_tmp['te_interval_controlled'] = float(interval_score(controlled_model_ens, x_te, y_te, args_for_score))
        except Exception as e:
            current_metrics_tmp['te_interval_controlled'] = float('nan')
            raise e
        print(f"interval_score test time: {time.time() - start_time:.2f} seconds")
        start_time = time.time()
        print("start check_loss test time")
        try:
            current_metrics_tmp['te_check_controlled'] = float(check_loss(controlled_model_ens, x_te, y_te, args_for_score))
        except Exception as e:
            current_metrics_tmp['te_check_controlled'] = float('nan')
            raise e
        print(f"check_loss test time: {time.time() - start_time:.2f} seconds")
        start_time = time.time()
        print("start mean_variance test time")
        try:
            current_metrics_tmp['te_variance_controlled'] = float(mean_variance(controlled_model_ens, x_te, y_te, args_for_score))
        except Exception as e:
            current_metrics_tmp['te_variance_controlled'] = float('nan')
            raise e
        print(f"mean_variance test time: {time.time() - start_time:.2f} seconds")
        # Recalibration for controlled model
        if args.recal:
            recal_model_controlled_tmp = iso_recal(va_exp_props_controlled_recal, va_obs_props_controlled_recal)
            recal_exp_props_controlled_tmp = torch.linspace(0.01, 0.99, 99, device=testing_device)
            # Recal on Validation
            start_time = time.time()
            print("start recal test_uq")
            current_metrics_tmp['recal_va_sharp_score_controlled'], current_metrics_tmp['recal_va_obs_props_controlled'] = test_uq(
                controlled_model_ens, x_va, y_va, recal_exp_props_controlled_tmp, y_range, recal_model=recal_model_controlled_tmp, recal_type="sklearn", output_sharp_score_only=True
            )
            print(f"recal val time: {time.time() - start_time:.2f} seconds")
            start_time = time.time()
            print("start recal val ece")
            current_metrics_tmp['recal_va_ece_controlled'] = average_calibration(
                controlled_model_ens, x_va, y_va, args=Namespace(exp_props=recal_exp_props_controlled_tmp, device=testing_device, metric="cal_q", recal_model=recal_model_controlled_tmp, recal_type="sklearn", calipso=args.loss == 'calipso')
            )
            print(f"recal val ece time: {time.time() - start_time:.2f} seconds")
            # Recal on Testing
            start_time = time.time()
            print("start recal test")
            current_metrics_tmp['recal_te_sharp_score_controlled'], current_metrics_tmp['recal_te_obs_props_controlled'] = test_uq(
                controlled_model_ens, x_te, y_te, recal_exp_props_controlled_tmp, y_range, recal_model=recal_model_controlled_tmp, recal_type="sklearn", output_sharp_score_only=True
            )
            print(f"recal test time: {time.time() - start_time:.2f} seconds")
            start_time = time.time()
            print("start recal test ece")
            current_metrics_tmp['recal_te_ece_controlled'] = average_calibration(
                controlled_model_ens, x_te, y_te, args=Namespace(exp_props=recal_exp_props_controlled_tmp, device=testing_device, metric="cal_q", recal_model=recal_model_controlled_tmp, recal_type="sklearn", calipso=args.loss == 'calipso')
            )
            print(f"recal test ece time: {time.time() - start_time:.2f} seconds")
            # Other scoring rules
            args_for_score = Namespace(device=testing_device, q_list=torch.linspace(0.01, 0.99, 99), alpha_list=torch.linspace(0.01, 0.20, 20), recal_model=recal_model_controlled_tmp, recal_type="sklearn", loss=args.loss)
            
            # start_time = time.time()
            # print("start recal bag_nll")
            # try:
            #     current_metrics_tmp['recal_te_bag_nll_controlled'] = float(bag_nll(controlled_model_ens, x_te, y_te, args_for_score))
            # except Exception as e:
            #     current_metrics_tmp['recal_te_bag_nll_controlled'] = float('nan')
            #     raise e
            # print(f"recal bag_nll time: {time.time() - start_time:.2f} seconds")
            start_time = time.time()
            print("start recal crps_score")
            try:
                current_metrics_tmp['recal_te_crps_controlled'] = float(crps_score(controlled_model_ens, x_te, y_te, args_for_score))
            except Exception as e:
                current_metrics_tmp['recal_te_crps_controlled'] = float('nan')
                raise e
            print(f"recal crps_score time: {time.time() - start_time:.2f} seconds")
            start_time = time.time()
            print("start recal mpiw")
            try:
                current_metrics_tmp['recal_te_mpiw_controlled'] = float(torch.mean(mpiw(controlled_model_ens, x_te, y_te, args_for_score)))
            except Exception as e:
                current_metrics_tmp['recal_te_mpiw_controlled'] = float('nan')
                raise e
            print(f"recal mpiw time: {time.time() - start_time:.2f} seconds")
            start_time = time.time()
            print("start recal interval_score")
            try:
                current_metrics_tmp['recal_te_interval_controlled'] = float(interval_score(controlled_model_ens, x_te, y_te, args_for_score))
            except Exception as e:
                current_metrics_tmp['recal_te_interval_controlled'] = float('nan')
                raise e
            print(f"recal interval_score time: {time.time() - start_time:.2f} seconds")
            start_time = time.time()
            print("start recal check_loss")
            try:
                current_metrics_tmp['recal_te_check_controlled'] = float(check_loss(controlled_model_ens, x_te, y_te, args_for_score))
            except Exception as e:
                current_metrics_tmp['recal_te_check_controlled'] = float('nan')
                raise e
            print(f"recal check_loss time: {time.time() - start_time:.2f} seconds")
            start_time = time.time()
            print("start recal mean_variance")
            try:
                current_metrics_tmp['recal_te_variance_controlled'] = float(mean_variance(controlled_model_ens, x_te, y_te, args_for_score))
            except Exception as e:
                current_metrics_tmp['recal_te_variance_controlled'] = float('nan')
                raise e
            print(f"recal mean_variance time: {time.time() - start_time:.2f} seconds")
        metrics_controlled.append(current_metrics_tmp)
        i += 1
    # Compute marginal sharpness of the target variable
    # va_marginal_sharpness = compute_marginal_sharpness(y_va, y_range)
    # te_marginal_sharpness = compute_marginal_sharpness(y_te, y_range)

    # Unpack metrics from the list of dictionaries into lists of metrics
    def dictlist_to_listdict(metrics_list, key):
        return [m[key] if m and key in m else None for m in metrics_list]
    
    save_dic = {}

    # Define keys for unpacking
    controlled_metric_keys = [
        'va_sharp_score_controlled', 'te_sharp_score_controlled', 'va_ece_controlled', 'te_ece_controlled',
        'va_variance_controlled', 'va_cali_score_controlled', 'va_obs_props_controlled', 'va_q_preds_controlled', 'va_g_cali_scores_controlled', 'va_scoring_rules_controlled', 'va_bag_nll_controlled', 'va_crps_controlled', 'va_mpiw_controlled', 'va_interval_controlled', 'va_check_controlled',
        'te_variance_controlled', 'te_cali_score_controlled', 'te_obs_props_controlled', 'te_q_preds_controlled', 'te_g_cali_scores_controlled', 'te_scoring_rules_controlled', 'te_bag_nll_controlled', 'te_crps_controlled', 'te_mpiw_controlled', 'te_interval_controlled', 'te_check_controlled',

        'recal_va_sharp_score_controlled', 'recal_te_sharp_score_controlled', 'recal_va_ece_controlled', 'recal_te_ece_controlled',
        'recal_va_variance_controlled', 'recal_va_cali_score_controlled', 'recal_va_obs_props_controlled', 'recal_va_q_preds_controlled', 'recal_va_g_cali_scores_controlled', 'recal_va_scoring_rules_controlled', 'recal_va_bag_nll_controlled', 'recal_va_crps_controlled', 'recal_va_mpiw_controlled', 'recal_va_interval_controlled', 'recal_va_check_controlled',
        'recal_te_variance_controlled', 'recal_te_cali_score_controlled', 'recal_te_obs_props_controlled', 'recal_te_q_preds_controlled', 'recal_te_g_cali_scores_controlled', 'recal_te_scoring_rules_controlled', 'recal_te_bag_nll_controlled', 'recal_te_crps_controlled', 'recal_te_mpiw_controlled', 'recal_te_interval_controlled', 'recal_te_check_controlled',
    ]

    # Create lists of metrics in the local scope for saving
    for list_name in controlled_metric_keys:
        if len(metrics_controlled) == 0 or list_name not in metrics_controlled[0]:
            save_dic[list_name] = [None for _ in range(len(metrics_controlled))]
        else:
            save_dic[list_name] = dictlist_to_listdict(metrics_controlled, list_name)
    
    save_dic['va_exp_props_controlled'] = [torch.linspace(-2.0, 3.0, 501) for _ in range(len(metrics_controlled))]
    save_dic['te_exp_props_controlled'] = [torch.linspace(0.01, 0.99, 99) for _ in range(len(metrics_controlled))]
    if args.recal:
        save_dic['recal_exp_props_controlled'] = [torch.linspace(0.01, 0.99, 99) for _ in range(len(metrics_controlled))]

    save_var_names = [
        "args", "va_marginal_sharpness", "te_marginal_sharpness",

        "tr_loss_list", "va_loss_list", "te_loss_list",
        "va_sharp_list", "va_ece_list", "va_bag_nll_list", "va_crps_list", "va_mpiw_list", "va_interval_list", "va_check_list",

        "va_sharp_score", "te_sharp_score", "va_ece", "te_ece",

        "va_cali_score", "va_exp_props", "va_obs_props", "va_q_preds", "va_bag_nll", "va_crps", "va_mpiw", "va_interval", "va_check", "va_variance", "va_g_cali_scores", "va_scoring_rules", 
        "te_cali_score", "te_exp_props", "te_obs_props", "te_q_preds", "te_bag_nll", "te_crps", "te_mpiw", "te_interval", "te_check", "te_variance", "te_g_cali_scores", "te_scoring_rules",

        "recal_exp_props",
        "recal_va_sharp_score", "recal_te_sharp_score", "recal_va_ece", "recal_te_ece",

        "recal_va_cali_score", "recal_va_obs_props", "recal_va_q_preds", "recal_va_g_cali_scores", "recal_va_scoring_rules", "recal_va_bag_nll", "recal_va_crps", "recal_va_mpiw", "recal_va_interval", "recal_va_check", "recal_va_variance",
        "recal_te_cali_score", "recal_te_obs_props", "recal_te_q_preds", "recal_te_g_cali_scores", "recal_te_scoring_rules", "recal_te_bag_nll", "recal_te_crps", "recal_te_mpiw", "recal_te_interval", "recal_te_check", "recal_te_variance",
    ]
    
    current_locals = locals()
    for name in save_var_names:
        if name in current_locals:
            save_dic[name] = current_locals[name]
        else:
            # print(f"Warning: Variable '{name}' not found in locals for saving.")
            continue

    with open(save_file_name, "wb") as pf:
        pkl.dump(save_dic, pf)
    print(f"Results saved to {save_file_name}")

    # restore stdout/stderr and close the log file
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    log_f.close()