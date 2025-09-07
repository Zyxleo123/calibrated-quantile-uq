import os, sys
import argparse
from argparse import Namespace
from copy import deepcopy
import numpy as np
import pickle as pkl
import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from data.fetch_data import get_uci_data, get_toy_data, get_fusion_data
from utils.misc_utils import (
    test_uq,
    set_seeds,
    discretize_domain,
    EceSharpFrontier,
    get_save_file_name,
    compute_marginal_sharpness
)
from recal import iso_recal
from utils.q_model_ens import QModelEns
from losses import (
    cali_loss,
    batch_cali_loss,
    qr_loss,
    batch_qr_loss,
    interval_loss,
    batch_interval_loss,
)
from quantile_models import average_calibration, bag_nll, crps_score, mpiw, interval_score, check_loss


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

    else:
        raise ValueError("loss arg not valid")

    return fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min_thres", type=float, default=0.01, help="Minimum ECE threshold"
    )
    parser.add_argument(
        "--max_thres", type=float, default=0.15, help="Maximum ECE threshold"
    )
    parser.add_argument(
        "--num_thres", type=int, default=100, help="Number of ECE thresholds"
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
    parser.add_argument("--nl", type=int, default=2, help="number of layers")
    parser.add_argument("--hs", type=int, default=64, help="hidden size")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="weight decay")
    parser.add_argument("--bs", type=int, default=64, help="batch size")
    parser.add_argument(
        "--wait",
        type=int,
        default=200,
        help="how long to wait for lower validation loss",
    )

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
        args.draw_group_every = args.num_ep

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

    args = parse_args()

    print("DEVICE: {}".format(args.device))

    if args.debug:
        import pudb

        pudb.set_trace()

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

    print(
        "Drawing group batches every {}, penalty {}".format(
            args.draw_group_every, args.sharp_penalty
        )
    )

    # Save file name
    if "penalty" not in args.loss:
        save_file_name = get_save_file_name(args)

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
    if os.path.exists(save_file_name) and args.skip_existing:
        print("skipping {}".format(save_file_name))
        sys.exit()

    # Set seeds
    set_seeds(args.seed)

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
    print("y range: {:.3f}".format(y_range))

    # Making models
    num_tr = x_tr.shape[0]
    dim_x = x_tr.shape[1]
    dim_y = y_tr.shape[1]
    model_ens = QModelEns(
        input_size=dim_x + 1,
        output_size=dim_y,
        hidden_size=args.hs,
        num_layers=args.nl,
        lr=args.lr,
        wd=args.wd,
        num_ens=args.num_ens,
        device=args.device,
        residual=args.residual,
        batch_norm=args.batch_norm,
        layer_norm=args.layer_norm,
        dropout=args.dropout,
        activation=args.activation,
    )

    # Data loader
    if not args.boot:
        loader = DataLoader(
            TensorDataset(x_tr, y_tr),
            shuffle=True,
            batch_size=args.bs,
        )
    else:
        rand_idx_list = [
            np.random.choice(num_tr, size=num_tr, replace=True)
            for _ in range(args.num_ens)
        ]
        loader_list = [
            DataLoader(
                TensorDataset(x_tr[idxs], y_tr[idxs]),
                shuffle=True,
                batch_size=args.bs,
            )
            for idxs in rand_idx_list
        ]

    # Loss function
    loss_fn = get_loss_fn(args.loss)
    args.scale = True if "scale" in args.loss else False
    batch_loss = True if "batch" in args.loss else False

    """ train loop """
    tr_loss_list = []
    va_loss_list = []
    te_loss_list = []
    va_sharp_list = []
    va_ece_list = []

    frontier = EceSharpFrontier()

    # setting batch groupings
    group_list = discretize_domain(x_tr.numpy(), args.bs)
    curr_group_idx = 0

    for ep in tqdm.tqdm(range(args.num_ep)):
        if model_ens.done_training:
            print("Done training ens at EP {}".format(ep))
            break

        # Take train step
        # list of losses from each batch, for one epoch
        ep_train_loss = []
        if not args.boot:
            if ep % args.draw_group_every == 0:
                # drawing a group batch
                group_idxs = group_list[curr_group_idx]
                curr_group_idx = (curr_group_idx + 1) % dim_x
                for g_idx in group_idxs:
                    xi = x_tr[g_idx.flatten()].to(args.device)
                    yi = y_tr[g_idx.flatten()].to(args.device)
                    q_list = torch.rand(args.num_q)
                    loss = model_ens.loss(
                        loss_fn,
                        xi,
                        yi,
                        q_list,
                        batch_q=batch_loss,
                        take_step=True,
                        args=args,
                    )
                    ep_train_loss.append(loss)
            else:
                # just doing ordinary random batch
                for (xi, yi) in loader:
                    xi, yi = xi.to(args.device), yi.to(args.device)
                    q_list = torch.rand(args.num_q)
                    loss = model_ens.loss(
                        loss_fn,
                        xi,
                        yi,
                        q_list,
                        batch_q=batch_loss,
                        take_step=True,
                        args=args,
                    )
                    ep_train_loss.append(loss)
        else:
            # bootstrapped ensemble of models
            for xi_yi_samp in zip(*loader_list):
                xi_list = [item[0].to(args.device) for item in xi_yi_samp]
                yi_list = [item[1].to(args.device) for item in xi_yi_samp]
                assert len(xi_list) == len(yi_list) == args.num_ens
                q_list = torch.rand(args.num_q)
                loss = model_ens.loss_boot(
                    loss_fn,
                    xi_list,
                    yi_list,
                    q_list,
                    batch_q=batch_loss,
                    take_step=True,
                    args=args,
                )
                ep_train_loss.append(loss)
        ep_tr_loss = np.nanmean(np.stack(ep_train_loss, axis=0), axis=0)
        tr_loss_list.append(ep_tr_loss)

        # Validation loss
        x_va, y_va = x_va.to(args.device), y_va.to(args.device)
        va_te_q_list = torch.linspace(0.01, 0.99, 99).to(args.device)
        ep_va_loss = model_ens.update_va_loss(
            loss_fn,
            x_va,
            y_va,
            va_te_q_list,
            batch_q=batch_loss,
            curr_ep=ep,
            num_wait=args.wait,
            args=args,
        )
        va_loss_list.append(ep_va_loss)

        # Printing some losses
        if (ep % 200 == 0) or (ep == args.num_ep - 1):
            print("EP:{}".format(ep))
            print("Train loss {}".format(ep_tr_loss))
            print("Val loss {}".format(ep_va_loss))

        validation_device = torch.device("cpu")
        model_ens.use_device(validation_device)
        ece = average_calibration(
            model_ens,
            x_va.to(validation_device),
            y_va.to(validation_device),
            args=Namespace(
                exp_props=va_te_q_list.to(validation_device),
                device=validation_device,
                metric="cal_q"
            )
        )
        if ece > args.max_thres:
            model_ens.use_device(args.device)
            continue

        sharp_score, _ = test_uq(
            model_ens,
            x_va.to(validation_device),
            y_va.to(validation_device),
            va_te_q_list.to(validation_device),
            y_range,
            recal_model=None,
            recal_type=None,
            output_sharp_score_only=True
        )
        model_ens.use_device(args.device)

        va_sharp_list.append(sharp_score)
        va_ece_list.append(ece)

        frontier.insert(ece, sharp_score, deepcopy(model_ens), only_frontier=True)

    # Finished training
    print(f"Total of {len(frontier.entries)} frontier entries recorded.")
    # We are interested best sharpness models controlled by set ECE thresholds
    thresholded_frontier = frontier.get_thresholded_frontier(args.min_thres, args.max_thres, args.num_thres).get_entries()

    # Move everything to testing device
    testing_device = torch.device('cpu')
    x_tr, y_tr, x_va, y_va, x_te, y_te = (
        x_tr.to(testing_device),
        y_tr.to(testing_device),
        x_va.to(testing_device),
        y_va.to(testing_device),
        x_te.to(testing_device),
        y_te.to(testing_device),
    )
    model_ens.use_device(testing_device)

    # Test UQ on val
    va_exp_props_recal = torch.linspace(-2.0, 3.0, 501, device=testing_device)
    _, va_obs_props_recal = test_uq(
        model_ens,
        x_va,
        y_va,
        va_exp_props_recal,
        y_range,
        recal_model=None,
        recal_type=None,
        output_sharp_score_only=True
    )

    va_exp_props = torch.linspace(0.01, 0.99, 99, device=testing_device)
    va_sharp_score, va_obs_props = test_uq(
        model_ens,
        x_va,
        y_va,
        va_exp_props,
        y_range,
        recal_model=None,
        recal_type=None,
        output_sharp_score_only=True
    )
    va_ece = average_calibration(
        model_ens,
        x_va,
        y_va,
        args=Namespace(
            exp_props=va_exp_props,
            device=testing_device,
            metric="cal_q"
        )
    )

    # Test UQ on test
    te_exp_props = torch.linspace(0.01, 0.99, 99, device=testing_device)
    te_sharp_score, te_obs_props = test_uq(
        model_ens,
        x_te,
        y_te,
        te_exp_props,
        y_range,
        recal_model=None,
        recal_type=None,
        output_sharp_score_only=True
    )
    te_ece = average_calibration(
        model_ens,
        x_te,
        y_te,
        args=Namespace(
            exp_props=te_exp_props,
            device=testing_device,
            metric="cal_q"
        )
    )

    if args.recal:
        recal_model = iso_recal(va_exp_props_recal, va_obs_props_recal)
        recal_exp_props = torch.linspace(0.01, 0.99, 99, device=testing_device)

        recal_va_sharp_score, recal_va_obs_props = test_uq(
            model_ens,
            x_va,
            y_va,
            recal_exp_props,
            y_range,
            recal_model=recal_model,
            recal_type="sklearn",
            output_sharp_score_only=True
        )
        recal_va_ece = average_calibration(
            model_ens,
            x_va,
            y_va,
            args=Namespace(
                exp_props=recal_exp_props,
                device=testing_device,
                metric="cal_q",
                recal_model=recal_model,
                recal_type="sklearn"
            )
        )

        recal_te_sharp_score, recal_te_obs_props = test_uq(
            model_ens,
            x_te,
            y_te,
            recal_exp_props,
            y_range,
            recal_model=recal_model,
            recal_type="sklearn",
            output_sharp_score_only=True
        )
        recal_te_ece = average_calibration(
            model_ens,
            x_te,
            y_te,
            args=Namespace(
                exp_props=recal_exp_props,
                device=testing_device,
                metric="cal_q",
                recal_model=recal_model,
                recal_type="sklearn"
            )
        )

    all_metrics_controlled = []
    for entry in tqdm.tqdm(thresholded_frontier, desc="Testing controlled models"):
        controlled_model_ens = entry['model']
        current_metrics_tmp = {}
        current_metrics_tmp['model_controlled'] = controlled_model_ens
        controlled_model_ens.use_device(testing_device)

        # Test UQ on val with controlled model
        va_exp_props_controlled_recal = torch.linspace(-2.0, 3.0, 501, device=testing_device)
        _, va_obs_props_controlled_recal = test_uq(
            controlled_model_ens, x_va, y_va, va_exp_props_controlled_recal, y_range, recal_model=None, recal_type=None, output_sharp_score_only=True
        )

        va_exp_props_controlled_tmp = torch.linspace(0.01, 0.99, 99, device=testing_device)
        current_metrics_tmp['va_sharp_score_controlled'], current_metrics_tmp['va_obs_props_controlled'] = test_uq(
            controlled_model_ens, x_va, y_va, va_exp_props_controlled_tmp, y_range, recal_model=None, recal_type=None, output_sharp_score_only=True
        )
        current_metrics_tmp['va_ece_controlled'] = average_calibration(
            controlled_model_ens, x_va, y_va, args=Namespace(exp_props=va_exp_props_controlled_tmp, device=testing_device, metric="cal_q")
        )

        # Test UQ on test with controlled model
        te_exp_props_controlled_tmp = torch.linspace(0.01, 0.99, 99, device=testing_device)
        current_metrics_tmp['te_sharp_score_controlled'], current_metrics_tmp['te_obs_props_controlled'] = test_uq(
            controlled_model_ens, x_te, y_te, te_exp_props_controlled_tmp, y_range, recal_model=None, recal_type=None, output_sharp_score_only=True
        )
        current_metrics_tmp['te_ece_controlled'] = average_calibration(
            controlled_model_ens, x_te, y_te, args=Namespace(exp_props=te_exp_props_controlled_tmp, device=testing_device, metric="cal_q")
        )

        # Recalibration for controlled model
        if args.recal:
            recal_model_controlled_tmp = iso_recal(va_exp_props_controlled_recal, va_obs_props_controlled_recal)
            current_metrics_tmp['recal_model_controlled'] = recal_model_controlled_tmp
            recal_exp_props_controlled_tmp = torch.linspace(0.01, 0.99, 99, device=testing_device)
            # Recal on Validation
            current_metrics_tmp['recal_va_sharp_score_controlled'], current_metrics_tmp['recal_va_obs_props_controlled'] = test_uq(
                controlled_model_ens, x_va, y_va, recal_exp_props_controlled_tmp, y_range, recal_model=recal_model_controlled_tmp, recal_type="sklearn", output_sharp_score_only=True
            )
            current_metrics_tmp['recal_va_ece_controlled'] = average_calibration(
                controlled_model_ens, x_va, y_va, args=Namespace(exp_props=recal_exp_props_controlled_tmp, device=testing_device, metric="cal_q", recal_model=recal_model_controlled_tmp, recal_type="sklearn")
            )
            # Recal on Test
            current_metrics_tmp['recal_te_sharp_score_controlled'], current_metrics_tmp['recal_te_obs_props_controlled'] = test_uq(
                controlled_model_ens, x_te, y_te, recal_exp_props_controlled_tmp, y_range, recal_model=recal_model_controlled_tmp, recal_type="sklearn", output_sharp_score_only=True 
            )
            current_metrics_tmp['recal_te_ece_controlled'] = average_calibration(
                controlled_model_ens, x_te, y_te, args=Namespace(exp_props=recal_exp_props_controlled_tmp, device=testing_device, metric="cal_q", recal_model=recal_model_controlled_tmp, recal_type="sklearn")
            )
        all_metrics_controlled.append(current_metrics_tmp)
    
    # Compute marginal sharpness of the target variable
    va_marginal_sharpness = compute_marginal_sharpness(y_va, y_range)
    te_marginal_sharpness = compute_marginal_sharpness(y_te, y_range)

    # Unpack metrics from the list of dictionaries into lists of metrics
    def dictlist_to_listdict(metrics_list, key):
        return [m[key] if m and key in m else None for m in metrics_list]
    
    # Define keys for unpacking
    controlled_model_metric_keys = [
        'model_controlled', 'va_cali_score_controlled', 'va_sharp_score_controlled', 'va_obs_props_controlled', 'va_q_preds_controlled', 'va_bag_nll_controlled', 'va_crps_controlled', 'va_mpiw_controlled', 'va_interval_controlled', 'va_check_controlled', 'va_ece_controlled',
        'te_cali_score_controlled', 'te_sharp_score_controlled', 'te_obs_props_controlled', 'te_q_preds_controlled', 'te_g_cali_scores_controlled', 'te_scoring_rules_controlled', 'te_bag_nll_controlled', 'te_crps_controlled', 'te_mpiw_controlled', 'te_interval_controlled', 'te_check_controlled', 'te_ece_controlled',
        'recal_model_controlled', 'recal_va_cali_score_controlled', 'recal_va_sharp_score_controlled', 'recal_va_obs_props_controlled', 'recal_va_q_preds_controlled', 'recal_va_g_cali_scores_controlled', 'recal_va_scoring_rules_controlled', 'recal_va_bag_nll_controlled', 'recal_va_crps_controlled', 'recal_va_mpiw_controlled', 'recal_va_interval_controlled', 'recal_va_check_controlled', 'recal_va_ece_controlled',
        'recal_te_cali_score_controlled', 'recal_te_sharp_score_controlled', 'recal_te_obs_props_controlled', 'recal_te_q_preds_controlled', 'recal_te_g_cali_scores_controlled', 'recal_te_scoring_rules_controlled', 'recal_te_bag_nll_controlled', 'recal_te_crps_controlled', 'recal_te_mpiw_controlled', 'recal_te_interval_controlled', 'recal_te_check_controlled', 'recal_te_ece_controlled',
    ]

    # Create lists of metrics in the local scope for saving
    for list_name in controlled_model_metric_keys:
        locals()[list_name] = dictlist_to_listdict(all_metrics_controlled, list_name.replace('_list_best', '_best'))

    save_var_names = [
        "args", "va_marginal_sharpness", "te_marginal_sharpness",

        "tr_loss_list", "va_loss_list", "te_loss_list",
        "va_sharp_list", "va_ece_list", "va_bag_nll_list", "va_crps_list", "va_mpiw_list", "va_interval_list", "va_check_list",

        "va_sharp_score", "te_sharp_score", "va_ece", "te_ece",

        "va_cali_score", "va_exp_props", "va_obs_props", "va_q_preds", "va_bag_nll", "va_crps", "va_mpiw", "va_interval", "va_check",
        "te_cali_score", "te_exp_props", "te_obs_props", "te_q_preds", "te_bag_nll", "te_crps", "te_mpiw", "te_interval", "te_check", "te_g_cali_scores", "te_scoring_rules",

        "recal_exp_props",
        "recal_va_sharp_score", "recal_te_sharp_score", "recal_va_ece", "recal_te_ece",
        "recal_va_cali_score", "recal_va_obs_props", "recal_va_q_preds", "recal_va_g_cali_scores", "recal_va_scoring_rules", "recal_va_bag_nll", "recal_va_crps", "recal_va_mpiw", "recal_va_interval", "recal_va_check",
        "recal_te_cali_score", "recal_te_obs_props", "recal_te_q_preds", "recal_te_g_cali_scores", "recal_te_scoring_rules", "recal_te_bag_nll", "recal_te_crps", "recal_te_mpiw", "recal_te_interval", "recal_te_check",
    ]
    
    save_var_names.extend(controlled_model_metric_keys)

    save_dic = {}
    current_locals = locals()
    for name in save_var_names:
        if name in current_locals:
            save_dic[name] = current_locals[name]
        else:
            print(f"Warning: Variable '{name}' not found in locals for saving.")
            continue
    
    save_dic['va_exp_props_controlled'] = [torch.linspace(-2.0, 3.0, 501) for _ in range(args.num_thres)]
    save_dic['te_exp_props_controlled'] = [torch.linspace(0.01, 0.99, 99) for _ in range(args.num_thres)]
    if args.recal:
        save_dic['recal_exp_props_controlled'] = [torch.linspace(0.01, 0.99, 99) for _ in range(args.num_thres)]
    save_dic['args'] = vars(args)

    with open(save_file_name, "wb") as pf:
        pkl.dump(save_dic, pf)
    print(f"Results saved to {save_file_name}")