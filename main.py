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
    get_q_idx,
    discretize_domain,
    gather_loss_per_q,
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
from quantile_models import average_calibration, bag_nll, crps_score, mpiw, interval_score


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
        "--ece_thres", type=float, default=0.1, help="Maximum ECE threshold we would tolarate"
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

    # New per-epoch validation lists for bag nll and crps
    va_bag_nll_list = []
    va_crps_list = []
    # New per-epoch validation lists for mpiw and interval score
    va_mpiw_list = []
    va_interval_list = []

    print(
        "Drawing group batches every {}, penalty {}".format(
            args.draw_group_every, args.sharp_penalty
        )
    )

    # Save file name
    if "penalty" not in args.loss:
        save_file_name = "{}/{}_loss{}_ens{}_boot{}_seed{}_thres{}.pkl".format(
            args.save_dir,
            args.data,
            args.loss,
            args.num_ens,
            args.boot,
            args.seed,
            args.ece_thres
        )
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
                args.ece_thres
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
                args.ece_thres
            )
    # if os.path.exists(save_file_name):
        # print("skipping {}".format(save_file_name))
        # sys.exit()

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
    best_sharp_score = float("inf")
    best_model_ens = None

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

        # Test loss
        x_te, y_te = x_te.to(args.device), y_te.to(args.device)
        with torch.no_grad():
            ep_te_loss = model_ens.loss(
                loss_fn,
                x_te,
                y_te,
                va_te_q_list,
                batch_q=batch_loss,
                take_step=False,
                args=args,
            )
        te_loss_list.append(ep_te_loss)

        # Printing some losses
        if (ep % 200 == 0) or (ep == args.num_ep - 1):
            print("EP:{}".format(ep))
            print("Train loss {}".format(ep_tr_loss))
            print("Val loss {}".format(ep_va_loss))
            print("Test loss {}".format(ep_te_loss))

        # ECE thresholding and best sharpness model selection
        model_ens.use_device(torch.device("cpu"))
        _, sharp_score, _, _, _, _ = test_uq(
            model_ens,
            x_va.cpu(),
            y_va.cpu(),
            va_te_q_list.cpu(),
            y_range,
            recal_model=None,
            recal_type=None,
        )
        ece = average_calibration(
            model_ens,
            x_va.cpu(),
            y_va.cpu(),
            args=Namespace(
                exp_props=va_te_q_list.cpu(),
                device=torch.device("cpu"),
                metric="cal_q"
            )
        )
        # Compute per-epoch validation bag_nll and crps (non-orig)
        args_for_score = Namespace(device=torch.device("cpu"))
        try:
            va_bag = float(bag_nll(model_ens, x_va.cpu(), y_va.cpu(), args_for_score))
        except Exception as e:
            raise ValueError(f"Bag NLL computation failed in EP {ep}: {e}")
        try:
            va_crps = float(crps_score(model_ens, x_va.cpu(), y_va.cpu(), args_for_score))
        except Exception as e:
            raise ValueError(f"CRPS computation failed in EP {ep}: {e}")
        # compute mpiw and interval score for validation (per-epoch)
        try:
            va_mpiw_val = mpiw(model_ens, x_va.cpu(), y_va.cpu(), args_for_score)
            # mpiw returns tensor per-alpha; take mean or first element depending on usage — take mean
            va_mpiw = float(torch.mean(va_mpiw_val).item()) if isinstance(va_mpiw_val, torch.Tensor) else float(va_mpiw_val)
        except Exception as e:
            raise ValueError(f"MPIW computation failed in EP {ep}: {e}")
        try:
            va_int_val = float(interval_score(model_ens, x_va.cpu(), y_va.cpu(), args_for_score))
            va_interval = va_int_val
        except Exception as e:
            raise ValueError(f"Interval score computation failed in EP {ep}: {e}")
        va_bag_nll_list.append(va_bag)
        va_crps_list.append(va_crps)
        va_mpiw_list.append(va_mpiw)
        va_interval_list.append(va_interval)
        # move back to GPU if available for training continuation
        model_ens.use_device(torch.device("cuda"))

        va_sharp_list.append(sharp_score)
        va_ece_list.append(ece)
        if ece < args.ece_thres:
            print(f"EP: {ep}")
            print(f"\tNew admissable ECE: {ece}")
            if sharp_score < best_sharp_score:
                best_sharp_score = sharp_score
                best_model_ens = deepcopy(model_ens)
                print(f"\tNew best sharpness found: {best_sharp_score}")

    # Finished training
    # Move everything to cpu
    x_tr, y_tr, x_va, y_va, x_te, y_te = (
        x_tr.cpu(),
        y_tr.cpu(),
        x_va.cpu(),
        y_va.cpu(),
        x_te.cpu(),
        y_te.cpu(),
    )
    model_ens.use_device(torch.device("cpu"))

    # Test UQ on val
    print("Testing UQ on val")
    va_exp_props = torch.linspace(-2.0, 3.0, 501)
    va_cali_score, va_sharp_score, va_obs_props, va_q_preds, _, _ = test_uq(
        model_ens,
        x_va,
        y_va,
        va_exp_props,
        y_range,
        recal_model=None,
        recal_type=None,
    )
    reduced_va_q_preds = va_q_preds[
        :, get_q_idx(va_exp_props, 0.01) : get_q_idx(va_exp_props, 0.99) + 1
    ]
    # Final model: compute bag_nll and crps on validation
    args_for_score = Namespace(device=torch.device("cpu"))
    va_bag_nll = float(bag_nll(model_ens, x_va, y_va, args_for_score))
    va_crps = float(crps_score(model_ens, x_va, y_va, args_for_score))
    # Final model: compute mpiw and interval on validation
    va_mpiw_val = mpiw(model_ens, x_va, y_va, args_for_score)
    va_mpiw = float(torch.mean(va_mpiw_val).item()) if isinstance(va_mpiw_val, torch.Tensor) else float(va_mpiw_val)
    va_interval = float(interval_score(model_ens, x_va, y_va, args_for_score))

    # Test UQ on test
    print("Testing UQ on test")
    te_exp_props = torch.linspace(0.01, 0.99, 99)
    (
        te_cali_score,
        te_sharp_score,
        te_obs_props,
        te_q_preds,
        te_g_cali_scores,
        te_scoring_rules,
    ) = test_uq(
        model_ens,
        x_te,
        y_te,
        te_exp_props,
        y_range,
        recal_model=None,
        recal_type=None,
        test_group_cal=True,
    )
    # Final model: compute bag_nll and crps on test
    te_bag_nll = float(bag_nll(model_ens, x_te, y_te, Namespace(device=torch.device("cpu"))))
    te_crps = float(crps_score(model_ens, x_te, y_te, Namespace(device=torch.device("cpu"))))
    # Final model: compute mpiw and interval on test
    te_mpiw_val = mpiw(model_ens, x_te, y_te, Namespace(device=torch.device("cpu")))
    te_mpiw = float(torch.mean(te_mpiw_val).item()) if isinstance(te_mpiw_val, torch.Tensor) else float(te_mpiw_val)
    te_interval = float(interval_score(model_ens, x_te, y_te, Namespace(device=torch.device("cpu"))))

    # print('val', va_cali_score, va_sharp_score)
    print("\n")
    print("-" * 80)
    print(args.data)
    print("Draw frequency:", args.draw_group_every)
    print(
        "Test Cali: {:.3f}, Sharp: {:.3f}".format(te_cali_score, te_sharp_score)
    )
    print(te_g_cali_scores[:5])
    print(te_g_cali_scores[5:])
    print(te_scoring_rules)
    print("-" * 80)

    if args.recal:
        recal_model = iso_recal(va_exp_props, va_obs_props)
        recal_exp_props = torch.linspace(0.01, 0.99, 99)

        (
            recal_va_cali_score,
            recal_va_sharp_score,
            recal_va_obs_props,
            recal_va_q_preds,
            recal_va_g_cali_scores,
            recal_va_scoring_rules
        ) = test_uq(
            model_ens,
            x_va,
            y_va,
            recal_exp_props,
            y_range,
            recal_model=recal_model,
            recal_type="sklearn",
            test_group_cal=True,
        )
        # compute recalibrated metrics for final model (using quantile_models functions with recal_model via args)
        args_recal = Namespace(device=torch.device("cpu"), recal_model=recal_model, recal_type="sklearn")
        recal_va_bag_nll = float(bag_nll(model_ens, x_va, y_va, args_recal))
        recal_va_crps = float(crps_score(model_ens, x_va, y_va, args_recal))
        recal_va_mpiw_val = mpiw(model_ens, x_va, y_va, args_recal)
        recal_va_mpiw = float(torch.mean(recal_va_mpiw_val).item()) if isinstance(recal_va_mpiw_val, torch.Tensor) else float(recal_va_mpiw_val)
        recal_va_interval = float(interval_score(model_ens, x_va, y_va, args_recal))

        (
            recal_te_cali_score,
            recal_te_sharp_score,
            recal_te_obs_props,
            recal_te_q_preds,
            recal_te_g_cali_scores,
            recal_te_scoring_rules
        ) = test_uq(
            model_ens,
            x_te,
            y_te,
            recal_exp_props,
            y_range,
            recal_model=recal_model,
            recal_type="sklearn",
            test_group_cal=True,
        )
        recal_te_bag_nll = float(bag_nll(model_ens, x_te, y_te, args_recal))
        recal_te_crps = float(crps_score(model_ens, x_te, y_te, args_recal))
        recal_te_mpiw_val = mpiw(model_ens, x_te, y_te, args_recal)
        recal_te_mpiw = float(torch.mean(recal_te_mpiw_val).item()) if isinstance(recal_te_mpiw_val, torch.Tensor) else float(recal_te_mpiw_val)
        recal_te_interval = float(interval_score(model_ens, x_te, y_te, args_recal))

    # Only perform the extra validation/test and recalibration for the best model
    # if a best model passing the ECE threshold was found during training.
    if best_model_ens is not None:
        best_model_ens.use_device(torch.device("cpu"))
        print("Testing UQ on val with best model")
        va_exp_props_best = torch.linspace(-2.0, 3.0, 501)
        va_cali_score_best, va_sharp_score_best, va_obs_props_best, va_q_preds_best, _, _ = test_uq(
            best_model_ens,
            x_va,
            y_va,
            va_exp_props_best,
            y_range,
            recal_model=None,
            recal_type=None,
        )
        reduced_va_q_preds_best = va_q_preds_best[
            :, get_q_idx(va_exp_props_best, 0.01) : get_q_idx(va_exp_props_best, 0.99) + 1
        ]

        # Best model: compute bag_nll and crps on validation
        va_bag_nll_best = float(bag_nll(best_model_ens, x_va, y_va, Namespace(device=torch.device("cpu"))))
        va_crps_best = float(crps_score(best_model_ens, x_va, y_va, Namespace(device=torch.device("cpu"))))
        va_mpiw_best_val = mpiw(best_model_ens, x_va, y_va, Namespace(device=torch.device("cpu")))
        va_mpiw_best = float(torch.mean(va_mpiw_best_val).item()) if isinstance(va_mpiw_best_val, torch.Tensor) else float(va_mpiw_best_val)
        va_interval_best = float(interval_score(best_model_ens, x_va, y_va, Namespace(device=torch.device("cpu"))))

        # Test UQ on test with best model
        print("Testing UQ on test with best model")
        te_exp_props_best = torch.linspace(0.01, 0.99, 99)
        (
            te_cali_score_best,
            te_sharp_score_best,
            te_obs_props_best,
            te_q_preds_best,
            te_g_cali_scores_best,
            te_scoring_rules_best,
        ) = test_uq(
            best_model_ens,
            x_te,
            y_te,
            te_exp_props_best,
            y_range,
            recal_model=None,
            recal_type=None,
            test_group_cal=True,
        )

        # Best model: compute bag_nll and crps on test
        te_bag_nll_best = float(bag_nll(best_model_ens, x_te, y_te, Namespace(device=torch.device("cpu"))))
        te_crps_best = float(crps_score(best_model_ens, x_te, y_te, Namespace(device=torch.device("cpu"))))
        te_mpiw_best_val = mpiw(best_model_ens, x_te, y_te, Namespace(device=torch.device("cpu")))
        te_mpiw_best = float(torch.mean(te_mpiw_best_val).item()) if isinstance(te_mpiw_best_val, torch.Tensor) else float(te_mpiw_best_val)
        te_interval_best = float(interval_score(best_model_ens, x_te, y_te, Namespace(device=torch.device("cpu"))))

        # Same recalibration procedure for best model
        if args.recal:
            recal_model = iso_recal(va_exp_props_best, va_obs_props_best)
            recal_exp_props = torch.linspace(0.01, 0.99, 99)

            (
                recal_va_cali_score_best,
                recal_va_sharp_score_best,
                recal_va_obs_props_best,
                recal_va_q_preds_best,
                recal_va_g_cali_scores_best,
                recal_va_scoring_rules_best
            ) = test_uq(
                best_model_ens,
                x_va,
                y_va,
                recal_exp_props,
                y_range,
                recal_model=recal_model,
                recal_type="sklearn",
                test_group_cal=True,
            )
            # compute recalibrated metrics for best model
            args_recal_best = Namespace(device=torch.device("cpu"), recal_model=recal_model, recal_type="sklearn")
            recal_va_bag_nll_best = float(bag_nll(best_model_ens, x_va, y_va, args_recal_best))
            recal_va_crps_best = float(crps_score(best_model_ens, x_va, y_va, args_recal_best))
            recal_va_mpiw_best_val = mpiw(best_model_ens, x_va, y_va, args_recal_best)
            recal_va_mpiw_best = float(torch.mean(recal_va_mpiw_best_val).item()) if isinstance(recal_va_mpiw_best_val, torch.Tensor) else float(recal_va_mpiw_best_val)
            recal_va_interval_best = float(interval_score(best_model_ens, x_va, y_va, args_recal_best))

            (
                recal_te_cali_score_best,
                recal_te_sharp_score_best,
                recal_te_obs_props_best,
                recal_te_q_preds_best,
                recal_te_g_cali_scores_best,
                recal_te_scoring_rules_best
            ) = test_uq(
                best_model_ens,
                x_te,
                y_te,
                recal_exp_props,
                y_range,
                recal_model=recal_model,
                recal_type="sklearn",
                test_group_cal=True,
            )
            recal_te_bag_nll_best = float(bag_nll(best_model_ens, x_te, y_te, args_recal_best))
            recal_te_crps_best = float(crps_score(best_model_ens, x_te, y_te, args_recal_best))
            recal_te_mpiw_best_val = mpiw(best_model_ens, x_te, y_te, args_recal_best)
            recal_te_mpiw_best = float(torch.mean(recal_te_mpiw_best_val).item()) if isinstance(recal_te_mpiw_best_val, torch.Tensor) else float(recal_te_mpiw_best_val)
            recal_te_interval_best = float(interval_score(best_model_ens, x_te, y_te, args_recal_best))

    save_dic = {
        # Loss lists from training
        "tr_loss_list": tr_loss_list,
        "va_loss_list": va_loss_list,
        "te_loss_list": te_loss_list,
        
        # Per-epoch validation metrics during training (grouped and top-level for compatibility)
        "train_metrics": {
            "va_sharp_list": va_sharp_list,
            "va_ece_list": va_ece_list,
            "va_bag_nll_list": va_bag_nll_list,
            "va_crps_list": va_crps_list,
            "va_mpiw_list": va_mpiw_list,
            "va_interval_list": va_interval_list,
        },
        "va_sharp_list": va_sharp_list,
        "va_ece_list": va_ece_list,
        "va_bag_nll_list": va_bag_nll_list,
        "va_crps_list": va_crps_list,
        "va_mpiw_list": va_mpiw_list,
        "va_interval_list": va_interval_list,

        # Test metrics for the final model (grouped)
        "test_metrics": {
            "va": {"cali": va_cali_score, "sharp": va_sharp_score, "bag_nll": va_bag_nll, "crps": va_crps, "mpiw": va_mpiw, "interval": va_interval},
            "te": {"cali": te_cali_score, "sharp": te_sharp_score, "bag_nll": te_bag_nll, "crps": te_crps, "mpiw": te_mpiw, "interval": te_interval},
        },

        # Detailed test results on validation and test sets for the final model
        "va_cali_score": va_cali_score, "va_sharp_score": va_sharp_score, "va_exp_props": va_exp_props, "va_obs_props": va_obs_props, "va_q_preds": va_q_preds, "va_bag_nll": va_bag_nll, "va_crps": va_crps, "va_mpiw": va_mpiw, "va_interval": va_interval,
        "te_cali_score": te_cali_score, "te_sharp_score": te_sharp_score, "te_exp_props": te_exp_props, "te_obs_props": te_obs_props, "te_q_preds": te_q_preds, "te_bag_nll": te_bag_nll, "te_crps": te_crps, "te_mpiw": te_mpiw, "te_interval": te_interval, "te_g_cali_scores": te_g_cali_scores, "te_scoring_rules": te_scoring_rules,
        
        # Configuration and final model object
        "args": args,
        "model": model_ens,
    }

    # Conditionally add recalibration results for the final model
    if args.recal:
        recal_final_updates = {
            "recal_model": recal_model, "recal_exp_props": recal_exp_props,
            "recal_va_cali_score": recal_va_cali_score, "recal_va_sharp_score": recal_va_sharp_score, "recal_va_obs_props": recal_va_obs_props, "recal_va_q_preds": recal_va_q_preds, "recal_va_g_cali_scores": recal_va_g_cali_scores, "recal_va_scoring_rules": recal_va_scoring_rules, "recal_va_bag_nll": recal_va_bag_nll, "recal_va_crps": recal_va_crps, "recal_va_mpiw": recal_va_mpiw, "recal_va_interval": recal_va_interval,
            "recal_te_cali_score": recal_te_cali_score, "recal_te_sharp_score": recal_te_sharp_score, "recal_te_obs_props": recal_te_obs_props, "recal_te_q_preds": recal_te_q_preds, "recal_te_g_cali_scores": recal_te_g_cali_scores, "recal_te_scoring_rules": recal_te_scoring_rules, "recal_te_bag_nll": recal_te_bag_nll, "recal_te_crps": recal_te_crps, "recal_te_mpiw": recal_te_mpiw, "recal_te_interval": recal_te_interval,
            "recal_final_metrics": {
                "va": {"cali": recal_va_cali_score, "bag_nll": recal_va_bag_nll, "crps": recal_va_crps, "mpiw": recal_va_mpiw, "interval": recal_va_interval},
                "te": {"cali": recal_te_cali_score, "bag_nll": recal_te_bag_nll, "crps": recal_te_crps, "mpiw": recal_te_mpiw, "interval": recal_te_interval},
            },
        }
        save_dic.update(recal_final_updates)

    # Conditionally add results for the best model found during training
    if best_model_ens is not None:
        best_model_updates = {
            "best_model": best_model_ens,
            "best_model_metrics": {
                "va": {"cali": va_cali_score_best, "sharp": va_sharp_score_best, "bag_nll": va_bag_nll_best, "crps": va_crps_best, "mpiw": va_mpiw_best, "interval": va_interval_best},
                "te": {"cali": te_cali_score_best, "sharp": te_sharp_score_best, "bag_nll": te_bag_nll_best, "crps": te_crps_best, "mpiw": te_mpiw_best, "interval": te_interval_best},
            },
            "va_cali_score_best": va_cali_score_best, "va_sharp_score_best": va_sharp_score_best, "va_exp_props_best": va_exp_props_best, "va_obs_props_best": va_obs_props_best, "va_q_preds_best": va_q_preds_best, "va_bag_nll_best": va_bag_nll_best, "va_crps_best": va_crps_best, "va_mpiw_best": va_mpiw_best, "va_interval_best": va_interval_best,
            "te_cali_score_best": te_cali_score_best, "te_sharp_score_best": te_sharp_score_best, "te_exp_props_best": te_exp_props_best, "te_obs_props_best": te_obs_props_best, "te_q_preds_best": te_q_preds_best, "te_bag_nll_best": te_bag_nll_best, "te_crps_best": te_crps_best, "te_mpiw_best": te_mpiw_best, "te_interval_best": te_interval_best, "te_g_cali_scores_best": te_g_cali_scores_best, "te_scoring_rules_best": te_scoring_rules_best,
        }
        
        # Conditionally add recalibration results for the best model
        if args.recal:
            recal_best_updates = {
                "recal_va_cali_score_best": recal_va_cali_score_best, "recal_va_sharp_score_best": recal_va_sharp_score_best, "recal_va_obs_props_best": recal_va_obs_props_best, "recal_va_q_preds_best": recal_va_q_preds_best, "recal_va_g_cali_scores_best": recal_va_g_cali_scores_best, "recal_va_scoring_rules_best": recal_va_scoring_rules_best, "recal_va_bag_nll_best": recal_va_bag_nll_best, "recal_va_crps_best": recal_va_crps_best, "recal_va_mpiw_best": recal_va_mpiw_best, "recal_va_interval_best": recal_va_interval_best,
                "recal_te_cali_score_best": recal_te_cali_score_best, "recal_te_sharp_score_best": recal_te_sharp_score_best, "recal_te_obs_props_best": recal_te_obs_props_best, "recal_te_q_preds_best": recal_te_q_preds_best, "recal_te_g_cali_scores_best": recal_te_g_cali_scores_best, "recal_te_scoring_rules_best": recal_te_scoring_rules_best, "recal_te_bag_nll_best": recal_te_bag_nll_best, "recal_te_crps_best": recal_te_crps_best, "recal_te_mpiw_best": recal_te_mpiw_best, "recal_te_interval_best": recal_te_interval_best,
                "recal_best_metrics": {
                    "va": {"cali": recal_va_cali_score_best, "bag_nll": recal_va_bag_nll_best, "crps": recal_va_crps_best, "mpiw": recal_va_mpiw_best, "interval": recal_va_interval_best},
                    "te": {"cali": recal_te_cali_score_best, "bag_nll": recal_te_bag_nll_best, "crps": recal_te_crps_best, "mpiw": recal_te_mpiw_best, "interval": recal_te_interval_best},
                },
            }
            best_model_updates.update(recal_best_updates)
        
        save_dic.update(best_model_updates)

    with open(save_file_name, "wb") as pf:
        pkl.dump(save_dic, pf)