import os, sys
from copy import deepcopy
import tqdm
import numpy as np
import torch
import torch.nn as nn

# sys.path.append('../utils/NNKit')
# sys.path.append('utils')
from scipy.stats import norm as norm_distr
from scipy.stats import t as t_distr
from scipy.interpolate import interp1d

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from NNKit.models.model import vanilla_nn

"""
Define wrapper uq_model class
All uq models will import this class
"""


class uq_model(object):
    def predict(self):
        raise NotImplementedError("Abstract Method")


""" QModelEns Utils """


def gather_loss_per_q(loss_fn, model, y, x, q_list, device, args):
    loss_list = []
    for q in q_list:
        q_loss = loss_fn(model, y, x, q, device, args)
        loss_list.append(q_loss)
    loss = torch.mean(torch.stack(loss_list))

    return loss


def get_ens_pred_interp(unc_preds, taus, fidelity=10000):
    """
    unc_preds 3D ndarray (ens_size, 99, num_x)
    where for each ens_member, each row corresonds to tau 0.01, 0.02...
    and the columns are for the set of x being predicted over.
    """
    # taus = np.arange(0.01, 1, 0.01)
    y_min, y_max = np.min(unc_preds), np.max(unc_preds)
    y_grid = np.linspace(y_min, y_max, fidelity)
    new_quants = []
    avg_cdfs = []
    for x_idx in tqdm.tqdm(range(unc_preds.shape[-1])):
        x_cdf = []
        for ens_idx in range(unc_preds.shape[0]):
            xs, ys = [], []
            targets = unc_preds[ens_idx, :, x_idx]
            for idx in np.argsort(targets):
                if len(xs) != 0 and targets[idx] <= xs[-1]:
                    continue
                xs.append(targets[idx])
                ys.append(taus[idx])
            intr = interp1d(
                xs, ys, kind="linear", fill_value=([0], [1]), bounds_error=False
            )
            x_cdf.append(intr(y_grid))
        x_cdf = np.asarray(x_cdf)
        avg_cdf = np.mean(x_cdf, axis=0)
        avg_cdfs.append(avg_cdf)
        t_idx = 0
        x_quants = []
        for idx in range(len(avg_cdf)):
            if t_idx >= len(taus):
                break
            if taus[t_idx] <= avg_cdf[idx]:
                x_quants.append(y_grid[idx])
                t_idx += 1
        while t_idx < len(taus):
            x_quants.append(y_grid[-1])
            t_idx += 1
        new_quants.append(x_quants)
    return np.asarray(new_quants).T


def get_ens_pred_conf_bound(unc_preds, taus, conf_level=0.95, score_distr="z"):
    """
    unc_preds 3D ndarray (ens_size, num_tau, num_x)
    where for each ens_member, each row corresonds to tau 0.01, 0.02...
    and the columns are for the set of x being predicted over.
    """
    num_ens, num_tau, num_x = unc_preds.shape
    len_tau = taus.size

    mean_pred = np.mean(unc_preds, axis=0)
    std_pred = np.std(unc_preds, axis=0, ddof=1)
    stderr_pred = std_pred / np.sqrt(num_ens)
    alpha = 1 - conf_level  # is (1-C)

    # determine coefficient
    if score_distr == "z":
        crit_value = norm_distr.ppf(1 - (0.5 * alpha))
    elif score_distr == "t":
        crit_value = t_distr.ppf(q=1 - (0.5 * alpha), df=(num_ens - 1))
    else:
        raise ValueError("score_distr must be one of z or t")

    gt_med = (taus > 0.5).reshape(-1, num_x)
    lt_med = ~gt_med
    assert gt_med.shape == mean_pred.shape == stderr_pred.shape
    out = (
        lt_med * (mean_pred - (float(crit_value) * stderr_pred))
        + gt_med * (mean_pred + (float(crit_value) * stderr_pred))
    ).T
    out = torch.from_numpy(out)
    return out


# New: small, isolated enhanced MLP that supports residuals, batch-norm, layer-norm and dropout.
class EnhancedMLP(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_layers,
        residual=False,
        batch_norm=False,
        layer_norm=False,
        dropout=0.0,
        activation="relu",
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >=1"
        self.num_layers = num_layers
        self.residual = residual
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.dropout_p = float(dropout) if dropout is not None else 0.0

        act_map = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
        }
        self.act = act_map.get(activation, nn.ReLU())

        # Build layers: first layer input -> hidden (if num_layers>1), then (num_layers-1) hidden layers, final linear to output
        self.hidden_size = hidden_size
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        if num_layers == 1:
            # single linear mapping input->output
            self.final = nn.Linear(input_size, output_size)
        else:
            # first hidden layer
            self.linears.append(nn.Linear(input_size, hidden_size))
            self.norms.append(
                nn.BatchNorm1d(hidden_size) if batch_norm else (nn.LayerNorm(hidden_size) if layer_norm else nn.Identity())
            )
            self.dropouts.append(nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity())

            # middle hidden layers (hidden->hidden)
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_size, hidden_size))
                self.norms.append(
                    nn.BatchNorm1d(hidden_size) if batch_norm else (nn.LayerNorm(hidden_size) if layer_norm else nn.Identity())
                )
                self.dropouts.append(nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity())

            # final projection from hidden to output
            self.final = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, features)
        if self.num_layers == 1:
            return self.final(x)
        h = x
        # first hidden
        h = self.linears[0](h)
        norm0 = self.norms[0]
        h = norm0(h)
        h = self.act(h)
        h = self.dropouts[0](h)

        # middle hidden layers
        for i in range(1, len(self.linears)):
            inp = h
            h = self.linears[i](h)
            norm = self.norms[i]
            h = norm(h)
            h = self.act(h)
            h = self.dropouts[i](h)
            if self.residual:
                # add residual only when shapes match (they should for hidden->hidden)
                if inp.shape == h.shape:
                    h = h + inp

        out = self.final(h)
        return out


class QModelEns(uq_model):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_layers,
        lr,
        wd,
        num_ens,
        device,
        # New options for layer-wise tricks:
        residual=False,
        batch_norm=False,
        layer_norm=False,
        dropout=0.0,
        activation="relu",
    ):

        self.num_ens = num_ens
        self.device = device
        # Instantiate EnhancedMLP for each ensemble member (isolated from QModelEns logic)
        self.model = [
            EnhancedMLP(
                input_size=input_size,
                output_size=output_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                residual=residual,
                batch_norm=batch_norm,
                layer_norm=layer_norm,
                dropout=dropout,
                activation=activation,
            ).to(device)
            for _ in range(num_ens)
        ]
        self.optimizers = [
            torch.optim.Adam(x.parameters(), lr=lr, weight_decay=wd)
            for x in self.model
        ]
        self.keep_training = [True for _ in range(num_ens)]
        self.best_va_loss = [np.inf for _ in range(num_ens)]
        self.best_va_model = [None for _ in range(num_ens)]
        self.best_va_ep = [0 for _ in range(num_ens)]
        self.done_training = False

    def use_device(self, device):
        self.device = device
        for idx in range(len(self.best_va_model)):
            if self.best_va_model[idx] is not None:
                self.best_va_model[idx] = self.best_va_model[idx].to(device)
        for idx in range(len(self.model)):
            self.model[idx] = self.model[idx].to(device)

        if device.type == "cuda":
            for idx in range(len(self.best_va_model)):
                assert self.best_va_model[idx] is None or next(self.best_va_model[idx].parameters()).is_cuda
            for idx in range(len(self.model)):
                assert next(self.model[idx].parameters()).is_cuda

    def print_device(self):
        device_list = []
        for idx in range(len(self.best_va_model)):
            if next(self.best_va_model[idx].parameters()).is_cuda:
                device_list.append("cuda")
            else:
                device_list.append("cpu")
        print(device_list)

    def loss(self, loss_fn, x, y, q_list, batch_q, take_step, args):
        ens_loss = []
        for idx in range(self.num_ens):
            self.optimizers[idx].zero_grad()
            if self.keep_training[idx]:
                if batch_q:
                    loss = loss_fn(
                        self.model[idx], y, x, q_list, self.device, args
                    )
                else:
                    loss = gather_loss_per_q(
                        loss_fn,
                        self.model[idx],
                        y,
                        x,
                        q_list,
                        self.device,
                        args,
                    )
                ens_loss.append(loss.item())

                if take_step:
                    loss.backward()
                    self.optimizers[idx].step()
            else:
                ens_loss.append(np.nan)

        return np.asarray(ens_loss)

    def loss_boot(
        self, loss_fn, x_list, y_list, q_list, batch_q, take_step, args
    ):
        ens_loss = []
        for idx in range(self.num_ens):
            self.optimizers[idx].zero_grad()
            if self.keep_training[idx]:
                if batch_q:
                    loss = loss_fn(
                        self.model[idx],
                        y_list[idx],
                        x_list[idx],
                        q_list,
                        self.device,
                        args,
                    )
                else:
                    loss = gather_loss_per_q(
                        loss_fn,
                        self.model[idx],
                        y_list[idx],
                        x_list[idx],
                        q_list,
                        self.device,
                        args,
                    )
                ens_loss.append(loss.item())

                if take_step:
                    loss.backward()
                    self.optimizers[idx].step()
            else:
                ens_loss.append(np.nan)

        return np.asarray(ens_loss)

    def update_va_loss(
        self, loss_fn, x, y, q_list, batch_q, curr_ep, num_wait, args
    ):
        with torch.no_grad():
            va_loss = self.loss(
                loss_fn, x, y, q_list, batch_q, take_step=False, args=args
            )

        for idx in range(self.num_ens):
            if self.keep_training[idx]:
                if va_loss[idx] < self.best_va_loss[idx]:
                    self.best_va_loss[idx] = va_loss[idx]
                    self.best_va_ep[idx] = curr_ep
                    self.best_va_model[idx] = deepcopy(self.model[idx])
                else:
                    if curr_ep - self.best_va_ep[idx] > num_wait:
                        print(
                            "Val loss stagnate for {}, model {}".format(
                                num_wait, idx
                            )
                        )
                        print("EP {}".format(curr_ep))
                        self.keep_training[idx] = False

        if not any(self.keep_training):
            self.done_training = True

        return va_loss

    #####
    def predict(
        self,
        cdf_in,
        conf_level=0.95,
        score_distr="z",
        recal_model=None,
        recal_type=None,
    ):
        """
        Only pass in cdf_in into model and return output
        If self is an ensemble, return a conservative output based on conf_bound
        specified by conf_level

        :param cdf_in: tensor [x, p], of size (num_x, dim_x + 1)
        :param conf_level: confidence level for ensemble prediction
        :param score_distr: 'z' or 't' for confidence bound coefficient
        :param recal_model:
        :param recal_type:
        :return:
        """

        if self.num_ens == 1:
            with torch.no_grad():
                pred = self.model[0](cdf_in)
        if self.num_ens > 1:
            pred_list = []
            for m in self.model:
                with torch.no_grad():
                    pred_list.append(m(cdf_in).T.unsqueeze(0))

            unc_preds = (
                torch.cat(pred_list, dim=0).detach().cpu().numpy()
            )  # shape (num_ens, num_x, 1)
            taus = cdf_in[:, -1].flatten().cpu().numpy()
            pred = get_ens_pred_conf_bound(
                unc_preds, taus, conf_level=0.95, score_distr="z"
            )
            pred = pred.to(cdf_in.device)

        return pred

    #####

    def predict_q(
        self,
        x,
        q_list=None,
        ens_pred_type="conf",
        recal_model=None,
        recal_type=None,
    ):
        """
        Get output for given list of quantiles

        :param x: tensor, of size (num_x, dim_x)
        :param q_list: flat tensor of quantiles, if None, is set to [0.01, ..., 0.99]
        :param ens_pred_type:
        :param recal_model:
        :param recal_type:
        :return:
        """

        if q_list is None:
            q_list = torch.arange(0.01, 0.99, 0.01)
        else:
            q_list = q_list.flatten()

        if self.num_ens > 1:
            # choose function to make ens predictions
            if ens_pred_type == "conf":
                ens_pred_fn = get_ens_pred_conf_bound
            elif ens_pred_type == "interp":
                ens_pred_fn = get_ens_pred_interp
            else:
                raise ValueError("ens_pred_type must be one of conf or interp")

        num_x = x.shape[0]
        num_q = q_list.shape[0]

        cdf_preds = []
        for p in q_list:
            if recal_model is not None:
                if recal_type == "torch":
                    recal_model.cpu()  # keep recal model on cpu
                    with torch.no_grad():
                        in_p = recal_model(p.reshape(1, -1)).item()
                elif recal_type == "sklearn":
                    in_p = float(recal_model.predict(p.flatten()))
                else:
                    raise ValueError("recal_type incorrect")
            else:
                in_p = float(p)
            p_tensor = (in_p * torch.ones(num_x)).reshape(-1, 1).to(self.device)
            cdf_in = torch.cat([x, p_tensor], dim=1).to(self.device)
            cdf_pred = self.predict(cdf_in)  # shape (num_x, 1)
            cdf_preds.append(cdf_pred)

        pred_mat = torch.cat(cdf_preds, dim=1)  # shape (num_x, num_q)
        assert pred_mat.shape == (num_x, num_q)
        return pred_mat
    
    def predict_q(
        self,
        x,
        q_list=None,
        ens_pred_type="conf",
        recal_model=None,
        recal_type=None,
    ):
        """
        Get output for given list of quantiles. (Vectorized Version)

        :param x: tensor, of size (num_x, dim_x)
        :param q_list: flat tensor of quantiles, if None, is set to [0.01, ..., 0.99]
        :param ens_pred_type:
        :param recal_model:
        :param recal_type:
        :return:
        """
        num_x = x.shape[0]
        
        if q_list is None:
            # Ensure q_list is on the correct device from the start
            q_list = torch.arange(0.01, 1.00, 0.01, device=self.device, dtype=x.dtype)
        else:
            q_list = q_list.flatten().to(self.device, dtype=x.dtype)
        
        num_q = q_list.shape[0]

        if recal_model is not None:
            if recal_type == "torch":
                recal_model.to(q_list.device)
                with torch.no_grad():
                    in_q_list = recal_model(q_list.view(-1, 1)).flatten()
            elif recal_type == "sklearn":
                q_numpy = q_list.cpu().numpy().reshape(-1, 1)
                in_q_numpy = recal_model.predict(q_numpy)
                in_q_list = torch.from_numpy(in_q_numpy).to(self.device).flatten().to(x.dtype)
            else:
                raise ValueError("recal_type incorrect")
        else:
            in_q_list = q_list

        x_expanded = x.repeat_interleave(num_q, dim=0)
        p_expanded = in_q_list.repeat(num_x).view(-1, 1)
        cdf_in_batch = torch.cat([x_expanded, p_expanded], dim=1)
        all_preds = self.predict(cdf_in_batch)
        pred_mat = all_preds.view(num_x, num_q)

        assert pred_mat.shape == (num_x, num_q)
        return pred_mat


if __name__ == "__main__":
    temp_model = QModelEns(
        input_size=1,
        output_size=1,
        hidden_size=10,
        num_layers=2,
        lr=0.01,
        wd=0.0,
        num_ens=5,
        device=torch.device("cuda:0"),
    )
