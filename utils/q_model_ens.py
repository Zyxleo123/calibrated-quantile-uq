import os, sys
from copy import deepcopy
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MixtureSameFamily, Distribution, constraints, CumulativeDistributionTransform, TransformedDistribution, Uniform
from pyro.distributions import Logistic

NUM_PARTS = 100
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
        self.waiting = [True for _ in range(num_ens)]
        self.best_va_loss = [np.inf for _ in range(num_ens)]
        self.best_va_model = [None for _ in range(num_ens)]
        self.best_va_ep = [0 for _ in range(num_ens)]

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

            if take_step:
                loss.backward()
                self.optimizers[idx].step()
            ens_loss.append(loss.detach())

        return torch.tensor(ens_loss, device=self.device)

    def loss_boot(
        self, loss_fn, x_list, y_list, q_list, batch_q, take_step, args
    ):
        ens_loss = []
        for idx in range(self.num_ens):
            self.optimizers[idx].zero_grad()
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

            if take_step:
                loss.backward()
                self.optimizers[idx].step()
            ens_loss.append(loss.detach())

        return torch.tensor(ens_loss, device=self.device)

    def update_va_loss(
        self, loss_fn, x, y, q_list, batch_q, curr_ep, num_wait, args
    ):
        with torch.no_grad():
            num_parts = 20
            size_parts = len(x) // num_parts
            loss_parts = []
            for i in range(num_parts):
                if i != num_parts - 1:
                    loss_part = self.loss(
                        loss_fn,
                        x[i * size_parts : (i + 1) * size_parts],
                        y[i * size_parts : (i + 1) * size_parts],
                        q_list,
                        batch_q,
                        take_step=False,
                        args=args,
                    )
                else:
                    loss_part = self.loss(
                        loss_fn,
                        x[i * size_parts :],
                        y[i * size_parts :],
                        q_list,
                        batch_q,
                        take_step=False,
                        args=args,
                    )
                loss_parts.append(loss_part)
            va_loss = torch.mean(torch.stack(loss_parts), dim=0).cpu().numpy()

        for idx in range(self.num_ens):
            if self.waiting[idx]:
                if va_loss[idx] < self.best_va_loss[idx]:
                    self.best_va_loss[idx] = va_loss[idx]
                    self.best_va_ep[idx] = curr_ep
                    self.best_va_model[idx] = deepcopy(self.model[idx])
                else:
                    if curr_ep - self.best_va_ep[idx] > num_wait:
                        self.waiting[idx] = False
                        print(f"Early stopping at ep {curr_ep}")
        return va_loss


    #####
    def predict(
        self,
        cdf_in,
        in_batch=True,
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
                if in_batch:
                    num_parts = NUM_PARTS
                    size_parts = len(cdf_in) // num_parts
                    # inference in parts to save memory
                    for i in range(num_parts):
                        if i == 0:
                            all_preds = self.model[0](cdf_in[i*size_parts:(i+1)*size_parts])
                        elif i == num_parts - 1:
                            part_preds = self.model[0](cdf_in[i*size_parts:])
                            all_preds = torch.cat([all_preds, part_preds], dim=0)
                        else:
                            part_preds = self.model[0](cdf_in[i*size_parts:(i+1)*size_parts])
                            all_preds = torch.cat([all_preds, part_preds], dim=0)
                else:
                    all_preds = self.model[0](cdf_in)
            pred = all_preds
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

        with torch.no_grad():
            num_parts = NUM_PARTS
            size_parts = len(cdf_in_batch) // num_parts
            # inference in parts to save memory
            for i in range(num_parts):
                if i == 0:
                    all_preds = self.predict(cdf_in_batch[i*size_parts:(i+1)*size_parts], in_batch=False)
                elif i == num_parts - 1:
                    part_preds = self.predict(cdf_in_batch[i*size_parts:])
                    all_preds = torch.cat([all_preds, part_preds], dim=0)
                else:
                    part_preds = self.predict(cdf_in_batch[i*size_parts:(i+1)*size_parts], in_batch=False)
                    all_preds = torch.cat([all_preds, part_preds], dim=0)
        pred_mat = all_preds.view(num_x, num_q)

        assert pred_mat.shape == (num_x, num_q)
        return pred_mat

def _logistic_pdf(x, mu, s):
    t = (x - mu) / s
    e = torch.exp(-t)
    return e / (s * (1.0 + e) ** 2 + 1e-12)

def _logistic_cdf(x, mu, s):
    t = (x - mu) / s
    return 1.0 / (1.0 + torch.exp(-t))


# def _kde_refl_log_density(z, centers, b):
#     B = centers.shape[0]
#     s = b * (float(B) ** (-1.0 / 5.0))
#     s = max(s, 1e-3)
#     z = z.view(-1, 1)
#     c = centers.view(1, -1)
#     main = _logistic_pdf(z, c, s)
#     left = _logistic_pdf(-z, c, s)
#     right = _logistic_pdf(2.0 - z, c, s)
#     dens = (main + left + right).mean(dim=1)
#     return torch.log(dens + 1e-12)

def icdf_from_cdf(dist, alpha, epsilon=1e-5, warn_precision=4e-3, low=None, high=None):
    """
    Compute the quantiles of a distribution using binary search, in a vectorized way.
    """

    alpha = adjust_unit_tensor(alpha)
    alpha, _ = torch.broadcast_tensors(alpha, torch.zeros(dist.batch_shape))
    # Expand to the left and right until we are sure that the quantile is in the interval
    expansion_factor = 4
    if low is None:
        low = torch.full(alpha.shape, -1.0)
        while (mask := dist.cdf(low) > alpha + epsilon).any():
            low[mask] *= expansion_factor
    else:
        low = low.clone()
    if high is None:
        high = torch.full(alpha.shape, 1.0)
        while (mask := dist.cdf(high) < alpha - epsilon).any():
            high[mask] *= expansion_factor
    else:
        high = high.clone()
    low, high, _ = torch.broadcast_tensors(low, high, torch.zeros(alpha.shape))
    assert dist.cdf(low).shape == alpha.shape

    # Binary search
    prev_precision = None
    while True:
        # To avoid "UserWarning: Use of index_put_ on expanded tensors is deprecated".
        low = low.clone()
        high = high.clone()
        precision = (high - low).max()
        # Stop if we have enough precision
        if precision < 1e-5:
            break
        # Stop if we can not improve the precision anymore
        if prev_precision is not None and precision >= prev_precision:
            break
        mid = (low + high) / 2
        mask = dist.cdf(mid) < alpha
        low[mask] = mid[mask]
        high[~mask] = mid[~mask]
        prev_precision = precision

    if precision > warn_precision:
        pass
        # log.warn(f'Imprecise quantile computation with precision {precision}')
    return low

def adjust_tensor(x, a=0.0, b=1.0, *, epsilon=1e-4):
    # We accept that, due to rounding errors, x is not in the interval up to epsilon
    mask = (a - epsilon <= x) & (x <= b + epsilon)
    assert mask.all(), (x[~mask], a, b)
    return x.clamp(a, b)

def adjust_unit_tensor(x, epsilon=1e-4):
    return adjust_tensor(x, a=0.0, b=1.0, epsilon=epsilon)

class ReflectedDist(Distribution):
    support = constraints.real
    has_rsample = False

    def __init__(self, dist, a=-torch.inf, b=torch.inf):
        self.dist = dist
        self.a = a
        self.b = b

    @property
    def batch_shape(self):
        return self.dist._batch_shape

    def cdf(self, value):
        value = adjust_tensor(value, self.a, self.b)
        vb = self.dist.cdf(2 * self.b - value)
        va = self.dist.cdf(2 * self.a - value)
        v = self.dist.cdf(value)
        # Beware that the CDF of MixtureSameFamily can be slightly outside [0, 1] due to precision errors.
        vb = adjust_tensor(vb, self.a, self.b)
        va = adjust_tensor(va, self.a, self.b)
        v = adjust_tensor(v, self.a, self.b)
        res = 1 - vb + v - va
        assert (vb <= 1).all() and (0 <= va).all() and (0 <= v).all() and (v <= 1).all(), f'{vb.max()}, {va.min()}, {v.min()}, {v.max()}'
        res[value < self.a] = 0
        res[self.b < value] = 1
        assert (0 <= res).all() and (res <= 1).all(), f'{res.min()}, {res.max()}'
        return res

    def icdf(self, value):
        value = adjust_unit_tensor(value)
        return icdf_from_cdf(self, value, low=self.a, high=self.b)

    def log_prob(self, value):
        value = adjust_tensor(value, self.a, self.b)

        # The code below is a more stable alternative to the following:
        # res = (
        #     self.dist.log_prob(2 * self.b - value).exp()
        #     + self.dist.log_prob(value).exp()
        #     + self.dist.log_prob(2 * self.a - value).exp()
        # ).log()

        log_probs = torch.stack([
            self.dist.log_prob(2 * self.b - value),
            self.dist.log_prob(value),
            self.dist.log_prob(2 * self.a - value)
        ], dim=-1)
        res = torch.logsumexp(log_probs, dim=-1)

        # .clone() is needed to avoid "RuntimeError: one of the variables needed for
        # gradient computation has been modified by an inplace operation."
        res = res.clone()
        res[value < self.a] = -torch.inf
        res[self.b < value] = -torch.inf
        return res

    def sample(self, sample_shape=torch.Size()):
        shape = torch.Size(sample_shape) + self.dist.batch_shape
        rand = torch.rand(shape, device=self.a.device)
        return self.icdf(rand)
    
class MixtureDist(MixtureSameFamily):
    def __init__(self, component_dist_class, means, stds, *, probs=None, logits=None):
        mix_dist = Categorical(probs=probs, logits=logits)
        self.component_dist_class = component_dist_class
        component_dist = self.component_dist_class(means, stds)
        super().__init__(mix_dist, component_dist)

    def icdf(self, value):
        return icdf_from_cdf(self, value)

    def affine_transform(self, loc, scale):
        """
        Let $X ~ Dist(\mu, \sigma)$. Then $a + bX ~ Dist(a + b \mu, b \sigma)$.
        The reasoning is similar for a mixture.
        """
        component_dist = self.component_distribution
        mix_dist = self.mixture_distribution
        means, stds = component_dist.loc, component_dist.scale
        means = loc + scale * means
        stds = scale * stds
        return type(self)(means, stds, logits=mix_dist.logits)

    def unnormalize(self, scaler):
        return self.affine_transform(scaler.mean_, scaler.scale_)

    def normalize(self, scaler):
        return self.affine_transform(-scaler.mean_ / scaler.scale_, 1.0 / scaler.scale_)

    def rsample(self, sample_shape=torch.Size(), tau=1):
        """
        Returns:
            Tensor: A tensor of shape `[batch_size, n_samples]`
        """
        raise NotImplementedError()

class LogisticMixtureDist(MixtureDist):
    def __init__(self, *args, base_module=None, **kwargs):
        super().__init__(Logistic, *args, **kwargs)
        self.base_module = base_module

    def log_sigmoid(x):
        return -torch.where(x > 0, torch.log(1 + torch.exp(-x)), x + torch.log(1 + torch.exp(-x)))
    
    def log_prob(self, x):
        # with elapsed_timer() as time:
        res = super().log_prob(x)
        # if self.base_module is not None:
        #     self.base_module.advance_timer('logistic_log_prob_time', time())
        return res

class SmoothEmpiricalCDF(CumulativeDistributionTransform):
    def __init__(
        self,
        x,
        device, 
        b=0.1,
        epoch=None,
        batch_idx=None,
        base_module=None,
        **kwargs,
    ):
        self.epoch = epoch
        self.batch_idx = batch_idx
        assert x.dim() == 1
        N = x.shape[0]
        dist = LogisticMixtureDist(x, torch.tensor(b * N ** (-1 / 5)).to(device), probs=torch.ones_like(x).to(device), base_module=base_module)
        dist = ReflectedDist(dist, torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))
        super().__init__(dist, **kwargs)

class UnitUniform(Uniform):
    def __init__(self, batch_shape, device, *args, **kwargs):
        super().__init__(
            torch.zeros(batch_shape).to(device), torch.ones(batch_shape).to(device), *args, **kwargs
        )

    def log_prob(self, value):
        # Workaround because values of 0 and 1 are not allowed
        eps = 1e-7
        value[value == 0.0] += eps
        value[value == 1.0] -= eps
        return super().log_prob(value)

class RecalibratedDist(TransformedDistribution):
    def __init__(self, dist, posthoc_model, device):
        base_dist = UnitUniform(dist.batch_shape, device)
        transforms = [
            posthoc_model.inv,
            CumulativeDistributionTransform(dist).inv,
        ]
        super().__init__(base_dist, transforms)

def _kde_refl_log_density(z, centers, b):
    B = centers.shape[0]
    s = b * (float(B) ** (-1.0 / 5.0))
    s = max(s, 1e-3)

    z = z.view(-1, 1)
    c = centers.view(1, -1)

    main = _logistic_pdf(z, c, s)
    left = _logistic_pdf(-z, c, s)
    right = _logistic_pdf(2.0 - z, c, s)

    dens = (main + left + right).mean(dim=1)
    log_dens = torch.log(dens + 1e-12)

    mask = (z.squeeze(-1) >= 0) & (z.squeeze(-1) <= 1)
    return torch.where(mask, log_dens, torch.zeros_like(log_dens))


class QRTNormalVanilla(nn.Module):
    """
    Single NNKit.vanilla_nn that outputs two values per x: (mean, rho).
    """
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        # One network with 2-dimensional output
        self.net = EnhancedMLP(
            input_size=input_size,
            output_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            residual=True
        )

    def forward(self, x):
        out = self.net(x)
        if out.dim() == 1:
            # unlikely, but guard
            out = out.view(-1, 2)
        mean = out[:, 0]
        rho = out[:, 1]
        std = F.softplus(rho) + 1e-3
        return mean, std


class QRTNormalAdapterVanilla:
    def __init__(self, input_size, hidden_size, num_layers, lr, wd, device, qrt_alpha=1.0, kde_b=0.1, x_cal=None, y_cal=None):
        self.device = device
        self.model = QRTNormalVanilla(input_size, hidden_size, num_layers).to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        self.done_training = False
        self.keep_training = True
        self.waiting = True
        self.best_va_loss = np.inf
        self.best_va_model = None
        self.best_va_ep = 0
        self.qrt_alpha = float(qrt_alpha)
        self.kde_b = float(kde_b)
        if x_cal is not None:
            self.x_cal = x_cal.to(device)
            self.y_cal = y_cal.to(device)
        else:
            self.x_cal, self.y_cal = None, None

    def use_device(self, device):
        self.device = device
        if self.best_va_model is not None:
            self.best_va_model = self.best_va_model.to(device)
        if self.x_cal is not None:
            self.x_cal = self.x_cal.to(device)
            self.y_cal = self.y_cal.to(device)
        self.model = self.model.to(device)

    def _dist(self, x):
        mean, std = self.model(x)
        return torch.distributions.Normal(mean, std)

    def loss(self, loss_fn, x, y, q_list, batch_q, take_step, args):
        x = x.to(self.device)
        y = y.to(self.device).squeeze(-1)
        dist = self._dist(x)
        pits = dist.cdf(y).clamp(1e-6, 1 - 1e-6)
        log_phi = _kde_refl_log_density(pits, pits, b=self.kde_b)
        loss = -(dist.log_prob(y) + self.qrt_alpha * log_phi).mean()
        if take_step:
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
        return loss.detach().unsqueeze(0)

    def predict(self, cdf_in, conf_level=0.95, score_distr="z", recal_model=None, recal_type=None):
        # assert False, "predict shouldnt be called..."
        x = cdf_in[:, :-1]
        q = cdf_in[:, -1].clamp(1e-6, 1 - 1e-6)
        with torch.no_grad():
            dist = self._dist(x)
            if recal_model is not None:
                if recal_type == "torch":
                    recal_model.to(q.device)
                    q = recal_model(q.view(-1, 1)).flatten()
                elif recal_type == "sklearn":
                    q_np = q.detach().cpu().numpy().reshape(-1, 1)
                    q = torch.from_numpy(recal_model.predict(q_np)).to(q.device).flatten()
                else:
                    raise ValueError("recal_type incorrect")
            if self.x_cal is not None:
                dist_cal = self._dist(self.x_cal)
                z_cal = dist_cal.cdf(self.y_cal.squeeze(-1))
                dist = RecalibratedDist(dist, SmoothEmpiricalCDF(z_cal, self.device), self.device)
            yq = dist.icdf(q).view(-1, 1)
        return yq

    def predict_q(self, x, q_list=None, ens_pred_type="conf", recal_model=None, recal_type=None):
        x = x.to(self.device)
        with torch.no_grad():
            # dist = self._dist(x)
            if q_list is None:
                q_list = torch.arange(0.01, 1.00, 0.01, device=x.device, dtype=x.dtype)
            else:
                q_list = q_list.flatten().to(x.device, dtype=x.dtype)
            if recal_model is not None:
                if recal_type == "torch":
                    recal_model.to(q_list.device)
                    q_in = recal_model(q_list.view(-1, 1)).flatten()
                elif recal_type == "sklearn":
                    q_np = q_list.detach().cpu().numpy().reshape(-1, 1)
                    q_in = torch.from_numpy(recal_model.predict(q_np)).to(x.device).flatten().to(x.dtype)
                else:
                    raise ValueError("recal_type incorrect")
            else:
                q_in = q_list

            q_in = q_in.clamp(1e-6, 1 - 1e-6)
            q_expanded = q_in.repeat(x.size(0)).view(-1, 1)
            x_expanded = x.repeat_interleave(q_in.size(0), dim=0)
            dist_expanded = self._dist(x_expanded)

            if self.x_cal is not None:
                dist_cal = self._dist(self.x_cal)
                z_cal = dist_cal.cdf(self.y_cal.squeeze(-1))
                dist_expanded = RecalibratedDist(dist_expanded, SmoothEmpiricalCDF(z_cal, self.device), self.device)
                # dist_expanded = TransformedDistribution(dist_expanded, transforms=[transform])
            yq = dist_expanded.icdf(q_expanded.view(-1)).view(x.size(0), q_in.size(0))
        return yq
    
    def update_va_loss(
        self, loss_fn, x, y, q_list, batch_q, curr_ep, num_wait, args
    ):
        if self.waiting:
            # compute in parts to save memory
            with torch.no_grad():
                num_parts = 20
                size_parts = len(x) // num_parts
                loss_parts = []
                for i in range(num_parts):
                    if i != num_parts - 1:
                        loss_part = self.loss(
                            loss_fn,
                            x[i * size_parts : (i + 1) * size_parts],
                            y[i * size_parts : (i + 1) * size_parts],
                            q_list,
                            batch_q,
                            take_step=False,
                            args=args,
                        )
                    else:
                        loss_part = self.loss(
                            loss_fn,
                            x[i * size_parts :],
                            y[i * size_parts :],
                            q_list,
                            batch_q,
                            take_step=False,
                            args=args,
                        )
                    loss_parts.append(loss_part)
                va_loss = torch.mean(torch.stack(loss_parts), dim=0).cpu().numpy()


        if self.waiting:
            if va_loss < self.best_va_loss:
                self.best_va_loss = va_loss
                self.best_va_ep = curr_ep
                self.best_va_model = deepcopy(self.model)
            else:
                if curr_ep - self.best_va_ep > num_wait:
                    self.waiting = False
                    print(f"Early stopping at ep {curr_ep}")

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
