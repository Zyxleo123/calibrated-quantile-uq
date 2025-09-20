import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR
from pathlib import Path
import copy
import traceback
import os
import contextlib
from sklearn.isotonic import IsotonicRegression
from typing import Union
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class Interp1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.

        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.

        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                        'at most 2-D.'
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, 'All parameters must be on the same device.'
        device = device[0]

        # Checking for the dimensions
        assert (v['x'].shape[1] == v['y'].shape[1]
                and (
                     v['x'].shape[0] == v['y'].shape[0]
                     or v['x'].shape[0] == 1
                     or v['y'].shape[0] == 1
                    )
                ), ("x and y must have the same number of columns, and either "
                    "the same number of row or one of them having only one "
                    "row.")

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
           and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (D, v['xnew'].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0]*shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v['xnew'].shape[0] == 1:
            v['xnew'] = v['xnew'].expand(v['x'].shape[0], -1)

        # the squeeze is because torch.searchsorted does accept either a nd with
        # matching shapes for x and xnew or a 1d vector for x. Here we would
        # have (1,len) for x sometimes
        torch.searchsorted(v['x'].contiguous().squeeze(),
                           v['xnew'].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ['x', 'y', 'xnew']:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [None, ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat['slopes'] = is_flat['x']
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v['slopes'] = (
                    (v['y'][:, 1:]-v['y'][:, :-1])
                    /
                    (eps + (v['x'][:, 1:]-v['x'][:, :-1]))
                )

            # now build the linear interpolation
            ynew = sel('y') + sel('slopes')*(
                                    v['xnew'] - sel('x'))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = torch.autograd.grad(
                        ctx.saved_tensors[0],
                        [i for i in inputs if i is not None],
                        grad_out, retain_graph=True)
        result = [None, ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)

interp1d = Interp1d.apply

##########################################################################

## LOWER QUANTILE MODEL
class quantile_model(nn.Module):
    def __init__(self, X, Y, vanilla_model, quantile):
        super().__init__()
        self.X = X
        self.Y = Y
        self.quantile = quantile
        self.vanilla_model = vanilla_model

    def forward(self, x, x_cal = None, y_cal = None):

        vanilla_model = self.vanilla_model(x)

        if x_cal is None and y_cal is None:
            y_cal = self.Y
            vanilla_model_Xcal = self.vanilla_model(self.X)
        else:
            vanilla_model_Xcal = self.vanilla_model(x_cal)

        #c_lb enforces calibration during training on calibration dataset
        c_lb = torch.quantile(y_cal - vanilla_model_Xcal, self.quantile, interpolation='linear')

        zero_quantile = c_lb + vanilla_model

        return zero_quantile

## SPECIFY CALIBRATED QUANTILE MODEL ENSEMBLE
class quantile_model_ensemble(nn.Module):
    def __init__(self, X, Y, vanilla_model, half_q_levels: Union[int, torch.Tensor], output_device, vanilla_weights = None, avg_weights=False):
        super().__init__()
        self.done_training = False
        self.output_device = output_device
        self.upper_quantile_models = nn.ModuleList()
        self.lower_quantile_models = nn.ModuleList()
        self.X = X
        self.Y = Y
        self.iso_reg = None
        self.printname = 'quantile_model'
        self.printcolor = 'C0'

        ## check if number of quantile levels is divisible by two
        if isinstance(half_q_levels, int):
            assert int((half_q_levels+1)/2) == (half_q_levels+1)/2
            quantile_model_levels = range(int((half_q_levels+1)/2))
        else:
            quantile_model_levels = half_q_levels

        self.half_q_levels = quantile_model_levels

        length_data = X.shape[0]

        for i in range(len(quantile_model_levels)):
            vanilla_quantile_model_lower = vanilla_model(nfeatures = X.shape[1])
            vanilla_quantile_model_upper = vanilla_model(nfeatures = X.shape[1])
            if isinstance(vanilla_weights, list):
                weights = vanilla_weights[i]
            else:
                weights = vanilla_weights
            if weights is not None:
                # Initialize weights from normal regression on training set
                lower_weights = copy.deepcopy(weights)#torch.load(path, weights_only=True, map_location=output_device)
                upper_weights = copy.deepcopy(weights)#copy.deepcopy(lower_weights)
                if avg_weights:
                    for key in lower_weights:
                        upper_weights[key] = (lower_weights[key].to(output_device) + vanilla_quantile_model_upper.state_dict()[key].to(output_device)) / 2
                        lower_weights[key] = (lower_weights[key].to(output_device) + vanilla_quantile_model_lower.state_dict()[key].to(output_device)) / 2
                vanilla_quantile_model_lower.load_state_dict(lower_weights)
                vanilla_quantile_model_upper.load_state_dict(upper_weights)
            q_model_upper = quantile_model(X=X, Y=Y, vanilla_model=vanilla_quantile_model_upper, quantile = 1)
            q_model_lower = quantile_model(X=X, Y=Y, vanilla_model=vanilla_quantile_model_lower, quantile = 0)
            self.upper_quantile_models.append(q_model_upper)
            self.lower_quantile_models.append(q_model_lower)

        self.to(output_device)


    def forward(self, x, quantile = []):

        half_q_levels = self.half_q_levels
        output_device = self.output_device
        n_quantile_models = len(half_q_levels)


        quantiles_lower = []
        quantiles_upper = []

        x_cal = self.X
        y_cal = self.Y

        interval_tot = 1
        ind_kept = torch.ones_like(y_cal, dtype=torch.bool)

        for n_mod in range(n_quantile_models):
            upper_quantile_model = self.upper_quantile_models[n_mod]
            lower_quantile_model = self.lower_quantile_models[n_mod]

            lower_quantile = lower_quantile_model(x, x_cal = x_cal, y_cal = y_cal)
            upper_quantile = upper_quantile_model(x, x_cal = x_cal, y_cal = y_cal)

            # USED EXCLUSIVELY TO COMPUTE CALIBRATING ELEMENTS
            lower_quantile_cal = lower_quantile_model(x_cal, x_cal=x_cal, y_cal=y_cal)
            upper_quantile_cal = upper_quantile_model(x_cal, x_cal=x_cal, y_cal=y_cal)


            # IN TRAINING, WE ONLY OPTIMIZE SHARPNESS FOR ENTRIES CORRESPONDING TO x_cal AND y_cal
            if self.training:
                # ALL ENTRIES THAT DO NOT CORRESPOND TO x_cal AND y_cal ARE REPLACED WITH DUMMY VALUES THAT DO NOT
                # AFFECT OPTIMIZATION
                # index_x = np.all([np.in1d(x.T[i].cpu(), x_cal.T[i].cpu()) for i in range(x.shape[1])], 0)
                index_x = np.all([np.isin(x.T[i].cpu(), x_cal.T[i].cpu()) for i in range(x.shape[1])], 0) #in1d
                lower_quantile[index_x.__invert__()] = 0
                upper_quantile[index_x.__invert__()] = 0
            else:
                if n_mod==0:
                    one_quantile = upper_quantile
                    one_quantile_cal = upper_quantile_cal
                else:
                    # Apply maximum and minimum operations to inner models as described in the paper
                    lower_quantile = torch.maximum(lower_quantile, previous_lower_quantile + delta_lower * (
                                previous_upper_quantile - previous_lower_quantile)).to(self.output_device)
                    upper_quantile = torch.minimum(upper_quantile, previous_lower_quantile + delta_upper * (
                                previous_upper_quantile - previous_lower_quantile)).to(self.output_device)
                    lower_quantile = torch.minimum(lower_quantile, one_quantile).to(self.output_device)

                    lower_quantile_cal = torch.maximum(lower_quantile_cal, previous_lower_quantile_cal[ind_kept] + delta_lower * (
                                previous_upper_quantile_cal[ind_kept] - previous_lower_quantile_cal[ind_kept])).to(self.output_device)
                    upper_quantile_cal = torch.minimum(upper_quantile_cal, previous_lower_quantile_cal[ind_kept] + delta_upper * (
                                previous_upper_quantile_cal[ind_kept] - previous_lower_quantile_cal[ind_kept])).to(self.output_device)
                    lower_quantile_cal = torch.minimum(lower_quantile_cal, one_quantile_cal[ind_kept]).to(self.output_device)

                upper_quantile = torch.maximum(lower_quantile, upper_quantile).to(self.output_device)
                previous_lower_quantile = lower_quantile
                previous_upper_quantile = upper_quantile
                previous_lower_quantile_cal = lower_quantile_cal
                previous_upper_quantile_cal = upper_quantile_cal
                one_quantile_cal = one_quantile_cal[ind_kept.squeeze()]

            quantiles_lower.append(lower_quantile)
            quantiles_upper.append(upper_quantile)

            if n_mod<n_quantile_models-1:
                delta_upper_lower = upper_quantile_cal - lower_quantile_cal
                delta_upper_lower = torch.maximum(delta_upper_lower, torch.tensor(1e-24))
                delta_quantile_level = (half_q_levels[n_mod+1] - half_q_levels[n_mod]).to(self.output_device)
                delta_upper = torch.quantile((y_cal - lower_quantile_cal)/delta_upper_lower, 1 - delta_quantile_level/interval_tot)
                delta_lower = torch.quantile((y_cal - lower_quantile_cal) / delta_upper_lower, delta_quantile_level/interval_tot)
                ind_kept = ((y_cal < lower_quantile_cal + delta_upper*(delta_upper_lower)) & (
                        y_cal > lower_quantile_cal + delta_lower*(delta_upper_lower))).reshape(y_cal.shape[0], )
                if not ind_kept.any() and n_mod < n_quantile_models-1:
                    STOP_FOR_DEBUGGING = 1
                x_cal = x_cal[ind_kept]
                y_cal = y_cal[ind_kept]


                interval_tot = 1 - 2*half_q_levels[n_mod+1]

        quantiles_upper.reverse()
        quantiles = torch.hstack([torch.hstack(quantiles_lower).to(output_device), torch.hstack(quantiles_upper).to(output_device)]).to(output_device)
        if quantile == []:
            return quantiles
        else:
            if not torch.is_tensor(quantile):
                quantile = torch.tensor(quantile)
            quantile = quantile.to(self.output_device)
            total_q_levels = torch.hstack([half_q_levels, 1- half_q_levels.flip(0) ]).to(self.output_device)
            return interp1d(total_q_levels.repeat(x.shape[0],1), quantiles, quantile)

    def get_quantiles(self, X, q_levels):
        if self.iso_reg is None:
            return self(X, q_levels)
        else:
            length_data = self.X.shape[0]
            quantile_linspace = torch.linspace(1/(length_data+1), length_data/(length_data+1), length_data)
            quantile_linspace = torch.linspace(0, 1, length_data)
            uncalibrated_quantiles = self(X, quantile_linspace.to(self.output_device))
            recalibrated_quantile_vals = self.iso_reg.predict(quantile_linspace.numpy())
            recalibrated_quantile_vals = torch.tensor(recalibrated_quantile_vals).to(self.output_device)
            if not torch.is_tensor(q_levels):
                q_levels = torch.tensor(q_levels)
            return interp1d(recalibrated_quantile_vals.repeat(X.shape[0], 1), uncalibrated_quantiles,
                                            q_levels.to(self.output_device)).to(self.output_device)


    def recalibrate(self, val=None):

        self.iso_reg = None
        self.eval()
        if val is None:
            cdf_vals = self.get_cdf_value(self.X, self.Y)
        else:
            cdf_vals = self.get_cdf_value(torch.cat((self.X, val[0]), dim=0), torch.cat((self.Y, val[1]), dim=0))
        Phat_vals = []
        for cdf in cdf_vals:
            cdf_l = cdf_vals <= cdf
            cdf_ind = torch.zeros(cdf_vals.shape)
            cdf_ind[cdf_l] = 1
            Phat_vals.append(torch.mean(cdf_ind))

        Phat_vals = torch.stack(Phat_vals).reshape(cdf_vals.shape[0], 1)

        ## CONVERT Phat_Vals and cdf_vals TO NUMPY FOR ISOTONIC REGRESSION
        cdf_vals_np = np.float64(cdf_vals.detach().cpu().numpy())
        Phat_vals_np = np.float64(Phat_vals.cpu().numpy().reshape(Phat_vals.shape[0], ))

        # Recalibrate using Isotonic Regression
        iso_reg = IsotonicRegression(out_of_bounds='clip').fit(cdf_vals_np, Phat_vals_np)
        self.iso_reg = iso_reg





    def get_cdf_value(self, X, Y, num_samples = 10000):
        return self.get_cdf1_cdf2(X=X, Y1=Y, Y2=None, num_samples=num_samples)


    def get_cdf1_cdf2(self, X, Y1, Y2 = None, num_samples = 10000):
        # COMPUTE CDF FOR SAME VALUE OF X AND TWO DIFFERENT VALUES OF Y. REQUIRED TO COMPUTE
        # FINITE DIFFERENCE USED TO COMPUTE LIKELIHOOD. SIMULTANEOUS COMPUTATION FOR DIFFERENT YS
        # NECESSARY DUE TO SAMPLING-BASED COMPUTATION


        output_device = self.output_device
        length_data = X.shape[0]
        q_levels = torch.linspace(0, 1, num_samples).to(output_device)

        quantiles = self(X, quantile=q_levels)

        F1 = interp1d(quantiles, q_levels, Y1).clamp(min=0, max=1)
        if Y2 is None:
            if self.iso_reg is None:
                return F1
            else:
                F1_recal = self.iso_reg.predict(F1.detach().cpu().numpy())
                return torch.tensor(F1_recal).to(output_device)
        else:
            F2 = interp1d(quantiles, q_levels, Y2).clamp(min=0, max=1)
            if self.iso_reg is None:
                return F1, F2
            else:
                F1_recal = self.iso_reg.predict(F1.detach().cpu().numpy())
                F2_recal = self.iso_reg.predict(F2.detach().cpu().numpy())
                return F1, F2


    def get_likelihood(self, X, Y, deltay = 1e-2, num_samples = 10000):
        output_device = self.output_device
        dY = deltay*torch.ones(Y.shape).to(output_device)
        Y_dplus = Y+dY
        Y_dminus = Y-dY
        F_minus, F_plus = self.get_cdf1_cdf2(X=X, Y1=Y_dminus, Y2=Y_dplus, num_samples = 10000)
        likelihood = (F_plus - F_minus)/(2*dY.reshape(F_minus.shape))
        return likelihood

    def train_epoch(self, dataloader, X_val=None, Y_val=None):
        self.train()
        batch_loss = []
        self.lambda_reg_vec *= self.scheduler.step()
        # import time
        # time1 = 0
        # time2 = 0
        # time3 = 0
        for Xbatch, Ybatch in dataloader:
            # time1 -= time.time()
            Xbatch, Ybatch = Xbatch.to(self.output_device), Ybatch.to(self.output_device)
            quantile_preds = self.forward(Xbatch)
            loss = self.loss_fun(quantile_preds,Ybatch.repeat(1, quantile_preds.shape[1]))
            # time1 += time.time()

            if X_val is not None:
                # time2 -= time.time()
                self.eval()
                quantile_preds_val = self.forward(X_val) # Majority of the val time spent here
                diff_val = quantile_preds_val - Y_val
                loss_ece = 0
                total_q_levels = torch.hstack([self.half_q_levels, 1-self.half_q_levels.flip(0)]).to(self.output_device)
                for q in torch.arange(total_q_levels.shape[0]):
                    loss_ece += torch.quantile(diff_val[:, q], total_q_levels.flip(0)[q])**2
                self.train()
                loss += loss_ece
                # time2 += time.time()

            for i in range(len(self.half_q_levels)):
                lambda_reg = self.lambda_reg_vec[i]
                if lambda_reg != 0:
                    for params in self.lower_quantile_models[i].parameters():
                        loss += lambda_reg*params.norm()
                    for params in self.upper_quantile_models[i].parameters():
                        loss += lambda_reg*params.norm()
            batch_loss.append(loss.item())
            # time3 -= time.time()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # time3 += time.time()
        self.eval()
        # print(f"Time1: {time1:.2f}, Time2: {time2:.2f}, Time3: {time3:.2f}")
        return batch_loss

    def update_va_loss(self, loss_fn, X_val, Y_val, va_te_q_list, batch_q, curr_ep, num_wait, args):
        with torch.no_grad():
            self.eval()
            Xbatch, Ybatch = X_val.to(self.output_device), Y_val.to(self.output_device)
            quantile_preds = self.forward(Xbatch)
            loss = self.loss_fun(quantile_preds,Ybatch.repeat(1, quantile_preds.shape[1]))
        return loss.item()

    def use_device(self, device):
        self.to(device)
        self.output_device = device
        # move stored calibration tensors too
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)
        for i in range(len(self.lower_quantile_models)):
            self.lower_quantile_models[i] = self.lower_quantile_models[i].to(device)
        for i in range(len(self.upper_quantile_models)):
            self.upper_quantile_models[i] = self.upper_quantile_models[i].to(device)
        return self

    def predict(self,cdf_in,conf_level=0.95,score_distr="z",recal_model=None,recal_type=None):
        # cdf_in: tensor [x, p] of shape (num_x, dim_x + 1)
        with torch.no_grad():
            if not torch.is_tensor(cdf_in):
                cdf_in = torch.tensor(cdf_in)
            cdf_in = cdf_in.to(self.output_device)

            x = cdf_in[:, :-1]
            p = cdf_in[:, -1]

            # Group by unique p and compute once per level; calipso doesn't need extra checks
            unique_p, inverse_idx = torch.unique(p, sorted=True, return_inverse=True)
            out = torch.empty((x.shape[0], 1), device=self.output_device, dtype=x.dtype)

            preds = self.get_quantiles(x, unique_p)

            for up_idx, up in enumerate(unique_p):
                rows = (inverse_idx == up_idx)
                preds_group = preds[rows, up_idx]
                out[rows] = preds_group.reshape(-1, 1)
        return out

    def predict_q(self, x, q_list=None, ens_pred_type="conf", recal_model=None, recal_type=None):
        # x: (num_x, dim_x). q_list: flat tensor of quantile levels
        with torch.no_grad():
            if not torch.is_tensor(x):
                x = torch.tensor(x)
            x = x.to(self.output_device)

            if q_list is None:
                q_list = torch.arange(0.01, 1.00, 0.01, device=self.output_device, dtype=x.dtype)
            else:
                if not torch.is_tensor(q_list):
                    q_list = torch.tensor(q_list)
                q_list = q_list.flatten().to(self.output_device, dtype=x.dtype)

            
            if recal_model is not None:
                if recal_type == "torch":
                    recal_model.to(q_list.device)
                    with torch.no_grad():
                        in_q_list = recal_model(q_list.view(-1, 1)).flatten()
                elif recal_type == "sklearn":
                    q_numpy = q_list.cpu().numpy().reshape(-1, 1)
                    in_q_numpy = recal_model.predict(q_numpy)
                    in_q_list = torch.from_numpy(in_q_numpy).to(self.output_device).flatten().to(x.dtype)
                else:
                    raise ValueError("recal_type incorrect")
            else:
                in_q_list = q_list

            # For calipso, ens_pred_type/recal_* are not used; get calibrated quantiles directly
            pred_mat = self.get_quantiles(x, in_q_list)  # (num_x, num_q)
        return pred_mat
##########################################################

def gen_model(X, Y, X_val, Y_val, output_device, vanilla_model, path=None):
    val = X_val is not None
    nepochs = 1000
    display = True

    X = X.to(output_device)
    Y = Y.to(output_device)
    X_val = X_val.to(output_device)
    Y_val = Y_val.to(output_device)

    class data_set(Dataset):
        def __init__(self, X, Y):
            self.X = X
            self.Y = Y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, index):
            return self.X[index], self.Y[index]

    data = data_set(X=X, Y=Y)
    dataloader = DataLoader(data, batch_size=64, shuffle=True)

    our_model = vanilla_model(X.shape[1]).to(output_device)
    optimizer = optim.Adam(our_model.parameters(), lr=1e-3)
    loss_fun = torch.nn.L1Loss()

    best_loss = torch.inf
    best_weights = None
    early_stop_count = 0

    for epoch in (tqdm(range(nepochs)) if display else range(nepochs)):
        our_model.train()
        batch_loss = []
        for Xbatch, Ybatch in dataloader:
            Xbatch, Ybatch = Xbatch.to(output_device), Ybatch.to(output_device)
            pred = our_model(Xbatch)
            loss = loss_fun(pred,Ybatch)
            batch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if val:
            our_model.eval()
            with torch.no_grad():
                Xbatch, Ybatch = X_val.to(output_device), Y_val.to(output_device)
                pred = our_model(Xbatch)
                loss = loss_fun(pred,Ybatch)
                if loss < best_loss:
                    early_stop_count = 0
                    best_loss = loss
                    best_weights = copy.deepcopy(our_model.state_dict())
                else:
                    early_stop_count += 1
            if early_stop_count > 200:
                break

    if val:
        our_model.load_state_dict(best_weights)

    if path is not None:
        torch.save(our_model.state_dict(), path)

    return copy.deepcopy(our_model.state_dict())

######################################################################################

config_path = '.'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Regularization helps prevent overfitting to outliers, particularly for outer quantiles
class param_scheduler:
    def __init__(self, update_rate: int = 50, decay_rate: float = 0.5, param0: float = 1.0):
        self.update_rate = update_rate
        self.decay_rate = decay_rate
        self.param = param0
        self.counter = 0
    def step(self):
        self.counter += 1
        if self.counter > self.update_rate:
            self.counter = 0
            self.param *= self.decay_rate
        return self.param

# Includes several hyperparameters, running the notebook as is will match the configuration reported in the paper.
# dataset can be one of: 'yacht','boston','concrete','energy','kin8nm','power','wine','naval','protein'
# val_recal: Should you recalibrate on a held-out validation set
# display: Whether to output model evaluation metrics
# dims: If >0, applies PCA to the input data with n_components=dims
# vanilla_weights_path: specifies path to L1 weights
# balanced_recal: when true, upsamples the heldout validation set to have the same number of samples as the training set when recalibrating the model
# va_split: defines a validation size, as a fraction of the train+val size
# cali_favoured: when using early stopping based on the best weighted sum of ECE and sharpness, cali_favoured is the weight of the ECE (sharpness has a fixed weight of 1)
# fix_cali: if true, stops based on the ECE achieving a satisfactory level (relative to the Beyond Pinball Loss paper's reported MAQR results), else use the best weighted sum as early stopping criterion
def run_experiment(X, Y, X_val, Y_val, output_device, vanilla_weights, vanilla_model):
    X = X.to(output_device)
    Y = Y.to(output_device)
    X_val = X_val.to(output_device)
    Y_val = Y_val.to(output_device)

    class data_set(Dataset):
        def __init__(self, X, Y):
            self.X = X
            self.Y = Y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, index):
            return self.X[index], self.Y[index]

    dataset = data_set(X=X, Y=Y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    half_q_levels = torch.tensor([0, 0.025, 0.05, 0.1, 0.2])
    lambda_reg_vec = torch.tensor([0.05, 0.005, 0.005, 0.005, 0.005])
    scheduler = param_scheduler()
    our_model = quantile_model_ensemble(X=X, Y=Y, vanilla_model=vanilla_model, half_q_levels=half_q_levels, output_device=output_device, vanilla_weights=[vanilla_weights, vanilla_weights, None, None, None])
    optimizer = optim.Adam(our_model.parameters(), lr=1e-3)
    loss_fun = torch.nn.MSELoss()
    our_model.train()

    our_model.lambda_reg_vec = lambda_reg_vec
    our_model.scheduler = scheduler
    our_model.optimizer = optimizer
    our_model.loss_fun = loss_fun

    return our_model, dataloader


def main(dataset, seed, X, Y, X_val, Y_val, output_device, vanilla_model):
    run_name = 'calipso_weights'
    # Path(f'{config_path}/{run_name}/').mkdir(parents=True, exist_ok=True)
    # try:
    #     Path(f'{config_path}/{run_name}/{dataset}').mkdir(parents=True, exist_ok=True)
    #     weights_path = f'{config_path}/{run_name}/{dataset}/{seed}_base_model.pt'
    #     if not os.path.exists(weights_path):
    weights = gen_model(X, Y, X_val, Y_val, output_device, vanilla_model, path=None)
    return run_experiment(X, Y, X_val, Y_val, output_device, weights, vanilla_model)

    # except Exception as e:
    #     print(traceback.format_exc())