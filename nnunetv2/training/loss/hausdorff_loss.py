"""
Hausdorff-Distance-Transform loss (Karimi & Salcudean, 2019: "Reducing the
Hausdorff Distance in Medical Image Segmentation with Convolutional Neural
Networks", https://arxiv.org/abs/1904.10030).

Interface matches the rest of nnunetv2/training/loss/*: forward(net_output,
target, loss_mask=None), net_output = raw logits (B, C, X, Y[, Z]), target =
either a label map (B, 1, X, Y[, Z]) or an already one-hot tensor with the
same shape as net_output.

IMPORTANT: this loss is unstable if used on its own from epoch 0 (the
network output is essentially noise at the start of training, so the
distance transform of the thresholded prediction can be huge and the
gradient explodes / becomes NaN on degenerate, all-background patches).
Always combine it with Dice+CE (see DC_and_CE_and_HD_loss in
compound_losses.py) and ramp its weight up only after a warmup period.
"""
from typing import Optional

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt as edt
from torch import nn

from nnunetv2.utilities.helpers import softmax_helper_dim1

__author__ = ["StePet99"]


class HausdorffDTLoss(nn.Module):
    def __init__(self,
                 apply_nonlin: Optional[str] = 'softmax_helper_dim1',
                 alpha: float = 2.0,
                 do_bg: bool = False):
        """
        :param apply_nonlin: name of the nonlinearity to apply to net_output
            before computing the loss. 'softmax_helper_dim1' for standard
            multiclass (mutually exclusive) nnU-Net output, or None if
            net_output is already a probability map.
        :param alpha: exponent on the distance maps (2.0 as in the original
            paper). Higher values penalize far-away mistakes more strongly.
        :param do_bg: whether to include channel 0 (background) in the loss,
            exactly like SoftDiceLoss's do_bg. Normally False.
        """
        super().__init__()
        self.alpha = alpha
        self.do_bg = do_bg
        if apply_nonlin == 'softmax_helper_dim1':
            self.apply_nonlin = softmax_helper_dim1
        elif apply_nonlin is None:
            self.apply_nonlin = None
        else:
            raise ValueError(f"Unsupported apply_nonlin '{apply_nonlin}'")

    @staticmethod
    def _distance_field(binary_mask: np.ndarray) -> np.ndarray:
        """
        binary_mask: (X, Y[, Z]) bool array for a single (batch, channel).
        Returns fg_dist + bg_dist (distance to the nearest boundary voxel,
        zero exactly on the boundary). If the mask is empty or entirely
        foreground (no boundary in this crop -- common with small, tightly
        cropped lesion patches), the field is left at 0 everywhere so the HD
        term contributes ~nothing for that sample instead of blowing up.
        """
        field = np.zeros_like(binary_mask, dtype=np.float32)
        if binary_mask.any() and not binary_mask.all():
            fg_dist = edt(binary_mask)
            bg_dist = edt(~binary_mask)
            field = (fg_dist + bg_dist).astype(np.float32)
        return field

    def forward(self, net_output: torch.Tensor, target: torch.Tensor,
                loss_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = self.apply_nonlin(net_output) if self.apply_nonlin is not None else net_output

        if probs.shape == target.shape:
            target_onehot = target
        else:
            with torch.no_grad():
                target_onehot = torch.cat(
                    [(target.long() == i).long() for i in range(probs.shape[1])], dim=1
                ).float()

        num_ch = probs.shape[1]
        start_ch = 0 if (self.do_bg or num_ch == 1) else 1

        total_loss = probs.new_zeros(())
        n_terms = 0

        for c in range(start_ch, num_ch):
            pred_c = probs[:, c]            # (B, X, Y[, Z]) -- differentiable
            target_c = target_onehot[:, c]  # (B, X, Y[, Z])

            # distance transforms are computed on numpy, no gradient needed here
            with torch.no_grad():
                pred_np = (pred_c.detach().cpu().numpy() > 0.5)
                target_np = (target_c.detach().cpu().numpy() > 0.5)

                batch_size = pred_np.shape[0]
                pred_dt = np.zeros(pred_np.shape, dtype=np.float32)
                target_dt = np.zeros(target_np.shape, dtype=np.float32)
                for b in range(batch_size):
                    pred_dt[b] = self._distance_field(pred_np[b])
                    target_dt[b] = self._distance_field(target_np[b])

                pred_dt_t = torch.from_numpy(pred_dt).to(pred_c.device)
                target_dt_t = torch.from_numpy(target_dt).to(pred_c.device)
                distance = pred_dt_t ** self.alpha + target_dt_t ** self.alpha

            # this part IS differentiable w.r.t. pred_c
            pred_error = (pred_c - target_c) ** 2
            total_loss = total_loss + (pred_error * distance).mean()
            n_terms += 1

        # loss_mask (e.g. ignore_label) kept only for interface compatibility
        # with the rest of nnU-Net's losses -- not applied to the DT term.
        return total_loss / max(n_terms, 1)
