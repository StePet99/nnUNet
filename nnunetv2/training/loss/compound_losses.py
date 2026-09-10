import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.hausdorff_loss import HausdorffDTLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_CE_and_HD_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, hd_kwargs,
                 weight_ce=1, weight_dice=1,
                 hd_final_weight=1.0, hd_warmup_epochs=20, hd_ramp_epochs=30,
                 ignore_label=None, dice_class=SoftDiceLoss):
        """
        DC + CE loss (see DC_and_CE_loss) plus a Hausdorff-Distance-Transform
        boundary term (HausdorffDTLoss), whose weight is scheduled rather than
        fixed: it is 0 for the first `hd_warmup_epochs` epochs (so Dice/CE can
        stabilize the network on something reasonable first, since the HD term
        on near-random early predictions is unstable), then ramps up linearly
        over `hd_ramp_epochs` epochs to `hd_final_weight`, where it stays for
        the rest of training.

        The trainer using this loss MUST call `update_weight(current_epoch)`
        once per epoch (typically from `on_train_epoch_start`) -- this class
        does not know the current epoch on its own.

        :param soft_dice_kwargs: kwargs for the underlying SoftDiceLoss/MemoryEfficientSoftDiceLoss
        :param ce_kwargs: kwargs for RobustCrossEntropyLoss
        :param hd_kwargs: kwargs for HausdorffDTLoss (apply_nonlin, alpha, do_bg)
        :param weight_ce / weight_dice: as in DC_and_CE_loss
        :param hd_final_weight: weight the HD term reaches at the end of the ramp
        :param hd_warmup_epochs: epochs during which HD weight stays at 0
        :param hd_ramp_epochs: epochs over which HD weight ramps 0 -> hd_final_weight
        :param ignore_label: as in DC_and_CE_loss
        :param dice_class: as in DC_and_CE_loss
        """
        super().__init__()
        self.dc_ce = DC_and_CE_loss(soft_dice_kwargs, ce_kwargs, weight_ce, weight_dice,
                                     ignore_label, dice_class)
        self.hd = HausdorffDTLoss(**hd_kwargs)

        self.hd_final_weight = hd_final_weight
        self.hd_warmup_epochs = hd_warmup_epochs
        self.hd_ramp_epochs = hd_ramp_epochs
        # updated once per epoch by update_weight(); starts at 0 so the HD
        # term is inert until the trainer's first on_train_epoch_start call
        self.weight_hd = 0.0

    def update_weight(self, current_epoch: int):
        if current_epoch < self.hd_warmup_epochs:
            self.weight_hd = 0.0
        elif current_epoch < self.hd_warmup_epochs + self.hd_ramp_epochs:
            progress = (current_epoch - self.hd_warmup_epochs) / max(self.hd_ramp_epochs, 1)
            self.weight_hd = self.hd_final_weight * progress
        else:
            self.weight_hd = self.hd_final_weight

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        loss = self.dc_ce(net_output, target)
        # skip computing the HD term entirely (incl. the CPU distance
        # transforms) while its weight is still 0, i.e. during warmup
        if self.weight_hd > 0:
            loss = loss + self.weight_hd * self.hd(net_output, target)
        return loss


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
