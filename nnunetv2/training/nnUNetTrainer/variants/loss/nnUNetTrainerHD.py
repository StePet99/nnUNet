import numpy as np

from nnunetv2.training.loss.compound_losses import DC_and_CE_and_HD_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerDCCEHausdorff(nnUNetTrainer):
    """
    DC + CE + Hausdorff-Distance-Transform boundary loss.

    The HD term's weight is scheduled rather than fixed: it stays at 0 for
    `hd_warmup_epochs` epochs (letting Dice/CE stabilize the network first,
    since the HD term on near-random early predictions is unstable and can
    dominate/explode the gradient), then ramps up linearly over
    `hd_ramp_epochs` epochs to `hd_final_weight`, where it stays for the rest
    of training.

    Tune the schedule by subclassing and overriding the class attributes
    below, e.g.:

        class nnUNetTrainerDCCEHausdorff_fast(nnUNetTrainerDCCEHausdorff):
            hd_warmup_epochs = 10
            hd_ramp_epochs = 15
            hd_final_weight = 0.5

    Usage: nnUNetv2_train DATASET CONFIG FOLD -tr nnUNetTrainerDCCEHausdorff

    NOTE: the Hausdorff term computes a CPU distance transform (scipy) per
    foreground channel, per batch item, at every deep-supervision scale where
    its weight is > 0 (deep-supervision scales below the highest resolution
    already get an exponentially decaying weight from the default nnU-Net
    schedule, so their HD contribution shrinks automatically). Expect a
    non-trivial slowdown once the ramp starts (i.e. once current_epoch >=
    hd_warmup_epochs) -- if it becomes a bottleneck, lower hd_final_weight or
    increase hd_warmup_epochs so it runs for fewer total epochs.

    NOTE 2: currently only supports label-map targets (label_manager.has_regions
    == False), same restriction as HausdorffDTLoss/DC_and_CE_loss's normal usage.
    """

    hd_final_weight = 1.0
    hd_warmup_epochs = 20
    hd_ramp_epochs = 30

    def _build_loss(self):
        assert not self.label_manager.has_regions, \
            "nnUNetTrainerDCCEHausdorff currently only supports label-map " \
            "(non region-based) targets."

        loss = DC_and_CE_and_HD_loss(
            soft_dice_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                               'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            ce_kwargs={},
            hd_kwargs={'apply_nonlin': 'softmax_helper_dim1', 'do_bg': False, 'alpha': 2.0},
            weight_ce=1, weight_dice=1,
            hd_final_weight=self.hd_final_weight,
            hd_warmup_epochs=self.hd_warmup_epochs,
            hd_ramp_epochs=self.hd_ramp_epochs,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def _get_hd_compound_loss(self) -> DC_and_CE_and_HD_loss:
        """Unwrap DeepSupervisionWrapper (if enabled) to reach the actual compound loss."""
        return self.loss.loss if isinstance(self.loss, DeepSupervisionWrapper) else self.loss

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._get_hd_compound_loss().update_weight(self.current_epoch)
