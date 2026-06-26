from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn

from dynamic_network_architectures.architectures.swin_unetr import SwinUNETR
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm

from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props


__author__ = ["Stefano Petraccini", "GitHub Copilot"]


class SwinUNETRPlanner(ExperimentPlanner):
    """
    Experiment planner for SwinUNETR (full variant, distinct from LiteSwinUNETR).

    This planner uses conservative transformer memory assumptions and produces plans
    targeting dynamic_network_architectures.architectures.swin_unetr.SwinUNETR.
    """

    def __init__(
        self,
        dataset_name_or_id: Union[str, int],
        gpu_memory_target_in_gb: float = 8,
        preprocessor_name: str = "DefaultPreprocessor",
        plans_name: str = "nnUNetSwinUNETRPlans",
        overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
        suppress_transpose: bool = False,
    ):
        super().__init__(
            dataset_name_or_id,
            gpu_memory_target_in_gb,
            preprocessor_name,
            plans_name,
            overwrite_target_spacing,
            suppress_transpose,
        )

        self.UNet_class = SwinUNETR

        # Conservative reference values for full SwinUNETR.
        self.UNet_reference_val_3d = 1250000000
        self.UNet_reference_val_2d = 300000000
        self.UNet_reference_val_corresp_GB = 8
        self.UNet_reference_val_corresp_bs_2d = 8
        self.UNet_reference_val_corresp_bs_3d = 2

        # Safety margins to compensate underestimation of real memory usage,
        # especially for transformer-heavy 3D configs and MPS.
        self.memory_estimate_safety_factor_2d = 1.10
        self.memory_estimate_safety_factor_3d = 1.70
        self.memory_estimate_safety_factor_3d_mps = 2.35
        self.batch_size_safety_factor_2d = 0.80
        self.batch_size_safety_factor_3d = 0.45
        self.batch_size_safety_factor_3d_mps = 0.30
        # SwinUNETR on MPS can require batch size 1; override default planner minimum.
        self.UNet_min_batch_size = 1

        self.UNet_base_num_features = 24
        self.UNet_max_features_2d = 768
        self.UNet_max_features_3d = 384

        self.max_2d_stages = 7
        self.max_3d_stages = 6

    def generate_data_identifier(self, configuration_name: str) -> str:
        return self.plans_identifier + "_" + configuration_name

    def get_plans_for_configuration(
        self,
        spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
        median_shape: Union[np.ndarray, Tuple[int, ...]],
        data_identifier: str,
        approximate_n_voxels_dataset: float,
        _cache: dict,
    ) -> dict:
        def _features_per_stage(num_stages: int, max_num_features: int) -> Tuple[int, ...]:
            return tuple(
                int(min(max_num_features, self.UNet_base_num_features * (2 ** i))) for i in range(num_stages)
            )

        def _stage_depths(num_stages: int) -> Tuple[int, ...]:
            return tuple([0] + [2] * (num_stages - 1))

        def _num_heads(features: Tuple[int, ...]) -> Tuple[int, ...]:
            heads = [1]
            for f in features[1:]:
                heads.append(max(1, f // self.UNet_base_num_features))
            return tuple(heads)

        def _decoder_blocks_per_stage(num_stages: int) -> Tuple[int, ...]:
            return tuple([2] * (num_stages - 1))

        def _keygen(patch_size, strides, stage_depths, features):
            return f"{list(patch_size)}_{list(strides)}_{list(stage_depths)}_{list(features)}"

        def _apply_estimate_safety_margin(estimate_value: float, dimensionality: int) -> float:
            if dimensionality == 3:
                if torch.backends.mps.is_available():
                    return float(estimate_value) * self.memory_estimate_safety_factor_3d_mps
                return float(estimate_value) * self.memory_estimate_safety_factor_3d
            return float(estimate_value) * self.memory_estimate_safety_factor_2d

        assert all([i > 0 for i in spacing]), f"Spacing must be > 0! Spacing: {spacing}"

        num_input_channels = len(
            self.dataset_json["channel_names"].keys()
            if "channel_names" in self.dataset_json.keys()
            else self.dataset_json["modality"].keys()
        )

        max_num_features = self.UNet_max_features_2d if len(spacing) == 2 else self.UNet_max_features_3d
        unet_conv_op = convert_dim_to_conv_op(len(spacing))
        norm = get_matching_instancenorm(unet_conv_op)

        # More conservative initial patch sizes for full transformer encoder.
        tmp = 1 / np.array(spacing)
        if len(spacing) == 3:
            initial_patch_size = [round(i) for i in tmp * (144 ** 3 / np.prod(tmp)) ** (1 / 3)]
        elif len(spacing) == 2:
            initial_patch_size = [round(i) for i in tmp * (1408 ** 2 / np.prod(tmp)) ** (1 / 2)]
        else:
            raise RuntimeError("Only 2D and 3D are supported")

        initial_patch_size = np.minimum(initial_patch_size, median_shape[: len(spacing)])

        max_stages = self.max_2d_stages if len(spacing) == 2 else self.max_3d_stages
        (
            _network_num_pool_per_axis,
            pool_op_kernel_sizes,
            conv_kernel_sizes,
            patch_size,
            shape_must_be_divisible_by,
        ) = get_pool_and_conv_props(
            spacing,
            initial_patch_size,
            self.UNet_featuremap_min_edge_length,
            max_stages,
        )

        num_stages = len(pool_op_kernel_sizes)
        features_per_stage = _features_per_stage(num_stages, max_num_features)
        stage_depths = _stage_depths(num_stages)
        num_heads = _num_heads(features_per_stage)
        window_size = tuple([7] * len(spacing))

        architecture_kwargs = {
            "network_class_name": self.UNet_class.__module__ + "." + self.UNet_class.__name__,
            "arch_kwargs": {
                "n_stages": num_stages,
                "features_per_stage": features_per_stage,
                "conv_op": unet_conv_op.__module__ + "." + unet_conv_op.__name__,
                "kernel_sizes": conv_kernel_sizes,
                "strides": pool_op_kernel_sizes,
                "conv_bias": True,
                "norm_op": norm.__module__ + "." + norm.__name__,
                "norm_op_kwargs": {"eps": 1e-5, "affine": True},
                "dropout_op": None,
                "dropout_op_kwargs": None,
                "nonlin": nn.LeakyReLU.__module__ + "." + nn.LeakyReLU.__name__,
                "nonlin_kwargs": {"inplace": True},
                "deep_supervision": True,
                "stage_depths": stage_depths,
                "num_heads": num_heads,
                "window_size": window_size,
                "mlp_ratio": 4.0,
                "qkv_bias": True,
                "drop_path_rate": 0.1,
                "proj_drop_rate": 0.0,
                "attn_drop_rate": 0.0,
                "n_conv_per_stage_decoder": _decoder_blocks_per_stage(num_stages),
            },
            "_kw_requires_import": ("conv_op", "norm_op", "dropout_op", "nonlin"),
        }

        cache_key = _keygen(patch_size, pool_op_kernel_sizes, stage_depths, features_per_stage)
        if cache_key in _cache.keys():
            estimate = _cache[cache_key]
        else:
            estimate = self.static_estimate_VRAM_usage(
                patch_size,
                num_input_channels,
                len(self.dataset_json["labels"].keys()),
                architecture_kwargs["network_class_name"],
                architecture_kwargs["arch_kwargs"],
                architecture_kwargs["_kw_requires_import"],
            )
            estimate = _apply_estimate_safety_margin(estimate, len(spacing))
            _cache[cache_key] = estimate

        reference = (
            self.UNet_reference_val_2d if len(spacing) == 2 else self.UNet_reference_val_3d
        ) * (self.UNet_vram_target_GB / self.UNet_reference_val_corresp_GB)

        while estimate > reference:
            axis_to_be_reduced = np.argsort([i / j for i, j in zip(patch_size, median_shape[: len(spacing)])])[-1]

            patch_size = list(patch_size)
            tmp_patch = deepcopy(patch_size)
            tmp_patch[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]

            _, _, _, _, shape_must_be_divisible_by = get_pool_and_conv_props(
                spacing,
                tmp_patch,
                self.UNet_featuremap_min_edge_length,
                max_stages,
            )
            patch_size[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]

            (
                _network_num_pool_per_axis,
                pool_op_kernel_sizes,
                conv_kernel_sizes,
                patch_size,
                shape_must_be_divisible_by,
            ) = get_pool_and_conv_props(
                spacing,
                patch_size,
                self.UNet_featuremap_min_edge_length,
                max_stages,
            )

            num_stages = len(pool_op_kernel_sizes)
            features_per_stage = _features_per_stage(num_stages, max_num_features)
            stage_depths = _stage_depths(num_stages)
            num_heads = _num_heads(features_per_stage)

            architecture_kwargs["arch_kwargs"].update(
                {
                    "n_stages": num_stages,
                    "kernel_sizes": conv_kernel_sizes,
                    "strides": pool_op_kernel_sizes,
                    "features_per_stage": features_per_stage,
                    "stage_depths": stage_depths,
                    "num_heads": num_heads,
                    "n_conv_per_stage_decoder": _decoder_blocks_per_stage(num_stages),
                }
            )

            cache_key = _keygen(patch_size, pool_op_kernel_sizes, stage_depths, features_per_stage)
            if cache_key in _cache.keys():
                estimate = _cache[cache_key]
            else:
                estimate = self.static_estimate_VRAM_usage(
                    patch_size,
                    num_input_channels,
                    len(self.dataset_json["labels"].keys()),
                    architecture_kwargs["network_class_name"],
                    architecture_kwargs["arch_kwargs"],
                    architecture_kwargs["_kw_requires_import"],
                )
                estimate = _apply_estimate_safety_margin(estimate, len(spacing))
                _cache[cache_key] = estimate

        ref_bs = self.UNet_reference_val_corresp_bs_2d if len(spacing) == 2 else self.UNet_reference_val_corresp_bs_3d
        if len(spacing) == 2:
            bs_safety_factor = self.batch_size_safety_factor_2d
        else:
            bs_safety_factor = (
                self.batch_size_safety_factor_3d_mps
                if torch.backends.mps.is_available()
                else self.batch_size_safety_factor_3d
            )
        # Keep a small safety margin for transformer memory spikes.
        batch_size = max(1, round((reference / estimate) * ref_bs * bs_safety_factor))

        bs_corresponding_to_5_percent = round(
            approximate_n_voxels_dataset * self.max_dataset_covered / np.prod(patch_size, dtype=np.float64)
        )
        batch_size = max(min(batch_size, bs_corresponding_to_5_percent), self.UNet_min_batch_size)

        resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs = self.determine_resampling()
        resampling_softmax, resampling_softmax_kwargs = self.determine_segmentation_softmax_export_fn()
        normalization_schemes, mask_is_used_for_norm = (
            self.determine_normalization_scheme_and_whether_mask_is_used_for_norm()
        )

        plan = {
            "data_identifier": data_identifier,
            "preprocessor_name": self.preprocessor_name,
            "batch_size": batch_size,
            "patch_size": patch_size,
            "median_image_size_in_voxels": median_shape,
            "spacing": spacing,
            "normalization_schemes": normalization_schemes,
            "use_mask_for_norm": mask_is_used_for_norm,
            "resampling_fn_data": resampling_data.__name__,
            "resampling_fn_seg": resampling_seg.__name__,
            "resampling_fn_data_kwargs": resampling_data_kwargs,
            "resampling_fn_seg_kwargs": resampling_seg_kwargs,
            "resampling_fn_probabilities": resampling_softmax.__name__,
            "resampling_fn_probabilities_kwargs": resampling_softmax_kwargs,
            "architecture": architecture_kwargs,
        }
        return plan
