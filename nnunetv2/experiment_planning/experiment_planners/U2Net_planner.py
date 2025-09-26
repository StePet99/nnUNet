import numpy as np
from copy import deepcopy
from typing import Union, List, Tuple

from dynamic_network_architectures.architectures.u2net import U2Net
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm
from torch import nn

from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner

from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props

__author__ = ["Stefano Petraccini"]
__email__ = ["stefano.petraccini@studio.unibo.it"]

class U2NetPlanner(ExperimentPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'U2NetPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = U2Net
        # the following two numbers are reference values for VRAM estimation
        self.UNet_reference_val_3d = 680000000
        self.UNet_reference_val_2d = 135000000

        # can be useful to set a maximum number of stages without having to reduce UNet_reference_val_ in order to keep a reasonable patch size.
        self.max_2d_stages = 5  
        self.max_3d_stages = 4  
        
        # RSU depths for each stage
        self.depth_per_stage = [7, 6, 5, 4, 4] #if changing self.max_3d_stages or self.max_2d_stages, make sure this is consistent.

        # next two lines override the default value in ExperimentPlanner
        self.UNet_max_features_3d = 320  # default is 320
        self.UNet_max_features_2d = 512  # default is 512

    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        configurations are unique within each plans file but different plans file can have configurations with the
        same name. In order to distinguish the associated data we need a data identifier that reflects not just the
        config but also the plans it originates from
        """
        if configuration_name == '2d' or configuration_name == '3d_fullres':
            # we do not deviate from ExperimentPlanner so we can reuse its data
            return 'nnUNetPlans' + '_' + configuration_name
        else:
            return self.plans_identifier + '_' + configuration_name

    def get_plans_for_configuration(self,
                                    spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
                                    median_shape: Union[np.ndarray, Tuple[int, ...]],
                                    data_identifier: str,
                                    approximate_n_voxels_dataset: float,
                                    _cache: dict) -> dict:
        def _features_per_stage(num_stages, max_num_features) -> Tuple[int, ...]:
            return tuple([min(max_num_features, self.UNet_base_num_features * 2 ** i) for
                          i in range(num_stages)])

        def _keygen(patch_size, strides):
            return str(patch_size) + '_' + str(strides)

        assert all([i > 0 for i in spacing]), f"Spacing must be > 0! Spacing: {spacing}"
        num_input_channels = len(self.dataset_json['channel_names'].keys()
                                 if 'channel_names' in self.dataset_json.keys()
                                 else self.dataset_json['modality'].keys())
        max_num_features = self.UNet_max_features_2d if len(spacing) == 2 else self.UNet_max_features_3d
        unet_conv_op = convert_dim_to_conv_op(len(spacing))

        # find an initial patch size
        # we first use the spacing to get an aspect ratio
        tmp = 1 / np.array(spacing)

        # we then upscale it so that it initially is certainly larger than what we need
        if len(spacing) == 3:
            initial_patch_size = [round(i) for i in tmp * (256 ** 3 / np.prod(tmp)) ** (1 / 3)]
        elif len(spacing) == 2:
            initial_patch_size = [round(i) for i in tmp * (2048 ** 2 / np.prod(tmp)) ** (1 / 2)]
        else:
            raise RuntimeError()

        # clip initial patch size to median_shape
        initial_patch_size = np.minimum(initial_patch_size, median_shape[:len(spacing)])

        # get network topology
        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
        shape_must_be_divisible_by = get_pool_and_conv_props(spacing, initial_patch_size,
                                                             self.UNet_featuremap_min_edge_length,
                                                             999999)
        num_stages = len(pool_op_kernel_sizes)

        # Apply the maximum stages limit for 3D
        if len(spacing) == 3 and self.max_3d_stages is not None and num_stages > self.max_3d_stages:
            # Limit the number of stages for 3D
            num_stages = self.max_3d_stages
            pool_op_kernel_sizes = pool_op_kernel_sizes[:num_stages]
            conv_kernel_sizes = conv_kernel_sizes[:num_stages]

        # Use appropriate depths for RSU blocks
        depth_per_stage = self.depth_per_stage[:num_stages]

        norm = get_matching_instancenorm(unet_conv_op)
        architecture_kwargs = {
            'network_class_name': self.UNet_class.__module__ + '.' + self.UNet_class.__name__,
            'arch_kwargs': {
                'n_stages': num_stages,
                'features_per_stage': _features_per_stage(num_stages, max_num_features),
                'conv_op': unet_conv_op.__module__ + '.' + unet_conv_op.__name__,
                'kernel_sizes': conv_kernel_sizes,
                'strides': pool_op_kernel_sizes,
                'deep_supervision': True,
                'conv_bias': True,
                'norm_op': norm.__module__ + '.' + norm.__name__,
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None,
                'dropout_op_kwargs': None,
                'nonlin': nn.Sigmoid.__module__ + '.' + nn.Sigmoid.__name__,  # Use string path
                'nonlin_kwargs': {'inplace': True},
                'blocks_nonlin': nn.ReLU.__module__ + '.' + nn.ReLU.__name__,  # Use string path
                'blocks_nonlin_kwargs': {'inplace': True},
                'depth_per_stage': depth_per_stage,
            },
            '_kw_requires_import': ('conv_op', 'norm_op', 'dropout_op', 'nonlin', 'blocks_nonlin'), 
        }

        # now estimate vram consumption
        if _keygen(patch_size, pool_op_kernel_sizes) in _cache.keys():
            estimate = _cache[_keygen(patch_size, pool_op_kernel_sizes)]
        else:
            estimate = self.static_estimate_VRAM_usage(patch_size,
                                                       num_input_channels,
                                                       len(self.dataset_json['labels'].keys()),
                                                       architecture_kwargs['network_class_name'],
                                                       architecture_kwargs['arch_kwargs'],
                                                       architecture_kwargs['_kw_requires_import'],
                                                       )
            _cache[_keygen(patch_size, pool_op_kernel_sizes)] = estimate

        # how large is the reference for us here (batch size etc)?
        # adapt for our vram target
        reference = (self.UNet_reference_val_2d if len(spacing) == 2 else self.UNet_reference_val_3d) * \
                    (self.UNet_vram_target_GB / self.UNet_reference_val_corresp_GB)

        while estimate > reference:
            # patch size seems to be too large, so we need to reduce it
            axis_to_be_reduced = np.argsort([i / j for i, j in zip(patch_size, median_shape[:len(spacing)])])[-1]

            patch_size = list(patch_size)
            tmp = deepcopy(patch_size)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            _, _, _, _, shape_must_be_divisible_by = \
                get_pool_and_conv_props(spacing, tmp,
                                        self.UNet_featuremap_min_edge_length,
                                        999999)
            patch_size[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]

            # now recompute topology
            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
            shape_must_be_divisible_by = get_pool_and_conv_props(spacing, patch_size,
                                                                 self.UNet_featuremap_min_edge_length,
                                                                 999999)

            num_stages = len(pool_op_kernel_sizes)
            # Apply the maximum stages limit for 3D
            if len(spacing) == 3 and self.max_3d_stages is not None and num_stages > self.max_3d_stages:
                # Limit the number of stages for 3D
                num_stages = self.max_3d_stages
                pool_op_kernel_sizes = pool_op_kernel_sizes[:num_stages]
                conv_kernel_sizes = conv_kernel_sizes[:num_stages]

            depth_per_stage = self.depth_per_stage[:num_stages]

            architecture_kwargs['arch_kwargs'].update({
                'n_stages': num_stages,
                'kernel_sizes': conv_kernel_sizes,
                'strides': pool_op_kernel_sizes,
                'features_per_stage': _features_per_stage(num_stages, max_num_features),
                'depth_per_stage': depth_per_stage,
            })
            if _keygen(patch_size, pool_op_kernel_sizes) in _cache.keys():
                estimate = _cache[_keygen(patch_size, pool_op_kernel_sizes)]
            else:
                estimate = self.static_estimate_VRAM_usage(
                    patch_size,
                    num_input_channels,
                    len(self.dataset_json['labels'].keys()),
                    architecture_kwargs['network_class_name'],
                    architecture_kwargs['arch_kwargs'],
                    architecture_kwargs['_kw_requires_import'],
                )
                _cache[_keygen(patch_size, pool_op_kernel_sizes)] = estimate

        # determine the batch size
        ref_bs = self.UNet_reference_val_corresp_bs_2d if len(spacing) == 2 else self.UNet_reference_val_corresp_bs_3d
        batch_size = round((reference / estimate) * ref_bs)

        # cap the batch size to cover at most 5% of the entire dataset
        bs_corresponding_to_5_percent = round(
            approximate_n_voxels_dataset * self.max_dataset_covered / np.prod(patch_size, dtype=np.float64))
        batch_size = max(min(batch_size, bs_corresponding_to_5_percent), self.UNet_min_batch_size)

        resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs = self.determine_resampling()
        resampling_softmax, resampling_softmax_kwargs = self.determine_segmentation_softmax_export_fn()

        normalization_schemes, mask_is_used_for_norm = \
            self.determine_normalization_scheme_and_whether_mask_is_used_for_norm()

        plan = {
            'data_identifier': data_identifier,
            'preprocessor_name': self.preprocessor_name,
            'batch_size': batch_size,
            'patch_size': patch_size,
            'median_image_size_in_voxels': median_shape,
            'spacing': spacing,
            'normalization_schemes': normalization_schemes,
            'use_mask_for_norm': mask_is_used_for_norm,
            'resampling_fn_data': resampling_data.__name__,
            'resampling_fn_seg': resampling_seg.__name__,
            'resampling_fn_data_kwargs': resampling_data_kwargs,
            'resampling_fn_seg_kwargs': resampling_seg_kwargs,
            'resampling_fn_probabilities': resampling_softmax.__name__,
            'resampling_fn_probabilities_kwargs': resampling_softmax_kwargs,
            'architecture': architecture_kwargs
        }
        return plan




