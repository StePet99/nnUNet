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
    """
    Experiment planner for U2Net architecture.
    
    This planner extends the default ExperimentPlanner to work with the U2Net architecture,
    allowing for custom configuration of network parameters such as depths per stage,
    maximum features, and stages limits.
    """

    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'U2NetPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        """
        Initialize the U2Net experiment planner.
        
        Parameters
        ----------
        dataset_name_or_id : Union[str, int]
            Dataset name or ID to plan experiments for.
        gpu_memory_target_in_gb : float, optional
            Target GPU memory usage in GB, by default 8.
        preprocessor_name : str, optional
            Name of the preprocessor to use, by default 'DefaultPreprocessor'.
        plans_name : str, optional
            Name for the plans file, by default 'U2NetPlans'.
        overwrite_target_spacing : Union[List[float], Tuple[float, ...]], optional
            Custom target spacing to use instead of computed one, by default None.
        suppress_transpose : bool, optional
            Whether to suppress the transpose of the data, by default False.
        """
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = U2Net
        
        # the following numbers are reference values for VRAM estimation
        self.UNet_reference_val_3d = 680000000
        self.UNet_reference_val_2d = 135000000
        self.UNet_reference_val_corresp_GB = 8  # Reference GPU memory in GB
        self.UNet_reference_val_corresp_bs_2d = 12  # Reference batch size for 2D
        self.UNet_reference_val_corresp_bs_3d = 2   # Reference batch size for 3D
        
        # GPU memory target is set by parent class but we ensure it's available
        # self.UNet_vram_target_GB is set by parent __init__

        # can be useful to set a maximum number of stages without having to reduce UNet_reference_val_ in order to keep a reasonable patch size.
        self.max_2d_stages = 10  
        self.max_3d_stages = 10  
        
        # RSU depths for each stage
        self.depth_per_stage = [7, 6, 5, 4, 4, 4, 4, 4, 4, 4]  # if changing self.max_3d_stages or self.max_2d_stages, make sure this is consistent.

        # next two lines override the default value in ExperimentPlanner
        self.UNet_max_features_3d = 512  # default is 320
        self.UNet_max_features_2d = 1024  # default is 512
        

    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        Generate a unique identifier for the data associated with a configuration.
        
        Configurations are unique within each plans file but different plans files can have 
        configurations with the same name. This method creates an identifier that reflects 
        not just the configuration but also the plans it originates from.
        
        Parameters
        ----------
        configuration_name : str
            Name of the configuration.
            
        Returns
        -------
        str
            Unique data identifier for the configuration.
        """
        return self.plans_identifier + '_' + configuration_name

    def get_plans_for_configuration(self,
                                    spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
                                    median_shape: Union[np.ndarray, Tuple[int, ...]],
                                    data_identifier: str,
                                    approximate_n_voxels_dataset: float,
                                    _cache: dict) -> dict:
        """
        Get plans for a specific network configuration optimized for U2Net architecture.
        
        This method determines the network architecture, batch size, and other 
        configuration parameters based on the provided spacing, median shape, and U2Net-specific constraints.
        
        Parameters
        ----------
        spacing : Union[np.ndarray, Tuple[float, ...], List[float]]
            Voxel spacing of the data.
        median_shape : Union[np.ndarray, Tuple[int, ...]]
            Median shape of the data in voxels.
        data_identifier : str
            Unique identifier for the data.
        approximate_n_voxels_dataset : float
            Approximate number of voxels in the dataset.
        _cache : dict
            Cache for storing intermediate results.
            
        Returns
        -------
        dict
            Dictionary containing the plans for the configuration.
        """
        def _features_per_stage(num_stages, max_num_features) -> Tuple[int, ...]:
            """
            Calculate the number of features for each stage for U2Net.
            U2Net typically uses fewer features than regular U-Net due to nested structure complexity.
            
            Parameters
            ----------
            num_stages : int
                Number of stages in the network.
            max_num_features : int
                Maximum number of features allowed.
                
            Returns
            -------
            Tuple[int, ...]
                Number of features for each stage.
            """
            return tuple([min(max_num_features, self.UNet_base_num_features * 2 ** i) for
                          i in range(num_stages)])

        def _estimate_rsu_memory_overhead(depth, features, patch_size):
            """
            Estimate memory overhead for RSU blocks compared to regular conv blocks.
            RSU blocks have nested U-structures that require additional memory.
            
            Parameters
            ----------
            depth : int
                Depth of the RSU block (number of nested layers).
            features : int
                Number of feature channels.
            patch_size : tuple
                Current patch size.
                
            Returns
            -------
            float
                Memory overhead multiplier (>1.0 for RSU vs regular conv).
            """
            # RSU blocks create multiple feature maps at different scales
            # Approximate overhead based on depth and nested pooling operations
            base_overhead = 1.0
            for d in range(1, min(depth, 5)):  # Cap at depth 5 for memory estimation
                # Each nested level adds feature maps at reduced resolution
                scale_factor = 2 ** d
                resolution_factor = 1.0 / (scale_factor ** len(patch_size))
                base_overhead += resolution_factor * 0.5  # Approximate memory contribution
            
            return min(base_overhead, 3.0)  # Cap overhead at 3x

        def _u2net_optimized_topology(spacing, initial_patch_size, min_edge_length, max_stages):
            """
            Calculate U2Net-optimized network topology.
            Considers RSU block memory requirements and nested structure constraints.
            
            Parameters
            ----------
            spacing : array-like
                Voxel spacing.
            initial_patch_size : array-like
                Initial patch size estimate.
            min_edge_length : int
                Minimum edge length for feature maps.
            max_stages : int
                Maximum number of stages allowed.
                
            Returns
            -------
            tuple
                Network topology parameters optimized for U2Net.
            """
            # Start with standard topology calculation
            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
            shape_must_be_divisible_by = get_pool_and_conv_props(spacing, initial_patch_size,
                                                                 min_edge_length, 999999)
            
            # Apply U2Net-specific constraints
            num_stages = len(pool_op_kernel_sizes)
            
            # Limit stages based on U2Net constraints and available depths
            max_available_depths = len(self.depth_per_stage)
            num_stages = min(num_stages, max_stages, max_available_depths)
            
            # U2Net requires minimum 2 stages for RSUDecoder to work properly
            num_stages = max(num_stages, 2)
            
            # Adjust patch size for U2Net memory requirements
            # RSU blocks are more memory-intensive, so we need smaller patches
            u2net_memory_factor = 1.5 if len(spacing) == 3 else 1.3
            adjusted_patch_size = [int(p / u2net_memory_factor) for p in patch_size]
            
            # Ensure patch size is still valid for the network topology
            for i in range(len(adjusted_patch_size)):
                # Make sure patch size is divisible by required factors
                if len(shape_must_be_divisible_by) > i:
                    divisor = shape_must_be_divisible_by[i]
                    adjusted_patch_size[i] = max(divisor, 
                                               (adjusted_patch_size[i] // divisor) * divisor)
            
            # Truncate or extend topology to match the number of stages
            original_stages = len(pool_op_kernel_sizes)
            if num_stages > original_stages:
                # Extend with safe defaults if we need more stages
                pool_op_kernel_sizes.extend([(1,) * len(spacing)] * (num_stages - original_stages))
                conv_kernel_sizes.extend([(3,) * len(spacing)] * (num_stages - original_stages))
            else:
                # Truncate to required stages
                pool_op_kernel_sizes = pool_op_kernel_sizes[:num_stages]
                conv_kernel_sizes = conv_kernel_sizes[:num_stages]
            
            return (network_num_pool_per_axis[:num_stages] if isinstance(network_num_pool_per_axis, list) 
                   else network_num_pool_per_axis,
                   pool_op_kernel_sizes, conv_kernel_sizes, adjusted_patch_size, shape_must_be_divisible_by)

        def _keygen(patch_size, strides, depths):
            """
            Generate a cache key based on patch size, strides, and RSU depths.
            
            Parameters
            ----------
            patch_size : list or tuple
                Patch size for the network.
            strides : list or tuple
                Strides for the network.
            depths : list or tuple
                RSU block depths.
                
            Returns
            -------
            str
                Cache key string.
            """
            return str(patch_size) + '_' + str(strides) + '_' + str(depths)

        assert all([i > 0 for i in spacing]), f"Spacing must be > 0! Spacing: {spacing}"
        num_input_channels = len(self.dataset_json['channel_names'].keys()
                                 if 'channel_names' in self.dataset_json.keys()
                                 else self.dataset_json['modality'].keys())
        max_num_features = self.UNet_max_features_2d if len(spacing) == 2 else self.UNet_max_features_3d
        unet_conv_op = convert_dim_to_conv_op(len(spacing))

        # U2Net-specific initial patch size calculation
        # Start more conservatively due to RSU memory overhead
        tmp = 1 / np.array(spacing)
        
        if len(spacing) == 3:
            # Reduce initial size by 20% for 3D U2Net due to higher memory requirements
            initial_patch_size = [round(i) for i in tmp * (200 ** 3 / np.prod(tmp)) ** (1 / 3)]
        elif len(spacing) == 2:
            # Reduce initial size by 15% for 2D U2Net
            initial_patch_size = [round(i) for i in tmp * (1700 ** 2 / np.prod(tmp)) ** (1 / 2)]
        else:
            raise RuntimeError("Only 2D and 3D are supported")

        # Clip initial patch size to median_shape
        initial_patch_size = np.minimum(initial_patch_size, median_shape[:len(spacing)])

        # Apply stage limits
        max_stages_for_dim = self.max_2d_stages if len(spacing) == 2 else self.max_3d_stages

        # Get U2Net-optimized network topology
        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
        shape_must_be_divisible_by = _u2net_optimized_topology(spacing, initial_patch_size,
                                                               self.UNet_featuremap_min_edge_length,
                                                               max_stages_for_dim)
        
        num_stages = len(pool_op_kernel_sizes)
        
        # U2Net requires minimum 2 stages for RSUDecoder to work properly
        num_stages = max(num_stages, 2)
        
        # Ensure we have enough depth values, extend with last value if needed
        depth_per_stage = self.depth_per_stage[:num_stages]
        if len(depth_per_stage) < num_stages:
            # Extend with the last available depth value
            last_depth = self.depth_per_stage[-1] if self.depth_per_stage else 4
            depth_per_stage.extend([last_depth] * (num_stages - len(depth_per_stage)))

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
                'nonlin': nn.Sigmoid.__module__ + '.' + nn.Sigmoid.__name__,
                'nonlin_kwargs': {'inplace': True},
                'blocks_nonlin': nn.ReLU.__module__ + '.' + nn.ReLU.__name__,
                'blocks_nonlin_kwargs': {'inplace': True},
                'depth_per_stage': depth_per_stage,
            },
            '_kw_requires_import': ('conv_op', 'norm_op', 'dropout_op', 'nonlin', 'blocks_nonlin'), 
        }

        # U2Net-aware VRAM estimation with RSU overhead
        cache_key = _keygen(patch_size, pool_op_kernel_sizes, depth_per_stage)
        if cache_key in _cache:
            base_estimate = _cache[cache_key]
        else:
            base_estimate = self.static_estimate_VRAM_usage(patch_size,
                                                           num_input_channels,
                                                           len(self.dataset_json['labels'].keys()),
                                                           architecture_kwargs['network_class_name'],
                                                           architecture_kwargs['arch_kwargs'],
                                                           architecture_kwargs['_kw_requires_import'])
        
        # Apply RSU memory overhead correction
        rsu_overhead = 1.0
        features_per_stage = _features_per_stage(num_stages, max_num_features)
        for stage_idx, (depth, features) in enumerate(zip(depth_per_stage, features_per_stage)):
            stage_overhead = _estimate_rsu_memory_overhead(depth, features, patch_size)
            rsu_overhead = max(rsu_overhead, stage_overhead)
        
        estimate = base_estimate * rsu_overhead
        _cache[cache_key] = estimate

        # Adjust reference values for U2Net
        reference = (self.UNet_reference_val_2d if len(spacing) == 2 else self.UNet_reference_val_3d) * \
                    (self.UNet_vram_target_GB / self.UNet_reference_val_corresp_GB)

        # Patch size reduction loop with U2Net-specific constraints
        iteration_count = 0
        max_iterations = 10  # Prevent infinite loops
        
        while estimate > reference and iteration_count < max_iterations:
            iteration_count += 1
            
            # More aggressive patch size reduction for U2Net
            axis_to_be_reduced = np.argsort([i / j for i, j in zip(patch_size, median_shape[:len(spacing)])])[-1]
            
            patch_size = list(patch_size)
            reduction_factor = 0.85  # Reduce by 15% each iteration (more aggressive than default)
            
            # Set minimum patch size for U2Net - needs to be large enough for at least 2 stages
            min_patch_size = 32 if len(spacing) == 3 else 64
            patch_size[axis_to_be_reduced] = max(min_patch_size, int(patch_size[axis_to_be_reduced] * reduction_factor))
            
            # Ensure divisibility requirements are met
            if len(shape_must_be_divisible_by) > axis_to_be_reduced:
                divisor = shape_must_be_divisible_by[axis_to_be_reduced]
                patch_size[axis_to_be_reduced] = max(divisor, 
                                                   (patch_size[axis_to_be_reduced] // divisor) * divisor)

            # Recompute topology with new patch size
            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
            shape_must_be_divisible_by = _u2net_optimized_topology(spacing, patch_size,
                                                                   self.UNet_featuremap_min_edge_length,
                                                                   max_stages_for_dim)
            
            num_stages = len(pool_op_kernel_sizes)
            
            # U2Net requires minimum 2 stages for RSUDecoder to work properly
            num_stages = max(num_stages, 2)
            
            # Ensure we have enough depth values, extend with last value if needed
            depth_per_stage = self.depth_per_stage[:num_stages]
            if len(depth_per_stage) < num_stages:
                # Extend with the last available depth value
                last_depth = self.depth_per_stage[-1] if self.depth_per_stage else 4
                depth_per_stage.extend([last_depth] * (num_stages - len(depth_per_stage)))
            
            architecture_kwargs['arch_kwargs'].update({
                'n_stages': num_stages,
                'kernel_sizes': conv_kernel_sizes,
                'strides': pool_op_kernel_sizes,
                'features_per_stage': _features_per_stage(num_stages, max_num_features),
                'depth_per_stage': depth_per_stage,
            })
            
            # Recalculate estimate with RSU overhead
            cache_key = _keygen(patch_size, pool_op_kernel_sizes, depth_per_stage)
            if cache_key in _cache:
                base_estimate = _cache[cache_key]
            else:
                base_estimate = self.static_estimate_VRAM_usage(patch_size,
                                                               num_input_channels,
                                                               len(self.dataset_json['labels'].keys()),
                                                               architecture_kwargs['network_class_name'],
                                                               architecture_kwargs['arch_kwargs'],
                                                               architecture_kwargs['_kw_requires_import'])
            
            # Recalculate RSU overhead for new configuration
            rsu_overhead = 1.0
            features_per_stage = _features_per_stage(num_stages, max_num_features)
            for stage_idx, (depth, features) in enumerate(zip(depth_per_stage, features_per_stage)):
                stage_overhead = _estimate_rsu_memory_overhead(depth, features, patch_size)
                rsu_overhead = max(rsu_overhead, stage_overhead)
            
            estimate = base_estimate * rsu_overhead
            _cache[cache_key] = estimate

        # Determine batch size with U2Net considerations
        ref_bs = self.UNet_reference_val_corresp_bs_2d if len(spacing) == 2 else self.UNet_reference_val_corresp_bs_3d
        
        # U2Net typically requires smaller batch sizes due to complexity
        u2net_batch_factor = 0.75  # Reduce batch size by 25%
        batch_size = max(1, round((reference / estimate) * ref_bs * u2net_batch_factor))

        # Cap the batch size to cover at most 5% of the entire dataset
        bs_corresponding_to_5_percent = round(
            approximate_n_voxels_dataset * self.max_dataset_covered / np.prod(patch_size, dtype=np.float64))
        batch_size = max(min(batch_size, bs_corresponding_to_5_percent), self.UNet_min_batch_size)

        # Get resampling and normalization configurations
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





