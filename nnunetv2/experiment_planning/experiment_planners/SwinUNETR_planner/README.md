# SwinUNETR Planner

This directory contains the experiment planner for running the **full SwinUNETR** inside nnUNet.

## Overview

`SwinUNETRPlanner` creates nnUNet plans tailored to the full SwinUNETR implementation in `dynamic-network-architectures`.

The planner is designed to:
- use conservative transformer memory assumptions,
- generate architecture settings compatible with nnUNet static VRAM estimation,
- keep plans naming and CLI usage consistent with standard nnUNet workflows.

Default plans name:
- `nnUNetSwinUNETRPlans`

Planner class:
- `SwinUNETRPlanner`

Network class targeted by the planner:
- `dynamic_network_architectures.architectures.swin_unetr.SwinUNETR`

## SwinUNETR vs LiteSwinUNETR

This planner is intentionally **not** for LiteSwinUNETR.

- `SwinUNETRPlanner` -> full `SwinUNETR`.
- `LiteSwinUNETRPlanner` -> lightweight `LiteSwinUNETR`.

Practical difference:
- Full SwinUNETR generally uses more memory/compute and can require smaller patches or batch size.
- LiteSwinUNETR is usually easier to fit on smaller GPUs.

## What the Planner Configures

Main defaults in `SwinUNETR_planner.py` include:
- `plans_name = "nnUNetSwinUNETRPlans"`
- `UNet_base_num_features = 24`
- `UNet_max_features_2d = 768`
- `UNet_max_features_3d = 384`
- `max_2d_stages = 7`
- `max_3d_stages = 6`
- Swin stage depths pattern: `[0, 2, 2, ...]`
- Window size: `7` per spatial dimension
- Decoder conv blocks per stage: `2`
- Normalization: `InstanceNorm`
- Nonlinearity: `LeakyReLU`

The planner also:
- iteratively shrinks patch size until estimated VRAM matches target,
- applies a small safety factor to batch size for transformer memory spikes.

## Requirements

Before using this planner, ensure:
- `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` are correctly set.
- Your environment can import both `nnunetv2` and `dynamic_network_architectures`.

If paths are not set, follow `documentation/setting_up_paths.md` in nnUNet.

## Typical Workflow

### 1. Plan and preprocess

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID -pl SwinUNETRPlanner
```

This generates `nnUNetSwinUNETRPlans.json` in the dataset preprocessed folder.

### 2. Train

```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD -p nnUNetSwinUNETRPlans
```

Example with all folds:

```bash
nnUNetv2_train DATASET_ID 3d_fullres all -p nnUNetSwinUNETRPlans
```

### 3. Predict

```bash
nnUNetv2_predict \
	-i INPUT_FOLDER \
	-o OUTPUT_FOLDER \
	-d DATASET_ID \
	-c 3d_fullres \
	-f 0 1 2 3 4 \
	-p nnUNetSwinUNETRPlans
```

## Using a Custom GPU Memory Target

If you want to force different patch/batch tradeoffs:

```bash
nnUNetv2_plan_experiment \
	-d DATASET_ID \
	-pl SwinUNETRPlanner \
	-gpu_memory_target 16 \
	-overwrite_plans_name nnUNetSwinUNETRPlans_16G
```

Then preprocess and train using the custom plans name:

```bash
nnUNetv2_preprocess -d DATASET_ID -plans_name nnUNetSwinUNETRPlans_16G
nnUNetv2_train DATASET_ID 3d_fullres FOLD -p nnUNetSwinUNETRPlans_16G
```

## Where Results Are Stored

Training output follows standard nnUNet naming:

```text
nnUNet_results/
└── DatasetXXX_NAME/
		└── nnUNetTrainer__nnUNetSwinUNETRPlans__3d_fullres/
				├── fold_0
				├── fold_1
				├── ...
				├── dataset.json
				├── dataset_fingerprint.json
				└── plans.json
```

## Modifying the Planner

If you want to customize behavior, edit:
- `nnunetv2/experiment_planning/experiment_planners/SwinUNETR_planner/SwinUNETR_planner.py`

Common knobs:
- `UNet_reference_val_2d`, `UNet_reference_val_3d`
- `UNet_base_num_features`
- `UNet_max_features_2d`, `UNet_max_features_3d`
- `max_2d_stages`, `max_3d_stages`
- Swin defaults (`stage_depths`, `window_size`, `drop_path_rate`, decoder blocks)

Recommendation:
- Keep a baseline run before changing planner internals.
- Rename plans with `-overwrite_plans_name` when experimenting.

## Troubleshooting

`Out of memory during training`
- Lower `-gpu_memory_target` when creating plans, or use a separate custom plans name.
- Reduce augmentation/training complexity only after confirming plan settings.

`Planner class not found`
- Ensure class name is exactly `SwinUNETRPlanner`.
- Ensure your local nnUNet package includes this folder.

`Unexpectedly small patch size`
- Full SwinUNETR is memory intensive. This can be expected for large 3D volumes.
- Compare against LiteSwinUNETR if your hardware budget is tight.

## References

- nnUNet repository: https://github.com/MIC-DKFZ/nnUNet
- Dynamic Network Architectures repository: https://github.com/MIC-DKFZ/dynamic-network-architectures
- Swin UNETR paper: https://arxiv.org/abs/2201.01266
