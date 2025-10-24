# U2Net Planner

This directory contains the experiment planner for running U2Net within the nnUNet framework.

## Overview

The U2Net planner is responsible for configuring and preparing experiments using the U2Net architecture. It handles data preprocessing, experiment setup, and parameter selection specifically tailored for U2Net. You can find the U^2Net code in the [Dynamic Network Architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures) repository.

## Usage

- When preprocessing and planning, add the flag `-pl U2NetPlanner` to create plans for the U^2 Net. The plans file will be called "U2NetPlans.json".
- When training, use the flag `-p U2NetPlans`, to address the right plans.

## Modifying the planner

Changing the planner can be useful for various purposes. \
The main parameters one can be interested in changing are the ones at the beginning of the init method.\
Keep in mind that rising too much the number of stages and the depth per stage can result in running out of memory. The current parameters are proven to work well without failing.\

## Tests

The U2Net planner includes a comprehensive test suite with 16 tests covering all major functionality and edge cases. The tests are located in `Tests/test_U2Net_planner.py` and can be run using pytest.

### Running Tests

```bash
# Run all tests
pytest Tests/test_U2Net_planner.py -v

# Run tests with coverage analysis (requires pytest-cov)
pytest Tests/test_U2Net_planner.py --cov=U2Net_planner --cov-report=term-missing

# Run tests without warnings (clean output)
pytest Tests/test_U2Net_planner.py -v --tb=short
```

### Test Coverage

The test suite provides comprehensive coverage (~90-95%) of the U2Net planner functionality:

**Core Functionality Tests:**
- U2Net-specific attribute validation
- Data identifier generation and consistency
- 2D and 3D planning pipeline validation
- U2Net network instantiation and architecture verification

**Advanced Feature Tests:**
- RSU block memory overhead calculations
- Feature scaling for different dimensionalities
- U2Net-optimized topology generation
- Patch size reduction under memory constraints
- Batch size calculation with U2Net-specific factors

**Edge Case and Error Handling:**
- Minimum stage constraint enforcement (prevents IndexError)
- Extreme patch size handling
- Invalid input validation
- Depth extension for insufficient stage configurations
- Cache key generation and behavior

**Performance and Optimization Tests:**
- Architecture parameter consistency validation
- Memory estimation accuracy
- U2Net-specific optimization verification

### Key Test Insights

The tests validate that the U2Net planner:
- Enforces minimum 2 stages (required by RSUDecoder architecture)
- Properly calculates RSU-specific memory overhead (1.5-3x regular convolutions)
- Handles extreme memory pressure scenarios without crashing
- Maintains architecture consistency across all configurations
- Generates proper cache keys including RSU depth information

### Test Dependencies

The tests use mocking to avoid dependencies on actual nnUNet datasets and paths, making them fast and reliable. No external data or GPU is required to run the test suite.


## References

- [nnUNet Documentation](https://github.com/MIC-DKFZ/nnUNet)
- [U2Net Paper](https://arxiv.org/abs/2005.09007)
- [Dynamic Network Architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures)
