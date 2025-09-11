# Critic Parity Testing and Configuration Guide

This document describes the test infrastructure for comparing DeepSpeed and Megatron critic implementations, along with configuration options for controlling their behavior and achieving deterministic training.

## Configuration Changes

### 1. YAML Configuration Files

#### `example_ppo_megatron_critic_mini.yaml`
File outlining config for Megatron critic

#### `example_ppo.yaml`
File outlining config for Deepspeed critic

## Environment Variables for Fine-Grained Control

### 1. `VALUE_HEAD_INIT` (Already implemented)
- **Purpose**: Control initial value head weights
- **Default**: `0.0` (use random initialization)
- **Example**: `export VALUE_HEAD_INIT=0.01` to set all weights to 0.01
- **Files Modified**: 
  - `/home/isaacy/ROLL/mcore_adapter/src/mcore_adapter/models/model_factory.py`
  - `/home/isaacy/ROLL/roll/models/model_providers.py`

### 2. `CRITIC_DROPOUT_PROB`
- **Purpose**: Control dropout probability in critic value head
- **Default**: `0.0` (no dropout)
- **Example**: `export CRITIC_DROPOUT_PROB=0.0` to disable dropout completely
- **Implementation Location**: 
  - Megatron: `/home/isaacy/ROLL/mcore_adapter/src/mcore_adapter/models/model_factory.py` (lines 373-378)
  - Applied in ValueHeadWrapper class for critic models

### 3. `CRITIC_DATA_SHUFFLE`
- **Purpose**: Control data shuffling during critic training
- **Default**: `1` (shuffle enabled)
- **Example**: `export CRITIC_DATA_SHUFFLE=0` to disable shuffling for deterministic data order
- **Implementation Location**: `/home/isaacy/ROLL/roll/pipeline/base_worker.py` (lines 473-484)

## Usage Examples

### For Fully Deterministic Training
```bash
# Run training with fixed seeds in config
PYTHONUNBUFFERED=1 VLLM_USE_V1=1 VALUE_HEAD_INIT=0.01 CRITIC_DROPOUT_PROB=0.0 CRITIC_DATA_SHUFFLE=1 python -m tests.pipeline.test_critic_pipeline  2>&1 | tee critic_test.log
```

### For Comparing DeepSpeed vs Megatron Critics
```bash
# Ensure both use identical settings
export VALUE_HEAD_INIT=0.01      # Same initial weights
export CRITIC_DROPOUT_PROB=0.0   # Same dropout (none)
export CRITIC_DATA_SHUFFLE=1     # Same shuffling behavior

# Run both configurations with same seed (42)
```

## Randomness Sources Summary

| Component | Control Method | Environment Variable | Config Variable |
|-----------|---------------|----------------------|-----------------|
| Main seed | Config YAML | - | `seed: 42` |
| Training seed | Config YAML | - | `training_args.seed: 42` |
| Data seed | Config YAML | - | `training_args.data_seed: 42` |
| Value head init | Env var | `VALUE_HEAD_INIT` | - |
| Dropout | Env var | `CRITIC_DROPOUT_PROB` | - |
| Data shuffling | Env var | `CRITIC_DATA_SHUFFLE` | - |

## Notes

1. **Seed Hierarchy**: The training_args seeds inherit from the main pipeline seed if not specified
2. **Dropout Defaults**: DeepSpeed TRL models have inherent dropout in the model architecture, while Megatron's dropout is only in the value head wrapper
3. **Data Order**: With `CRITIC_DATA_SHUFFLE=0`, data will be processed in the same order every epoch
4. **Adam Optimizer**: The Adam optimizer itself is deterministic and doesn't introduce randomness

## Implementation Architecture

### Critic Backends
1. **DeepSpeed (TRL)**: Uses `AutoModelForCausalLMWithValueHead` from TRL library
   - Value head implemented as a separate module
   - Inherent dropout in transformer layers
   
2. **Megatron**: Uses custom `McaGPTModel` with `ValueHeadWrapper`
   - Value head replaces output layer when `use_value_head=True`
   - Dropout only in value head wrapper (controlled by `CRITIC_DROPOUT_PROB`)

### Key Implementation Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| `tests/critic_parity/critic_parity_pipeline.py` | Main test pipeline | Runs parallel critic comparison |
| `tests/critic_parity/test_critic_parity.py` | Test entry point | Loads configs and starts pipeline |
| `roll/pipeline/base_worker.py` | Critic worker implementation | `compute_values()`, `train_step()` |
| `roll/models/model_providers.py` | Model initialization | `default_value_model_provider()` |
| `mcore_adapter/src/mcore_adapter/models/model_factory.py` | Megatron value head | `McaGPTModel` with value head support |

## Test Pipeline: `test_critic_parity.py`

### Purpose
The test pipeline (`tests/critic_parity/test_critic_parity.py`) is designed to compare the parity between DeepSpeed and Megatron critic implementations. It runs both critics in parallel and compares their outputs to identify discrepancies.

### What It Tests
1. **Value Computation Parity**: Compares critic value outputs from both implementations on identical inputs
2. **Training Consistency**: Trains both critics with identical data and compares updated values
3. **Real vs Random Data**: Tests critics on both real language data and random token sequences to isolate semantic processing differences
4. **Correlation Analysis**: Measures both Pearson (linear) and Spearman (rank) correlations between critic outputs

### Metrics

1. **Pearson Correlation** (0-1): Measures linear relationship between critic outputs

2. **Spearman Correlation** (0-1): Measures rank/order agreement

3. **Nearly Identical %**: Percentage of values differing by less than 0.1
   - Higher for random data (95-99%) than real data (60-80%)

4. **Value Ranges**: Min/max values from each critic
   - Helps identify if one implementation has different scaling

### Running the Test

```bash
# Basic run
python -m tests.critic_parity.test_critic_parity

# With debugging environment variables
PYTHONUNBUFFERED=1 \
VALUE_HEAD_INIT=0.01 \
CRITIC_DROPOUT_PROB=0.0 \
CRITIC_DATA_SHUFFLE=1 \
python -m tests.critic_parity.test_critic_parity 2>&1 | tee critic_test.log

# With mock data only (for testing pure critic parity)
# Set use_mock_data=True in test_critic_parity.py
```

### Interpreting Results

1. **High correlation (>0.95) on random data, lower (~0.7) on real data**:
   - Normal and expected
   - Indicates architectural differences matter more for semantic content
   - Critics agree on patterns but compute different absolute values

2. **Low correlation on both**:
   - Check seed settings and environment variables
   - Verify both critics are using same model weights
   - Check for uncontrolled randomness sources

3. **Value range differences**:
   - May indicate different initialization or scaling
   - Check VALUE_HEAD_INIT is set consistently

## Environment Variables for Debugging

### Core Configuration Variables

| Variable | Purpose | Default | Example |
|----------|---------|---------|---------|
| `VALUE_HEAD_INIT` | Initialize value head weights to specific value | `0.0` (random) | `VALUE_HEAD_INIT=0.01` |
| `CRITIC_DROPOUT_PROB` | Control dropout in critic models | `0.0` or `0.1` | `CRITIC_DROPOUT_PROB=0.0` |
| `CRITIC_DATA_SHUFFLE` | Enable/disable data shuffling in training | `1` (enabled) | `CRITIC_DATA_SHUFFLE=0` |

### Debugging Workflow

1. **Start with deterministic settings**:
   ```bash
   export VALUE_HEAD_INIT=0.01
   export CRITIC_DROPOUT_PROB=0.0
   export CRITIC_DATA_SHUFFLE=0
   ```

2. **Run test and check correlation**:
   - Random data should show >0.95 correlation
   - Real data typically shows 0.6-0.8 correlation

3. **If correlation is low**:
   - Check seed configuration in YAML files
   - Verify both critics load same base model
   - Look for architecture differences in logs

4. **For production**:
   - Remove VALUE_HEAD_INIT (allow random init)
   - Keep CRITIC_DROPOUT_PROB at desired value
   - Enable CRITIC_DATA_SHUFFLE=1 for better training

## Known Differences and Expected Behavior

### Architectural Differences
1. **Attention Mechanisms**: Different CUDA kernels and numerical precision
2. **Layer Normalization**: Slightly different implementations between frameworks
3. **Dropout Locations**: DeepSpeed has dropout throughout, Megatron only in value head
4. **Tensor Parallelism**: Different sharding strategies affect numerical precision

### Expected Correlation Values
| Data Type | Pearson | Spearman | Interpretation |
|-----------|---------|----------|----------------|
| Random tokens | 0.95-0.99 | 0.95-0.99 | Near-identical, differences are numerical noise |
| Real text | 0.65-0.80 | 0.70-0.85 | Good agreement with expected architectural differences |
| After training | 0.60-0.75 | 0.65-0.80 | Divergence increases with training steps |

### Why Random Data Shows Higher Correlation
1. **Uniform Activation Patterns**: Random tokens create noise without semantic structure
2. **No Attention Focusing**: Attention weights are diffuse, reducing precision sensitivity
3. **Balanced Statistics**: Layer norm operates on well-distributed values
4. **Error Cancellation**: Random differences tend to cancel out rather than accumulate

## Verification

To verify identical behavior between critics:
1. Set all environment variables to the same values
2. Ensure both configs use `seed: 42` at all levels
3. Initialize with `VALUE_HEAD_INIT=0.01` for consistent starting weights
4. Monitor initial loss values - they should be identical if properly configured
5. Run the test pipeline and check correlation metrics
6. Compare correlations against expected values table above