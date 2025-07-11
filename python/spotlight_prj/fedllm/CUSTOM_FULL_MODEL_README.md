# Custom FedLLM Scripts for Full Model Training

This directory contains custom scripts that fix the issue with FedML not properly handling `peft_type: "none"` configuration for full model fine-tuning.

## Problem
FedML's `run_fedllm.py` has a bug where it tries to use PEFT-specific functions even when `peft_type: "none"` is configured, causing an `AttributeError: 'Qwen3ForCausalLM' object has no attribute 'peft_config'`.

## Solution
We've created custom subclasses that properly handle both PEFT and non-PEFT models without modifying the library code.

## Checkpoint Format Fix
The original code saves initial checkpoints as `model.safetensors` by default, but the `load_checkpoint` function expects `pytorch_model.bin`. Our custom code includes a fix that:
- Overrides the `_save_checkpoint` function to force `safe_serialization=False`
- Ensures all checkpoints are saved as `pytorch_model.bin` for compatibility
- This fix is automatically applied when using `run_fedllm_custom.py`

## Files Created
- `custom_trainer.py`: Contains `FullModelLLMTrainer` and `FullModelLLMAggregator` classes that check model type before applying state dict
- `run_fedllm_custom.py`: Custom runner that uses the fixed trainer/aggregator classes and patches checkpoint saving
- `launch_fedllm_custom.py`: Custom launcher that runs the custom runner
- `scripts/run_fedml_client_custom.sh`: Custom client launch script
- `scripts/run_fedml_server_custom.sh`: Custom server launch script

## Usage

Instead of using the original scripts:
```bash
# Original (will fail with peft_type: "none")
bash scripts/run_fedml_client.sh 1 "$RUN_ID" localhost 29500 1 auto fedml_config/empty_loop_client.yaml
```

Use the custom scripts:
```bash
# For the server (typically in one terminal)
bash scripts/run_fedml_server_custom.sh 0 "$RUN_ID" localhost 29500 1 auto fedml_config/your_server_config.yaml

# For the client (in another terminal)
bash scripts/run_fedml_client_custom.sh 1 "$RUN_ID" localhost 29500 1 auto fedml_config/empty_loop_client.yaml
```

Note: You'll need to create a server config file similar to `empty_loop_client.yaml` but with server-specific settings.

## Your Training Workflow
1. Launch FedML server with the custom script (if running in federated mode)
2. Launch FedML client with the custom script 
3. In another terminal, run your GRPO training script: `python scripts/train_grpo_gsm8k.py`
4. The custom scripts will properly handle model parameter synchronization for full model training

## When to Remove
Once FedML fixes the bug upstream to properly handle `peft_type: "none"`, you can switch back to the original scripts and remove these custom files. 