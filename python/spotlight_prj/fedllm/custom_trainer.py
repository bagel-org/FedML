"""Custom trainer that properly handles non-PEFT models for full fine-tuning.

This fixes a bug in FedML where set_model_params tries to call set_peft_model_state_dict
even when peft_type="none" is configured, causing AttributeError for non-PEFT models.

This version also integrates GRPO training for GSM8K dataset.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import re
import torch
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

from accelerate.utils import broadcast_object_list
from datasets import load_dataset
from fedml.train.llm.modeling_utils import to_device
from fedml.train.llm.distributed import barrier
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig

from run_fedllm import LLMTrainer, LLMAggregator, save_checkpoint, load_checkpoint
from src.peft_utils import set_peft_model_state_dict
from src.modeling_utils import load_state_dict


class FullModelLLMTrainer(LLMTrainer):
    """Custom trainer that properly handles both PEFT and non-PEFT models with GRPO training."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # GSM8K specific regex patterns
        # Regex for GSM8K dataset format (####)
        self.DATASET_ANS = re.compile(r"####\s*([-+]?\d+\.?\d*)")
        # Regex for model completion format (\boxed{})
        self.MODEL_ANS = re.compile(r"\\boxed\{([^}]*)\}")
    
    def reward_fn(self, completions, answer, **_):
        """Reward function for GSM8K that checks if the predicted answer matches the true answer."""
        out = []
        for c, ans in zip(completions, answer):
            # Extract from dataset answer (GSM8K format)
            tru = self.DATASET_ANS.search(ans)
            # Extract from model completion (boxed format, fallback to GSM8K format)
            pred = self.MODEL_ANS.search(c)
            if not pred:
                pred = self.DATASET_ANS.search(c)
            
            if pred and tru:
                pred_num = pred.group(1)
                tru_num = tru.group(1)
                out.append(1.0 if pred_num == tru_num else -0.2)
            else:
                out.append(-0.2)
        return out
    
    def train(self, train_data, device, args):
        """Override train to use GRPO training on GSM8K dataset."""
        self.log("Starting GRPO training on GSM8K")
        
        # Load GSM8K dataset
        ds = load_dataset("openai/gsm8k", "main", split="train")
        ds = ds.rename_column("question", "prompt")
        
        # Get GRPO-specific settings from FedML config or use defaults
        grpo_max_steps = getattr(args, 'grpo_max_steps', -1)  # -1 means use epochs
        grpo_num_epochs = getattr(args, 'grpo_num_epochs', 3)
        grpo_batch_size = getattr(args, 'grpo_batch_size', 4)
        
        # Calculate effective batch size for GRPO constraint
        # effective_batch_size = num_gpus * per_device_batch_size * gradient_accumulation_steps
        gradient_accumulation_steps = 2
        effective_batch_size = 1 * grpo_batch_size * gradient_accumulation_steps
        
        # Num generations must evenly divide the effective batch size
        # For testing with small batch sizes, use a smaller num_generations
        if effective_batch_size >= 8:
            num_generations = 8
        elif effective_batch_size >= 4:
            num_generations = 4
        else:
            num_generations = 2
        
        # For testing, we can use a very small number of steps
        if grpo_max_steps > 0:
            self.log(f"GRPO training for {grpo_max_steps} steps (test mode)")
        else:
            self.log(f"GRPO training for {grpo_num_epochs} epochs")
        
        self.log(f"Using num_generations={num_generations} with effective batch size={effective_batch_size}")
        
        # Configure GRPO training
        cfg = GRPOConfig(
            output_dir=str(self.checkpoint_dir / "grpo"),
            per_device_train_batch_size=grpo_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_completion_length=1024,
            num_generations=num_generations,  # Adjusted based on effective batch size
            num_train_epochs=grpo_num_epochs if grpo_max_steps <= 0 else 1,  # Use 1 epoch if max_steps is set
            max_steps=grpo_max_steps if grpo_max_steps > 0 else -1,  # Override epochs with max_steps
            learning_rate=5e-6,
            bf16=True,
            gradient_checkpointing=False,
            logging_steps=5 if grpo_max_steps > 0 and grpo_max_steps < 50 else 25,  # More frequent logging for short runs
            log_completions=True,
            save_steps=grpo_max_steps if grpo_max_steps > 0 else 500,  # Save at the end if using max_steps
            # Add seed for reproducibility in federated setting
            seed=42 + self.round_idx * 100 + args.rank,  # Different seed per round and client
        )
        
        # Create GRPO trainer
        grpo_trainer = GRPOTrainer(
            model=self.model,  # Use FedML's model
            args=cfg,
            train_dataset=ds.shuffle(seed=cfg.seed),
            processing_class=self.tokenizer,  # Use FedML's tokenizer
            reward_funcs=self.reward_fn,
        )
        
        # Run GRPO training
        grpo_trainer.train()
        
        # Save the trained model in FedML's expected location
        self.latest_checkpoint_dir = self.checkpoint_dir / f"round_{self.round_idx}_before_agg"
        self.log(f"Saving GRPO-trained model to \"{self.latest_checkpoint_dir}\"")
        
        # GRPO trainer updates the model in-place, so we can directly save the current model state
        save_checkpoint(
            self.model,
            self.latest_checkpoint_dir,
            is_saving_process=self.training_args.should_save,
            synchronize=True
        )
        
        self.log("GRPO training finished")
    
    def on_after_local_training(self, train_data, device, args):
        """Override to skip the parent's checkpoint saving since we handle it in train()."""
        self.log("Skipping parent's on_after_local_training (already saved in train method)")
        # We already saved the checkpoint in the train() method, so we don't need to do anything here
        # This prevents the AttributeError from trying to save the trainer's optimizer state
        return None
    
    def set_model_params(self, model_parameters) -> None:
        self.log("start")

        model_parameters = to_device(model_parameters, device="cpu")

        barrier()
        # Check if model is a PEFT model
        if isinstance(self.model, PeftModel):
            set_peft_model_state_dict(self.model, model_parameters)
        else:
            # For non-PEFT models, use regular load_state_dict
            load_state_dict(self.model, model_parameters, strict=False)
        barrier()

        if self.round_idx >= 0 and self.should_save:
            # save aggregated model checkpoint
            self.latest_checkpoint_dir = self.checkpoint_dir / f"round_{self.round_idx}_after_agg"
            self.log(f"saving aggregated model to \"{self.latest_checkpoint_dir}\"")
            save_checkpoint(
                self.model,
                self.latest_checkpoint_dir,
                is_saving_process=self.training_args.should_save,
                state_dict=model_parameters,
                synchronize=True
            )

        self.log("finished")
    
    # Explicitly define sync_process_group to ensure FedML recognizes it
    def sync_process_group(
            self,
            round_idx: Optional[int] = None,
            model_params: Optional[Any] = None,
            client_index: Optional[int] = None,
            from_process: int = 0
    ) -> None:
        self.log("start")

        if round_idx is None:
            round_idx = self.round_idx

        broadcast_object_list([round_idx, model_params, client_index], from_process=from_process)

        self.log("finished")

    def await_sync_process_group(self, from_process: int = 0) -> list:
        self.log("start")

        outputs = broadcast_object_list([None, None, None], from_process=from_process)

        self.log("finished")
        return outputs


class FullModelLLMAggregator(LLMAggregator):
    """Custom aggregator that properly handles both PEFT and non-PEFT models."""
    
    def set_model_params(self, model_parameters) -> None:
        self.log("start")

        model_parameters = to_device(model_parameters, device="cpu")

        barrier()
        # Check if model is a PEFT model
        if isinstance(self.model, PeftModel):
            set_peft_model_state_dict(self.model, model_parameters)
        else:
            # For non-PEFT models, use regular load_state_dict
            load_state_dict(self.model, model_parameters, strict=False)
        barrier()

        if self.round_idx >= 0 and self.should_save:
            # save aggregated model checkpoint
            self.latest_checkpoint_dir = self.checkpoint_dir / f"round_{self.round_idx}_after_agg"
            self.log(f"saving aggregated model to \"{self.latest_checkpoint_dir}\"")
            save_checkpoint(
                self.model,
                self.latest_checkpoint_dir,
                is_saving_process=self.training_args.should_save,
                state_dict=model_parameters,
                synchronize=True
            )

        self.log("finished") 