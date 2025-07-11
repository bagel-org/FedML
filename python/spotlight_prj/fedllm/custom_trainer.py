"""Custom trainer that properly handles non-PEFT models for full fine-tuning.

This fixes a bug in FedML where set_model_params tries to call set_peft_model_state_dict
even when peft_type="none" is configured, causing AttributeError for non-PEFT models.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collections import OrderedDict
from pathlib import Path
from typing import Any

from fedml.train.llm.modeling_utils import to_device
from fedml.train.llm.distributed import barrier
from peft import PeftModel

from run_fedllm import LLMTrainer, LLMAggregator, save_checkpoint
from src.peft_utils import set_peft_model_state_dict
from src.modeling_utils import load_state_dict


class FullModelLLMTrainer(LLMTrainer):
    """Custom trainer that properly handles both PEFT and non-PEFT models."""
    
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