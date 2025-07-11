#!/usr/bin/env python3
"""Custom launcher for FedLLM with full model training support.

This launcher points to run_fedllm_custom.py instead of run_fedllm.py to use
the fixed trainer/aggregator classes that properly handle peft_type="none".
"""

import os
import sys
from pathlib import Path

# Get the directory of the original launch script
fedllm_dir = Path(__file__).parent
sys.path.insert(0, str(fedllm_dir))

# Import the original launch script
from launch_fedllm import main

# Monkey patch the MAIN_SCRIPT to use our custom runner
import launch_fedllm
launch_fedllm.MAIN_SCRIPT = "run_fedllm_custom.py"

if __name__ == "__main__":
    exit(main()) 