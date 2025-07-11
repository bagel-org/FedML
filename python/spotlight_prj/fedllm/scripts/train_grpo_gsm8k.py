import os, re, torch, shutil
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

RUN_ID = os.environ["RUN_ID"]
OUT = f".logs/FedML/{RUN_ID}/node_1/grpo"
BASE = "Qwen/Qwen3-0.6B"            # policy & reference

tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.eos_token

ds = load_dataset("openai/gsm8k", "main", split="train")
ds = ds.rename_column("question", "prompt")

# Regex for GSM8K dataset format (####)
DATASET_ANS = re.compile(r"####\s*([-+]?\d+\.?\d*)")
# Regex for model completion format (\boxed{})
MODEL_ANS = re.compile(r"\\boxed\{([^}]*)\}")

def reward_fn(completions, answer, **_):
    out = []
    for c, ans in zip(completions, answer):
        # Extract from dataset answer (GSM8K format)
        tru = DATASET_ANS.search(ans)
        # Extract from model completion (boxed format, fallback to GSM8K format)
        pred = MODEL_ANS.search(c)
        if not pred:
            pred = DATASET_ANS.search(c)
        
        if pred and tru:
            pred_num = pred.group(1)
            tru_num = tru.group(1)
            out.append(1.0 if pred_num == tru_num else -0.2)
        else:
            out.append(-0.2)
    return out


cfg = GRPOConfig(
    output_dir=OUT,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    max_completion_length=1024,
    num_generations=8,     # "group" size
    num_train_epochs=3,
    learning_rate=5e-6,
    bf16=True,
    gradient_checkpointing=False,
    logging_steps=25,
    log_completions=True,
)

policy = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, use_cache=False)
trainer = GRPOTrainer(
    model=policy,
    args=cfg,
    train_dataset=ds.shuffle(seed=42),
    processing_class=tok,
    reward_funcs=reward_fn,
)
trainer.train()

# expose the last checkpoint one directory up so FedML syncs it
last = max(
    (p for p in os.listdir(OUT) if p.startswith("checkpoint-")),
    key=lambda x: int(x.split("-")[-1])
)
shutil.copytree(f"{OUT}/{last}",
                f".logs/FedML/{RUN_ID}/node_1/{last}",
                dirs_exist_ok=True)
