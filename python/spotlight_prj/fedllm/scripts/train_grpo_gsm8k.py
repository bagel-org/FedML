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

ANS = re.compile(r"####\s*([-+]?\d+\.?\d*)")
def reward_fn(completions, samples, **_):
    out = []
    for c, s in zip(completions, samples):
        tru = ANS.search(s["answer"])
        pred = ANS.search(c)
        out.append(1.0 if pred and tru and pred.group(1)==tru.group(1) else -0.2)
    return out

cfg = GRPOConfig(
    output_dir=OUT,
    per_device_batch_size=1,
    gradient_accumulation_steps=16,
    max_length=1024,
    max_new_tokens=256,
    num_train_epochs=3,
    generate_num_samples=4,     # “group” size
    learning_rate=5e-6,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=25,
)

policy = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
trainer = GRPOTrainer(
    model=policy,
    args=cfg,
    train_dataset=ds.shuffle(seed=42),
    tokenizer=tok,
    reward_funcs=reward_fn,
    reference_model=BASE,
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
