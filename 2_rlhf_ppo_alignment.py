"""
RLHF Alignment: PPO with human-centric safety, toxicity & sentiment constraints.
Uses TRL (Transformer Reinforcement Learning) library.
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
from trl import (
    PPOConfig,
    PPOTrainer,
    AutoModelForCausalLMWithValueHead,
    create_reference_model,
)
from peft import LoraConfig, TaskType

# ── Config ────────────────────────────────────────────────────────────────────
SFT_MODEL       = "lvwerra/gpt2-imdb"      # lightweight SFT checkpoint for demo
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
TOXICITY_MODEL  = "s-nlp/roberta_toxicity_classifier"
MAX_NEW_TOKENS  = 100
BATCH_SIZE      = 16
MINI_BATCH_SIZE = 4
PPO_EPOCHS      = 4
LR              = 1.4e-5
KL_COEFF        = 0.2       # penalty weight to stay close to reference policy

# ── 1. Model & tokenizer setup ────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# Policy model with a value head for PPO critic
lora_config = LoraConfig(
    task_type    = TaskType.CAUSAL_LM,
    r            = 16,
    lora_alpha   = 32,
    lora_dropout = 0.05,
    bias         = "none",
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    SFT_MODEL,
    peft_config = lora_config,
)
ref_model = create_reference_model(model)   # frozen reference for KL divergence

# ── 2. Reward models (human-centric constraints) ──────────────────────────────
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model     = SENTIMENT_MODEL,
    device    = 0 if torch.cuda.is_available() else -1,
    top_k     = None,
)

toxicity_pipeline = pipeline(
    "text-classification",
    model     = TOXICITY_MODEL,
    device    = 0 if torch.cuda.is_available() else -1,
    top_k     = None,
)

def compute_reward(responses: list[str]) -> list[torch.Tensor]:
    """
    Composite reward signal:
      +1.0  if sentiment == POSITIVE
      -2.0  if toxicity score > 0.5   (hard safety penalty)
       0.0  otherwise
    """
    rewards = []
    sentiment_results = sentiment_pipeline(responses, truncation=True, max_length=256)
    toxicity_results  = toxicity_pipeline(responses,  truncation=True, max_length=256)

    for sent, tox in zip(sentiment_results, toxicity_results):
        # sentiment_results returns list of label/score dicts per sample
        sent_score = next(s["score"] for s in sent if s["label"] == "POSITIVE")
        tox_score  = next(t["score"] for t in tox  if t["label"] in ("toxic", "LABEL_1"))

        reward = sent_score                       # base: positive sentiment
        if tox_score > 0.5:
            reward -= 2.0 * tox_score            # safety penalty

        rewards.append(torch.tensor(reward, dtype=torch.float))
    return rewards

# ── 3. Dataset ────────────────────────────────────────────────────────────────
dataset = load_dataset("imdb", split="train[:2000]")

def tokenize(sample):
    # Use only the first 128 tokens as prompt context
    encoded = tokenizer(sample["text"], truncation=True, max_length=128, padding=False)
    sample["input_ids"] = encoded["input_ids"]
    return sample

dataset = dataset.map(tokenize)
dataset.set_format("torch")

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

# ── 4. PPO Trainer ────────────────────────────────────────────────────────────
ppo_config = PPOConfig(
    model_name        = SFT_MODEL,
    learning_rate     = LR,
    batch_size        = BATCH_SIZE,
    mini_batch_size   = MINI_BATCH_SIZE,
    ppo_epochs        = PPO_EPOCHS,
    init_kl_coef      = KL_COEFF,
    target_kl         = 6.0,            # adaptive KL target
    adap_kl_ctrl      = True,
    log_with          = None,
)

ppo_trainer = PPOTrainer(
    config        = ppo_config,
    model         = model,
    ref_model     = ref_model,
    tokenizer     = tokenizer,
    dataset       = dataset,
    data_collator = collator,
)

# ── 5. PPO Training Loop ──────────────────────────────────────────────────────
gen_kwargs = {
    "min_length"   : -1,
    "top_k"        : 0,
    "top_p"        : 1.0,
    "do_sample"    : True,
    "max_new_tokens": MAX_NEW_TOKENS,
    "pad_token_id" : tokenizer.eos_token_id,
}

for epoch, batch in enumerate(ppo_trainer.dataloader):
    query_tensors = batch["input_ids"]

    # Policy generates responses
    response_tensors = ppo_trainer.generate(
        query_tensors, return_prompt=False, **gen_kwargs
    )
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # Compute composite reward
    rewards = compute_reward(batch["response"])

    # PPO update step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    if epoch % 10 == 0:
        avg_reward = torch.stack(rewards).mean().item()
        print(f"Epoch {epoch:4d} | avg_reward={avg_reward:.4f} | kl={stats['objective/kl']:.4f}")

# ── 6. Save aligned model ─────────────────────────────────────────────────────
ppo_trainer.save_pretrained("./rlhf-ppo-aligned-model")
print("RLHF-aligned model saved.")
