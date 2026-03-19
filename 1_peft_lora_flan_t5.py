"""
PEFT & Fine-Tuning: LoRA on FLAN-T5 for Dialogue Summarization
Validates performance using ROUGE metrics.
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
import evaluate
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME   = "google/flan-t5-base"
DATASET_NAME = "knkarthick/dialogsum"
OUTPUT_DIR   = "./flan-t5-lora-dialogsum"
MAX_INPUT    = 512
MAX_TARGET   = 128
BATCH_SIZE   = 8
EPOCHS       = 3
LR           = 3e-4

# ── 1. Load tokenizer & model ────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)

# ── 2. Attach LoRA adapters ──────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type  = TaskType.SEQ_2_SEQ_LM,
    r          = 16,          # rank of the update matrices
    lora_alpha = 32,          # scaling factor
    lora_dropout = 0.05,
    bias       = "none",
    target_modules = ["q", "v"],   # attention query & value projections
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: ~2.4M / total: 250M → ~0.96% — massive reduction in overhead

# ── 3. Dataset preprocessing ─────────────────────────────────────────────────
dataset = load_dataset(DATASET_NAME)

def preprocess(batch):
    inputs  = ["Summarize the following dialogue:\n\n" + d for d in batch["dialogue"]]
    targets = batch["summary"]

    model_inputs = tokenizer(
        inputs, max_length=MAX_INPUT, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        text_target=targets, max_length=MAX_TARGET, truncation=True, padding="max_length"
    )
    # Replace pad token id in labels with -100 so loss ignores padding
    labels["input_ids"] = [
        [(t if t != tokenizer.pad_token_id else -100) for t in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

# ── 4. ROUGE metric ──────────────────────────────────────────────────────────
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 padding in labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )
    return {k: round(v, 4) for k, v in result.items()}

# ── 5. Training arguments ────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir              = OUTPUT_DIR,
    num_train_epochs        = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    learning_rate           = LR,
    evaluation_strategy     = "epoch",
    save_strategy           = "epoch",
    load_best_model_at_end  = True,
    metric_for_best_model   = "rougeL",
    predict_with_generate   = True,
    fp16                    = False,
    bf16                    = True,
    logging_steps           = 50,
    report_to               = "none",
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = tokenized["train"],
    eval_dataset    = tokenized["validation"],
    tokenizer       = tokenizer,
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
)

# ── 6. Train ─────────────────────────────────────────────────────────────────
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ── 7. Inference with fine-tuned model ───────────────────────────────────────
def summarize(dialogue: str, model, tokenizer) -> str:
    prompt = f"Summarize the following dialogue:\n\n{dialogue}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_config = GenerationConfig(
        max_new_tokens = MAX_TARGET,
        num_beams      = 4,
        early_stopping = True,
    )
    output = model.generate(**inputs, generation_config=gen_config)
    return tokenizer.decode(output[0], skip_special_tokens=True)


# Load saved PEFT model for inference
base_model  = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
peft_model  = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
peft_model.eval()

sample_dialogue = dataset["test"][0]["dialogue"]
print("=== Generated Summary ===")
print(summarize(sample_dialogue, peft_model, tokenizer))
print("\n=== Reference Summary ===")
print(dataset["test"][0]["summary"])
