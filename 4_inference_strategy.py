"""
Inference Strategy:
  - Zero-shot & few-shot prompt engineering
  - Decoding configuration tuning (temperature, top-k, top-p, beam search)
  - Structured comparison of strategies for generative accuracy
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from dataclasses import dataclass, field
from typing import Optional
import textwrap

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"   # swap for any causal LM

@dataclass
class DecodingConfig:
    """Encapsulates a named decoding strategy for easy comparison."""
    name             : str
    max_new_tokens   : int   = 256
    do_sample        : bool  = True
    temperature      : float = 1.0
    top_k            : int   = 50
    top_p            : float = 1.0
    num_beams        : int   = 1
    repetition_penalty: float = 1.0
    early_stopping   : bool  = False

# Pre-defined decoding strategies
DECODING_STRATEGIES = [
    DecodingConfig(
        name="greedy",
        do_sample=False, temperature=1.0, top_k=0, top_p=1.0, num_beams=1,
    ),
    DecodingConfig(
        name="beam_search_4",
        do_sample=False, num_beams=4, early_stopping=True,
    ),
    DecodingConfig(
        name="temperature_0.3_precise",
        do_sample=True, temperature=0.3, top_k=40, top_p=0.9,
    ),
    DecodingConfig(
        name="temperature_0.7_balanced",
        do_sample=True, temperature=0.7, top_k=50, top_p=0.92,
    ),
    DecodingConfig(
        name="temperature_1.2_creative",
        do_sample=True, temperature=1.2, top_k=0, top_p=0.95,
    ),
    DecodingConfig(
        name="nucleus_sampling_top_p_0.9",
        do_sample=True, temperature=0.8, top_k=0, top_p=0.9,
    ),
]

# ── 1. Model loading ──────────────────────────────────────────────────────────
def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype = torch.float16,
        device_map  = "auto",
    )
    model.eval()
    return model, tokenizer

# ── 2. Zero-shot prompting ────────────────────────────────────────────────────
def zero_shot_prompt(task: str, input_text: str) -> str:
    """Direct task specification with no examples."""
    return textwrap.dedent(f"""
        [INST] {task}

        Input: {input_text}
        [/INST]
    """).strip()

# ── 3. Few-shot prompting ─────────────────────────────────────────────────────
@dataclass
class FewShotExample:
    input  : str
    output : str

def few_shot_prompt(
    task       : str,
    examples   : list[FewShotExample],
    input_text : str,
    n_shots    : int = 3,
) -> str:
    """Build a few-shot prompt from a pool of examples."""
    shots = examples[:n_shots]
    example_block = "\n\n".join(
        f"Input: {ex.input}\nOutput: {ex.output}" for ex in shots
    )
    return textwrap.dedent(f"""
        [INST] {task}

        Here are some examples:

        {example_block}

        Now complete this one:
        Input: {input_text}
        Output: [/INST]
    """).strip()

# ── 4. Chain-of-thought (CoT) prompting ───────────────────────────────────────
def chain_of_thought_prompt(task: str, input_text: str) -> str:
    return textwrap.dedent(f"""
        [INST] {task}
        Think step by step before giving your final answer.

        Input: {input_text}
        Let's think step by step: [/INST]
    """).strip()

# ── 5. Inference engine ───────────────────────────────────────────────────────
def generate(
    model      ,
    tokenizer  ,
    prompt     : str,
    decoding   : DecodingConfig,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_config = GenerationConfig(
        max_new_tokens    = decoding.max_new_tokens,
        do_sample         = decoding.do_sample,
        temperature       = decoding.temperature if decoding.do_sample else 1.0,
        top_k             = decoding.top_k,
        top_p             = decoding.top_p,
        num_beams         = decoding.num_beams,
        repetition_penalty= decoding.repetition_penalty,
        early_stopping    = decoding.early_stopping,
        pad_token_id      = tokenizer.eos_token_id,
    )

    with torch.no_grad():
        output_ids = model.generate(**inputs, generation_config=gen_config)

    # Decode only newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# ── 6. Strategy comparison runner ────────────────────────────────────────────
def compare_strategies(model, tokenizer, prompt: str, strategies=DECODING_STRATEGIES):
    print("=" * 70)
    print("PROMPT:\n", textwrap.fill(prompt[:300], 70), "...\n")
    results = {}
    for cfg in strategies:
        output = generate(model, tokenizer, prompt, cfg)
        results[cfg.name] = output
        print(f"[{cfg.name}]")
        print(textwrap.fill(output, 70))
        print()
    return results

# ── 7. Demo ───────────────────────────────────────────────────────────────────
SENTIMENT_EXAMPLES = [
    FewShotExample("I absolutely loved the movie!", "positive"),
    FewShotExample("The food was cold and service was rude.", "negative"),
    FewShotExample("It was okay, nothing special.", "neutral"),
    FewShotExample("Best purchase I've ever made!", "positive"),
]

CLASSIFICATION_TASK = "Classify the sentiment of the following text as positive, negative, or neutral."
TEST_INPUT          = "The product arrived on time but the instructions were confusing."

if __name__ == "__main__":
    model, tokenizer = load_model(MODEL_NAME)

    print("\n" + "="*70)
    print("ZERO-SHOT")
    zs_prompt  = zero_shot_prompt(CLASSIFICATION_TASK, TEST_INPUT)
    zs_results = compare_strategies(model, tokenizer, zs_prompt)

    print("\n" + "="*70)
    print("FEW-SHOT (3-shot)")
    fs_prompt  = few_shot_prompt(CLASSIFICATION_TASK, SENTIMENT_EXAMPLES, TEST_INPUT, n_shots=3)
    fs_results = compare_strategies(model, tokenizer, fs_prompt)

    print("\n" + "="*70)
    print("CHAIN-OF-THOUGHT")
    cot_prompt  = chain_of_thought_prompt(CLASSIFICATION_TASK, TEST_INPUT)
    cot_results = compare_strategies(
        model, tokenizer, cot_prompt,
        strategies=[DECODING_STRATEGIES[2]]  # balanced temperature for CoT
    )
