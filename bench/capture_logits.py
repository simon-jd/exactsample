"""Capture real decode-step logits for the sampler benchmark.

Generates `--steps` tokens for each of a fixed set of prompts with a local
Hugging Face model (temperature 0.7 sampling), and saves the raw logits of
every decode step together with the token sampled there and its absolute
position. Consecutive steps of one prompt are what a speculative-decoding
request verifies, so the benchmark builds both ordinary and draft-row
workloads from this file.

    python bench/capture_logits.py --model Qwen/Qwen3-8B --output results/raw/logits/qwen3-8b.npz
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PROMPTS = [
    "Explain why the sky is blue to a ten-year-old.",
    "Write a Python function that merges two sorted lists.",
    "What are the main causes of inflation?",
    "Summarize the plot of Hamlet in three sentences.",
    "Give me a recipe for a quick vegetarian dinner.",
    "How does a transformer neural network work?",
    "Translate 'The weather is lovely today' into French and German.",
    "What is the difference between TCP and UDP?",
    "Write a haiku about autumn rain.",
    "Prove that the square root of 2 is irrational.",
    "What should I consider when buying a used car?",
    "Describe the water cycle.",
    "Write a SQL query that finds the second highest salary.",
    "Why do cats purr?",
    "List five tips for better sleep.",
    "Explain the CAP theorem with an example.",
    "Write a short story opening set on a night train.",
    "What is the capital of Australia, and why is it not Sydney?",
    "How do vaccines train the immune system?",
    "Compare Rust and C++ for systems programming.",
    "Solve: a train travels 300 km in 2.5 hours. What is its average speed?",
    "What happened during the French Revolution?",
    "Write a bash one-liner that counts lines in all .py files.",
    "Give three arguments for and against remote work.",
    "Explain gradient descent in simple terms.",
    "What is a black hole?",
    "Draft a polite email declining a meeting invitation.",
    "How does public-key cryptography work?",
    "What are good habits for learning a new language?",
    "Explain the rules of chess castling.",
    "Write a limerick about a programmer.",
    "What is the role of mitochondria in a cell?",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    texts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False) for p in PROMPTS]
    batch = tokenizer(texts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**batch, do_sample=True, temperature=0.7, top_k=0, top_p=1.0,
                             max_new_tokens=args.steps, min_new_tokens=args.steps,
                             output_logits=True, return_dict_in_generate=True)
    # out.logits[s] holds the raw (unprocessed) logits that produced token s.
    logits = torch.stack(list(out.logits), dim=1)  # [prompts, steps, V]
    sampled = out.sequences[:, batch["input_ids"].shape[1]:]
    prompt_lengths = batch["attention_mask"].sum(dim=1)
    prompts, steps, vocab = logits.shape
    positions = prompt_lengths[:, None] + torch.arange(steps, device="cuda")[None, :]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output,
             logits_bf16=logits.to(torch.bfloat16).reshape(-1, vocab).view(torch.int16).cpu().numpy(),
             sequence=np.repeat(np.arange(prompts), steps), step=np.tile(np.arange(steps), prompts),
             position=positions.reshape(-1).cpu().numpy(), sampled=sampled.reshape(-1).cpu().numpy())
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    revision = next((part for part in Path(model.config._name_or_path).parts if len(part) == 40), None)
    meta = {"model": args.model, "revision": revision, "transformers": transformers.__version__,
            "torch": torch.__version__, "prompts": PROMPTS, "steps": steps, "vocab": vocab,
            "sampling": {"temperature": 0.7, "seed": args.seed}, "rows": prompts * steps, "sha256": digest}
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "prompts"}))


if __name__ == "__main__":
    main()
