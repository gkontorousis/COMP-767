#!/usr/bin/env python3
"""
Tiny MDL-style scoring demo for Qwen on Beluga.

This script:
1) Builds a small synthetic sequence dataset from functions.py generators.
2) Defines a dummy natural-language explanation z.
3) Scores:
   - explanation complexity: -log p(z)
   - data complexity:       -sum log p(x_i | z)
4) Prints train/test metrics similar to the proposal's MDL framing.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import functions as seq_functions


@dataclass
class Example:
    task_name: str
    context: List[int]
    target: int


def take_n(generator: Iterable[int], n: int) -> List[int]:
    out = []
    for _ in range(n):
        out.append(int(next(generator)))
    return out


def sliding_window_examples(
    values: Sequence[int],
    task_name: str,
    window: int = 5,
) -> List[Example]:
    examples: List[Example] = []
    for i in range(len(values) - window):
        context = [int(v) for v in values[i : i + window]]
        target = int(values[i + window])
        examples.append(Example(task_name=task_name, context=context, target=target))
    return examples


def build_tiny_dataset(points_per_task: int = 14, window: int = 5) -> Tuple[List[Example], List[Example]]:
    tasks = [
        ("arithmetic_2x_plus1", seq_functions.artihmetic(2, 1)),
        ("fibonacci_start11", seq_functions.fibonacci_starting_at_11()),
        ("odd_even_squares", seq_functions.odd_even_squares()),
    ]

    all_examples: List[Example] = []
    for task_name, gen in tasks:
        vals = take_n(gen, points_per_task)
        all_examples.extend(sliding_window_examples(vals, task_name=task_name, window=window))

    # Small train/test split for fast sanity checks.
    split = int(0.75 * len(all_examples))
    train = all_examples[:split]
    test = all_examples[split:]
    return train, test


def format_prompt(explanation: str, ex: Example) -> str:
    seq_str = ", ".join(str(x) for x in ex.context)
    return (
        "You are evaluating a hypothesized rule for integer sequences.\n"
        f"Rule explanation: {explanation}\n"
        f"Task tag: {ex.task_name}\n"
        f"Observed sequence prefix: {seq_str}\n"
        "Predict only the next integer.\n"
        "Answer:"
    )


def continuation_nll(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix: str,
    continuation: str,
    device: torch.device,
) -> float:
    """
    Returns -log p(continuation | prefix) in natural log units.
    """
    prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(prefix + continuation, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    if full_ids.shape[1] <= prefix_ids.shape[1]:
        return 0.0

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits[:, :-1, :]
        labels = full_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    start = max(prefix_ids.shape[1] - 1, 0)
    continuation_log_prob_sum = token_log_probs[:, start:].sum().item()
    return -float(continuation_log_prob_sum)


def score_explanation_complexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    explanation: str,
    device: torch.device,
) -> float:
    # Approximation of -log p(z) with an empty prefix.
    return continuation_nll(model, tokenizer, prefix="", continuation=explanation, device=device)


def score_dataset_given_explanation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    explanation: str,
    dataset: Sequence[Example],
    device: torch.device,
) -> float:
    total = 0.0
    for ex in dataset:
        prefix = format_prompt(explanation, ex)
        continuation = f" {ex.target}"
        total += continuation_nll(model, tokenizer, prefix=prefix, continuation=continuation, device=device)
    return total


def format_example(ex: Example) -> str:
    context_str = ", ".join(str(x) for x in ex.context)
    return f"{ex.task_name}: [{context_str}] -> {ex.target}"


def print_dataset_preview(train_set: Sequence[Example], test_set: Sequence[Example], n: int = 3) -> None:
    print("\n=== Dataset Preview ===")
    print(f"Train preview ({min(n, len(train_set))} examples):")
    for ex in train_set[:n]:
        print(f"  - {format_example(ex)}")
    print(f"Test preview ({min(n, len(test_set))} examples):")
    for ex in test_set[:n]:
        print(f"  - {format_example(ex)}")


def build_explanation_prompt(dataset: Sequence[Example], n: int = 6) -> str:
    lines = [
        "You are given integer-sequence continuation examples.",
        "Write a short natural-language explanation of the rule family behind these examples.",
        "Be concise and predictive. Return explanation text only.",
        "",
        "Examples:",
    ]
    for ex in dataset[:n]:
        lines.append(f"- {format_example(ex)}")
    lines.append("")
    lines.append("Explanation:")
    return "\n".join(lines)


def generate_candidate_explanations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Sequence[Example],
    device: torch.device,
    num_candidates: int = 3,
    prompt_examples: int = 6,
    max_new_tokens: int = 80,
    temperature: float = 0.9,
    top_p: float = 0.95,
) -> List[str]:
    prompt = build_explanation_prompt(dataset, n=prompt_examples)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated: List[str] = []

    for _ in range(num_candidates):
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        continuation_ids = output_ids[0, input_ids.shape[1] :]
        text = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
        if text:
            generated.append(text.split("\n\n")[0].strip())
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Dummy MDL scoring pipeline with Qwen on tiny synthetic data.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF model id (set to your Qwen 8B variant if different).",
    )
    parser.add_argument("--points_per_task", type=int, default=14, help="How many raw points to sample per generator.")
    parser.add_argument("--window", type=int, default=5, help="Context length for next-step prediction.")
    parser.add_argument(
        "--explanation",
        type=str,
        default=(
            "Each task follows a deterministic arithmetic-style rule over previous terms. "
            "Infer the local pattern from the prefix and output the next integer only."
        ),
        help="Dummy linguistic explanation z.",
    )
    parser.add_argument("--preview_examples", type=int, default=4, help="Number of train/test examples to print.")
    parser.add_argument(
        "--num_generated_explanations",
        type=int,
        default=2,
        help="How many model-generated explanation candidates to sample and score.",
    )
    parser.add_argument(
        "--generation_prompt_examples",
        type=int,
        default=6,
        help="How many training examples to include when prompting model for an explanation.",
    )
    parser.add_argument("--max_expl_tokens", type=int, default=80, help="Max tokens for each generated explanation.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if not torch.cuda.is_available():
        model = model.to(device)
    model.eval()

    train_set, test_set = build_tiny_dataset(points_per_task=args.points_per_task, window=args.window)
    print_dataset_preview(train_set, test_set, n=args.preview_examples)

    z_complexity = score_explanation_complexity(model, tokenizer, args.explanation, device=device)
    train_data_complexity = score_dataset_given_explanation(model, tokenizer, args.explanation, train_set, device=device)
    test_data_complexity = score_dataset_given_explanation(model, tokenizer, args.explanation, test_set, device=device)

    mdl_train = z_complexity + train_data_complexity
    avg_train_nll = train_data_complexity / max(len(train_set), 1)
    avg_test_nll = test_data_complexity / max(len(test_set), 1)
    ppl_train = math.exp(avg_train_nll) if avg_train_nll < 40 else float("inf")
    ppl_test = math.exp(avg_test_nll) if avg_test_nll < 40 else float("inf")

    print("=== Dummy MDL Scoring (Qwen) ===")
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Train examples: {len(train_set)} | Test examples: {len(test_set)}")
    print(f"Explanation z: {args.explanation}")
    print()
    print(f"-log p(z):                       {z_complexity:.4f}")
    print(f"-sum log p(x_train | z):        {train_data_complexity:.4f}")
    print(f"MDL train score (two-part):     {mdl_train:.4f}")
    print(f"Avg train NLL per example:      { avg_train_nll:.4f}")
    print(f"Avg test NLL per example:       {avg_test_nll:.4f}")
    print(f"Approx train perplexity:        {ppl_train:.4f}")
    print(f"Approx test perplexity:         {ppl_test:.4f}")

    if args.num_generated_explanations > 0:
        print("\n=== Candidate Explanations From Model ===")
        candidates = generate_candidate_explanations(
            model=model,
            tokenizer=tokenizer,
            dataset=train_set,
            device=device,
            num_candidates=args.num_generated_explanations,
            prompt_examples=args.generation_prompt_examples,
            max_new_tokens=args.max_expl_tokens,
        )

        if not candidates:
            print("No explanation candidates generated.")
        for idx, cand in enumerate(candidates, start=1):
            cand_z = score_explanation_complexity(model, tokenizer, cand, device=device)
            cand_train = score_dataset_given_explanation(model, tokenizer, cand, train_set, device=device)
            cand_test = score_dataset_given_explanation(model, tokenizer, cand, test_set, device=device)
            cand_mdl = cand_z + cand_train
            print(f"\n[{idx}] Explanation:")
            print(cand)
            print(f"    -log p(z):            {cand_z:.4f}")
            print(f"    -sum log p(train|z):  {cand_train:.4f}")
            print(f"    MDL train score:      {cand_mdl:.4f}")
            print(f"    Avg test NLL/example: {cand_test / max(len(test_set), 1):.4f}")


if __name__ == "__main__":
    main()
