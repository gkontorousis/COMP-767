import argparse
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


"""
MDL-style sanity check on short narrative datasets with hidden explanations.

Why this file exists:
- The ScienceWorld rollout experiment scores long action traces.
- That can hide whether the evaluator model actually understands a concise
  explanation, because most loss comes from many unrelated rollout tokens.
- Here we score only short answer spans whose prediction should depend on a
  latent belief, intention, or emotion.

This makes it easier to test whether a good explanation lowers answer loss
relative to generic or incorrect explanations before we invest in RL.
"""


@dataclass
class NarrativeExample:
    dataset_name: str
    name: str
    story: str
    question: str
    answer: str
    candidate_explanations: dict


def continuation_nll_and_token_count(model, tokenizer, prefix: str, continuation: str, device):
    """
    Returns:
      - nll = -log p(continuation | prefix)
      - num_continuation_tokens
    """
    prefix_ids = tokenizer(
        prefix,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)

    full_ids = tokenizer(
        prefix + continuation,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)
    

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits[:, :-1, :].float()
        labels = full_ids[:, 1:]

        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    start = prefix_ids.shape[1] - 1
    continuation_log_prob_sum = token_log_probs[:, start:].sum().item()
    continuation_token_count = full_ids.shape[1] - prefix_ids.shape[1]

    return -float(continuation_log_prob_sum), int(continuation_token_count)


def build_prior_prefix(example: NarrativeExample) -> str:
    return (
        f"Below is a concise hidden-state explanation for a {example.dataset_name} "
        "story-question pair.\n"
        "The explanation should focus on the key unobserved belief, intention, "
        "emotion, or causal factor that best predicts the answer.\n\n"
        "Explanation:\n"
    )


def build_answer_prefix(example: NarrativeExample, explanation: str | None = None) -> str:
    parts = [
        f"Dataset: {example.dataset_name}",
        "",
        "Story:",
        example.story.strip(),
        "",
        f"Question: {example.question.strip()}",
        "",
    ]

    if explanation is None:
        parts.extend(
            [
                "Answer the question using only the story.",
                "",
                "Answer:",
            ]
        )
    else:
        parts.extend(
            [
                "Hidden explanation:",
                explanation.strip(),
                "",
                "Answer the question using the story and the hidden explanation.",
                "",
                "Answer:",
            ]
        )

    return "\n".join(parts)


def score_prior(model, tokenizer, example: NarrativeExample, explanation: str, device):
    nll, token_count = continuation_nll_and_token_count(
        model=model,
        tokenizer=tokenizer,
        prefix=build_prior_prefix(example),
        continuation=explanation,
        device=device,
    )
    return {
        "nll": nll,
        "token_count": token_count,
        "avg_nll_per_token": nll / max(token_count, 1),
    }


def score_answer(model, tokenizer, example: NarrativeExample, explanation: str | None, device):
    nll, token_count = continuation_nll_and_token_count(
        model=model,
        tokenizer=tokenizer,
        prefix=build_answer_prefix(example, explanation=explanation),
        continuation=example.answer,
        device=device,
    )
    return {
        "nll": nll,
        "token_count": token_count,
        "avg_nll_per_token": nll / max(token_count, 1),
    }


def build_examples():
    return [
        NarrativeExample(
            dataset_name="bAbI-style false belief",
            name="milk_location_false_belief",
            story=(
                "Sarah puts the milk in the fridge before leaving for work. "
                "While Sarah is gone, Ben moves the milk to the cupboard. "
                "Sarah does not see Ben move it."
            ),
            question="Where will Sarah look for the milk first?",
            answer="the fridge",
            candidate_explanations={
                "GOOD": (
                    "Sarah still believes the milk is in the fridge because she left "
                    "before it was moved."
                ),
                "GENERIC": (
                    "This is a story about a kitchen item being moved between locations."
                ),
                "BAD": (
                    "Sarah knows Ben moved the milk, so she believes it is in the cupboard."
                ),
            },
        ),
        NarrativeExample(
            dataset_name="ToMi-style belief tracking",
            name="gift_location_belief_tracking",
            story=(
                "Lena hides a gift in the closet while Mark is watching. "
                "Mark then goes outside. "
                "While Mark is outside, Nina moves the gift from the closet to the attic. "
                "Mark does not see Nina move it."
            ),
            question="Where does Mark think the gift is now?",
            answer="the closet",
            candidate_explanations={
                "GOOD": (
                    "Mark has an outdated belief: he saw the gift placed in the closet "
                    "but did not observe the later move to the attic."
                ),
                "GENERIC": (
                    "Different people in the story know different things about the gift."
                ),
                "BAD": (
                    "Mark saw Nina move the gift, so he believes the gift is in the attic."
                ),
            },
        ),
        NarrativeExample(
            dataset_name="SocialIQA-style intention inference",
            name="recital_comfort_intention",
            story=(
                "Jordan practiced violin for weeks for the recital. "
                "During the performance, Jordan missed a note and ran backstage in tears. "
                "Alex followed a moment later and sat beside Jordan quietly."
            ),
            question="Why did Alex sit beside Jordan?",
            answer="to comfort Jordan",
            candidate_explanations={
                "GOOD": (
                    "Alex inferred that Jordan felt embarrassed and upset, so Alex wanted "
                    "to comfort Jordan."
                ),
                "GENERIC": (
                    "Alex and Jordan were both involved in the same recital."
                ),
                "BAD": (
                    "Alex wanted to criticize Jordan for making a mistake during the performance."
                ),
            },
        ),
        NarrativeExample(
            dataset_name="emotion inference narrative",
            name="late_bake_sale_emotion",
            story=(
                "Emma spent all afternoon baking cookies for the school sale. "
                "When she reached the gym, the table was empty and a sign said the sale "
                "had ended an hour earlier. "
                "Emma stood still and looked down at the tray."
            ),
            question="How did Emma probably feel?",
            answer="disappointed",
            candidate_explanations={
                "GOOD": (
                    "Emma expected to contribute to the sale, so realizing she arrived too "
                    "late would make her feel disappointed."
                ),
                "GENERIC": (
                    "Emma baked cookies and brought them to the gym."
                ),
                "BAD": (
                    "Emma was pleased that the sale had ended because now she could keep "
                    "all the cookies for herself."
                ),
            },
        ),
    ]


def print_baseline_summary(info: dict):
    print("NO_EXPLANATION baseline")
    print(f"Answer loss:             {info['answer']['nll']:.4f}")
    print(f"Answer avg loss/token:   {info['answer']['avg_nll_per_token']:.4f}")


def print_candidate_summary(label: str, info: dict):
    print(label)
    print(f"Prior loss:              {info['prior']['nll']:.4f}")
    print(f"Prior avg loss/token:    {info['prior']['avg_nll_per_token']:.4f}")
    print(f"Answer loss:             {info['answer']['nll']:.4f}")
    print(f"Answer avg loss/token:   {info['answer']['avg_nll_per_token']:.4f}")
    print(f"Two-part MDL total:      {info['mdl_total']:.4f}")
    print(f"Gain vs baseline:        {info['gain_vs_baseline']:.4f}")


def evaluate_example(model, tokenizer, example: NarrativeExample, device):
    baseline_answer = score_answer(
        model=model,
        tokenizer=tokenizer,
        example=example,
        explanation=None,
        device=device,
    )

    results = {
        "NO_EXPLANATION": {
            "answer": baseline_answer,
        }
    }

    for label, explanation in example.candidate_explanations.items():
        prior_info = score_prior(
            model=model,
            tokenizer=tokenizer,
            example=example,
            explanation=explanation,
            device=device,
        )
        answer_info = score_answer(
            model=model,
            tokenizer=tokenizer,
            example=example,
            explanation=explanation,
            device=device,
        )

        results[label] = {
            "explanation": explanation,
            "prior": prior_info,
            "answer": answer_info,
            "mdl_total": prior_info["nll"] + answer_info["nll"],
            "gain_vs_baseline": baseline_answer["nll"] - answer_info["nll"],
        }

    return results


def print_example_results(example: NarrativeExample, results: dict):
    print(f"\n{'=' * 80}")
    print(f"Example: {example.name}")
    print(f"Dataset: {example.dataset_name}")
    print(f"Story: {example.story}")
    print(f"Question: {example.question}")
    print(f"Gold answer: {example.answer}")
    print("-" * 80)

    print_baseline_summary(results["NO_EXPLANATION"])
    print("-" * 80)

    ranked = sorted(
        (
            (label, info)
            for label, info in results.items()
            if label != "NO_EXPLANATION"
        ),
        key=lambda kv: kv[1]["answer"]["avg_nll_per_token"],
    )

    for rank, (label, info) in enumerate(ranked, start=1):
        print(f"Rank {rank}")
        print(f"Explanation: {info['explanation']}")
        print_candidate_summary(label, info)
        print("-" * 80)


def print_aggregate_summary(all_results):
    grouped_answer_losses = defaultdict(list)
    grouped_total_losses = defaultdict(list)
    grouped_gains = defaultdict(list)

    for _, results in all_results:
        for label, info in results.items():
            if label == "NO_EXPLANATION":
                grouped_answer_losses[label].append(info["answer"]["avg_nll_per_token"])
                continue

            grouped_answer_losses[label].append(info["answer"]["avg_nll_per_token"])
            grouped_total_losses[label].append(info["mdl_total"])
            grouped_gains[label].append(info["gain_vs_baseline"])

    print(f"\n{'=' * 80}")
    print("Aggregate answer-likelihood summary")
    print("(Lower answer loss is better; positive gain means explanation helped.)")
    print("-" * 80)

    baseline_mean = mean(grouped_answer_losses["NO_EXPLANATION"])
    print(
        "NO_EXPLANATION "
        f"mean answer avg loss/token = {baseline_mean:.4f}"
    )

    for label in ["GOOD", "GENERIC", "BAD"]:
        answer_mean = mean(grouped_answer_losses[label])
        mdl_mean = mean(grouped_total_losses[label])
        gain_mean = mean(grouped_gains[label])
        print(
            f"{label:<8} "
            f"mean answer avg loss/token = {answer_mean:.4f} | "
            f"mean two-part MDL = {mdl_mean:.4f} | "
            f"mean gain vs baseline = {gain_mean:.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap for a faster smoke test.",
    )
    return parser.parse_args()


def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Using device: {device}")
    print(f"Loading model: {args.model_name}")

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

    examples = build_examples()
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    print(f"Number of examples: {len(examples)}")

    all_results = []
    for example in examples:
        results = evaluate_example(model, tokenizer, example, device)
        all_results.append((example, results))
        print_example_results(example, results)

    print_aggregate_summary(all_results)


def main():
    args = parse_args()
    print("dummy_scoring_new_dataset.py")
    print(f"Model name: {args.model_name}")
    print(f"Max examples: {args.max_examples}")
    print("-" * 80)
    run_evaluation(args)


if __name__ == "__main__":
    main()
