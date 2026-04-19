import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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

        # Cast to float32 before log_softmax for more stable scoring.
        logits = outputs.logits[:, :-1, :].float()
        labels = full_ids[:, 1:]

        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    start = prefix_ids.shape[1] - 1
    continuation_log_prob_sum = token_log_probs[:, start:].sum().item()
    continuation_token_count = full_ids.shape[1] - prefix_ids.shape[1]

    return -float(continuation_log_prob_sum), int(continuation_token_count)


def build_prior_prefix() -> str:
    """
    Plain-text prefix for the base model prior over explanations.
    """
    return (
        "Below is a concise explanation of the task or pattern in the data.\n\n"
        "Explanation:\n"
    )


def build_action_likelihood_prefix(task_description: str) -> str:
    """
    Plain-text prefix for scoring an action sequence under a candidate task.
    """
    return (
        "ScienceWorld task description:\n"
        f"{task_description}\n\n"
        "Below is a sequence of actions that an agent might take to solve the task.\n"
        "The sequence should be specifically relevant to the task, not just a generic household routine.\n"
        "Each line is one action and begins with '> '.\n\n"
        "Action sequence:\n"
    )


def score_prior(model, tokenizer, task_description: str, device):
    """
    Prior loss: -log p(task_description | prior prefix)
    """
    prefix = build_prior_prefix()
    nll, token_count = continuation_nll_and_token_count(
        model=model,
        tokenizer=tokenizer,
        prefix=prefix,
        continuation=task_description,
        device=device,
    )
    return {
        "nll": nll,
        "token_count": token_count,
        "avg_nll_per_token": nll / max(token_count, 1),
    }


def split_task_and_rollout(full_trace: str):
    """
    Splits a full ScienceWorld trace into:
      - task description
      - rollout text
    """
    lines = full_trace.splitlines()
    if not lines or not lines[0].startswith("Task Description:"):
        raise ValueError("Trace must start with 'Task Description:'")

    task_line = lines[0]
    task_description = task_line.replace("Task Description:", "", 1).strip()
    rollout = "\n".join(lines[1:]).lstrip()
    return task_description, rollout


def extract_actions_from_rollout(rollout_text: str):
    """
    Extract only action lines from a ScienceWorld rollout.

    Returns a list like:
      ["> look around", "> go to kitchen", ...]
    """
    actions = []
    for raw_line in rollout_text.splitlines():
        line = raw_line.strip()
        if line.startswith("> "):
            actions.append(line)
    return actions


def actions_to_continuation(actions):
    """
    Convert action list into a continuation string.
    """
    if not actions:
        return ""
    return "\n".join(actions)


def score_action_sequence(model, tokenizer, task_description: str, actions, device):
    """
    Score only the action sequence under the candidate task description.
    """
    prefix = build_action_likelihood_prefix(task_description)
    continuation = actions_to_continuation(actions)

    total_nll, total_action_tokens = continuation_nll_and_token_count(
        model=model,
        tokenizer=tokenizer,
        prefix=prefix,
        continuation=continuation,
        device=device,
    )

    num_actions = len(actions)

    return {
        "total_nll": total_nll,
        "num_actions": num_actions,
        "total_action_tokens": total_action_tokens,
        "avg_nll_per_action": total_nll / max(num_actions, 1),
        "avg_nll_per_token": total_nll / max(total_action_tokens, 1),
    }


def score_likelihood(model, tokenizer, task_description: str, rollout_text: str, device):
    """
    Likelihood based only on the extracted action sequence.
    Observations are ignored.
    """
    actions = extract_actions_from_rollout(rollout_text)
    return score_action_sequence(
        model=model,
        tokenizer=tokenizer,
        task_description=task_description,
        actions=actions,
        device=device,
    )


def print_score_summary(name: str, prior_info: dict, likelihood_info: dict):
    print(f"\n=== {name} ===")
    print(f"Prior loss (-log p(z)):                 {prior_info['nll']:.4f}")
    print(f"Prior avg loss / token:                {prior_info['avg_nll_per_token']:.4f}")
    print(f"Likelihood loss (-log p(actions|z)):   {likelihood_info['total_nll']:.4f}")
    print(f"Num actions:                           {likelihood_info['num_actions']}")
    print(f"Total action tokens:                   {likelihood_info['total_action_tokens']}")
    print(f"Avg likelihood loss / action:          {likelihood_info['avg_nll_per_action']:.4f}")
    print(f"Avg likelihood loss / token:           {likelihood_info['avg_nll_per_token']:.4f}")


def print_likelihood_ranking(results):
    """
    Rank candidates by likelihood only.
    Lower is better.
    """
    ranked = sorted(results.items(), key=lambda kv: kv[1]["likelihood"]["avg_nll_per_token"])

    print("\n=== Likelihood-only ranking (lower avg loss/token is better) ===")
    for i, (name, info) in enumerate(ranked, start=1):
        print(
            f"{i}. {name:<8} "
            f"avg_nll/token={info['likelihood']['avg_nll_per_token']:.4f} | "
            f"total_nll={info['likelihood']['total_nll']:.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-14B-Base",
        help=(
            "Model identifier. If --model-path is set, this should be the subdirectory "
            "name under that folder (e.g. Qwen3-14B-Base). If --model-path is not set, "
            "this is passed directly to from_pretrained (often a Hub repo id)."
        ),
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Optional parent directory of locally downloaded models. The resolved model "
            "directory is MODEL_PATH / MODEL_NAME."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer only from local files instead of the network.",
    )
    return parser.parse_args()


def load_model_and_tokenizer(args, dtype, device):
    if args.model_path is not None:
        if ("/" in args.model_name) or ("\\" in args.model_name):
            raise ValueError(
                "When --model-path is set, --model-name must be a single folder name "
                f"(no path separators). Got: {args.model_name!r}"
            )
        model_source = str(Path(args.model_path).expanduser().resolve() / args.model_name)
    else:
        model_source = args.model_name

    load_kwargs = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if args.cache_dir is not None:
        load_kwargs["cache_dir"] = args.cache_dir

    print(f"Model source: {model_source}")
    if args.model_path is not None:
        print(f"Models root: {args.model_path}")
        print(f"Model folder: {args.model_name}")
    print(f"Local files only: {args.local_files_only}")
    if args.cache_dir is not None:
        print(f"Cache dir: {args.cache_dir}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source, **load_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            **load_kwargs,
        )
    except OSError as exc:
        hint_lines = [
            "",
            "Model loading failed.",
            "If this is running on Narval, the compute node likely has no outbound network access.",
            "Use one of these options:",
            "1. Pre-download the model on a login node or other machine with internet access.",
            "2. Pass --model-path to the parent folder and --model-name to the per-model folder.",
            "3. Re-run with --local-files-only so transformers does not try the network.",
            "4. If you use a cache, point --cache-dir at the same cache location.",
        ]
        raise RuntimeError("\n".join(hint_lines)) from exc

    if not torch.cuda.is_available():
        model = model.to(device)

    model.eval()
    return tokenizer, model


def main():
    args = parse_args()

    full_trace = """
Task Description: Your task is to change the state of matter of water. First, focus on the substance. Then, take actions that will cause it to change its state of matter.

> look around
This room is called the hallway. In it, you see: 
	the agent
	a substance called air
	a picture
You also see:
	A door to the green house (that is open)
	A door to the living room (that is open)
	A door to the art studio (that is open)
	A door to the kitchen (that is open)
	A door to the bedroom (that is open)
	A door to the workshop (that is open)

> open door to kitchen
The door is already open.

> go to kitchen
You move to the kitchen.

> look around
This room is called the kitchen. In it, you see: 
	the agent
	a sink, which is turned off. In the sink is: nothing.
	a freezer. The freezer door is closed. 
	a substance called air
	a stopwatch, which is deactivated. 
	a counter. On the counter is: a bowl (containing a banana, an orange, a potato, a red apple), a drawer.
	a oven, which is turned off. The oven door is closed. 
	a cupboard. The cupboard door is closed. 
	a glass jar (containing a substance called sodium chloride)
	a chair. On the chair is: nothing.
	a thermometer, currently reading a temperature of 10 degrees celsius
	a painting
	a stove, which is turned off. On the stove is: nothing.
	a substance called soap
	a lighter
	a fridge. The fridge door is closed. 
	a table. On the table is: a glass cup (containing nothing).
You also see:
	A door to the hallway (that is open)
	A door to the bathroom (that is open)
	A door to the outside (that is open)

> pick up thermometer
You move the thermometer to the inventory.

> open cupboard
The cupboard is now open.

> pick up metal pot
You move the metal pot to the inventory.

> look around
This room is called the kitchen. In it, you see: 
	the agent
	a sink, which is turned off. In the sink is: nothing.
	a freezer. The freezer door is closed. 
	a substance called air
	a stopwatch, which is deactivated. 
	a counter. On the counter is: a bowl (containing a banana, an orange, a potato, a red apple), a drawer.
	a oven, which is turned off. The oven door is closed. 
	a cupboard. The cupboard door is open. In the cupboard is: a ceramic cup (containing nothing), a drawer, a tin cup (containing nothing).
	a glass jar (containing a substance called sodium chloride)
	a chair. On the chair is: nothing.
	a painting
	a stove, which is turned off. On the stove is: nothing.
	a substance called soap
	a lighter
	a fridge. The fridge door is closed. 
	a table. On the table is: a glass cup (containing nothing).
You also see:
	A door to the hallway (that is open)
	A door to the bathroom (that is open)
	A door to the outside (that is open)

> move metal pot to sink
You move the metal pot to the sink.

> activate sink
The sink is now activated.

> deactivate sink
The sink is now deactivated.

> pick up metal pot
You move the metal pot to the inventory.

> focus on substance in metal pot
You focus on the water.

> pour metal pot into metal pot
You can't move something into itself.

> pick up metal pot
You move the metal pot to the inventory.

> open freezer
The freezer is now open.

> move metal pot to freezer
You move the metal pot to the freezer.

> examine substance in metal pot
a substance called water

> use thermometer in inventory on substance in metal pot
the thermometer measures a temperature of 10 degrees celsius

> wait
You decide to wait for 10 iterations.

> examine substance in metal pot
a substance called water

> use thermometer in inventory on substance in metal pot
the thermometer measures a temperature of 3 degrees celsius

> wait
You decide to wait for 10 iterations. (Task Completed)

> examine ice
a substance called ice (Task Completed)

> use thermometer in inventory on ice
the thermometer measures a temperature of -4 degrees celsius (Task Completed)

> wait
You decide to wait for 10 iterations. (Task Completed)

> wait1
You decide to wait for 1 iterations. (Task Completed)
""".strip()

    candidate_concise = (
        "Your task is to change the state of matter of water. First, focus on the "
        "substance. Then, take actions that will cause it to change its state of matter."
    )

    candidate_dummy = (
        "Go to the kitchen, pick up the thermometer, open the cupboard, pick up the "
        "metal pot, fill it with water in the sink, move it to the freezer, and wait "
        "until it freezes."
    )

    candidate_false = (
        "Your task is to grow a plant by placing a seed in soil, watering it, and "
        "leaving it in sunlight."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Using device: {device}")
    print(f"Requested model name: {args.model_name}")
    tokenizer, model = load_model_and_tokenizer(args, dtype, device)

    observed_task, observed_rollout = split_task_and_rollout(full_trace)
    actions = extract_actions_from_rollout(observed_rollout)

    print("Observed task description:")
    print(observed_task)

    print("\nExtracted actions:")
    for action in actions:
        print(action)

    candidates = {
        "CONCISE": candidate_concise,
        "DUMMY": candidate_dummy,
        "FALSE": candidate_false,
    }

    results = {}

    for name, candidate in candidates.items():
        prior_info = score_prior(model, tokenizer, candidate, device)
        likelihood_info = score_likelihood(model, tokenizer, candidate, observed_rollout, device)
        results[name] = {
            "task": candidate,
            "prior": prior_info,
            "likelihood": likelihood_info,
        }

    for name in ["CONCISE", "DUMMY", "FALSE"]:
        print(f"\n{name} task description:")
        print(results[name]["task"])
        print_score_summary(name, results[name]["prior"], results[name]["likelihood"])

    print_likelihood_ranking(results)


if __name__ == "__main__":
    main()