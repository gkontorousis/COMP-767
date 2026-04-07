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
        logits = outputs.logits[:, :-1, :]
        labels = full_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    # token_log_probs[i] corresponds to labels[i], i.e. tokens after the first token
    # We want only the contribution from the continuation tokens.
    start = prefix_ids.shape[1] - 1
    continuation_log_prob_sum = token_log_probs[:, start:].sum().item()
    continuation_token_count = full_ids.shape[1] - prefix_ids.shape[1]

    return -float(continuation_log_prob_sum), int(continuation_token_count)


def qwen_render_messages(tokenizer, messages, add_generation_prompt: bool) -> str:
    """
    Render a Qwen chat conversation into plain text for scoring/generation.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def build_prior_system_prompt() -> str:
    return "Generate a task to test an agent's science skills."


def build_agent_system_prompt(task_description: str) -> str:
    return (
        "You are a ScienceWorld agent.\n\n"
        f"Task Description: {task_description}\n\n"
        "You will interact with a text environment.\n"
        "Environment observations will be provided as user messages.\n"
        "You must respond only with the next action.\n\n"
        "Formatting rules:\n"
        '- Output exactly one action.\n'
        '- The action must begin with "> ".\n'
        "- Do not explain your reasoning.\n"
        "- Do not summarize.\n"
        "- Do not output observations.\n"
        "- Stay in ScienceWorld action format.\n"
    )


def score_prior(model, tokenizer, z: str, device) -> float:
    """
    Prior loss: -log p(z | system prior prompt)
    """
    messages = [{"role": "system", "content": build_prior_system_prompt()}]
    prefix = qwen_render_messages(tokenizer, messages, add_generation_prompt=True)

    nll, _ = continuation_nll_and_token_count(
        model=model,
        tokenizer=tokenizer,
        prefix=prefix,
        continuation=z,
        device=device,
    )
    return nll


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


def parse_scienceworld_rollout(rollout_text: str):
    """
    Parses rollout text into a list of (action, observation) pairs.

    Expected format:
      > action
      observation line 1
      observation line 2
      ...
      > next action
      next observation
      ...

    Returns:
      pairs = [
        {"action": "> look around", "observation": "This room is called ..."},
        ...
      ]
    """
    lines = rollout_text.splitlines()

    pairs = []
    current_action = None
    current_observation_lines = []

    for raw_line in lines:
        line = raw_line.rstrip()

        if line.startswith("> "):
            if current_action is not None:
                pairs.append({
                    "action": current_action,
                    "observation": "\n".join(current_observation_lines).strip(),
                })
            current_action = line.strip()
            current_observation_lines = []
        else:
            if current_action is None:
                # Ignore stray text before first action
                continue
            current_observation_lines.append(line)

    if current_action is not None:
        pairs.append({
            "action": current_action,
            "observation": "\n".join(current_observation_lines).strip(),
        })

    return pairs


def score_action_trajectory(model, tokenizer, task_description: str, action_observation_pairs, device):
    """
    Scores only assistant action turns in an alternating chat:

      system: task + format instructions
      assistant: > action_1      [scored]
      user: observation_1        [not scored]
      assistant: > action_2      [scored]
      user: observation_2        [not scored]
      ...

    Returns a dict with total and average losses.
    """
    messages = [
        {"role": "system", "content": build_agent_system_prompt(task_description)}
    ]

    total_nll = 0.0
    total_action_tokens = 0
    turn_losses = []

    for step_idx, pair in enumerate(action_observation_pairs):
        action = pair["action"].strip()
        observation = pair["observation"].strip()

        # Score the next assistant action given the full prior conversation.
        prefix = qwen_render_messages(tokenizer, messages, add_generation_prompt=True)
        action_text = action  # score exactly the action, no observation

        action_nll, action_token_count = continuation_nll_and_token_count(
            model=model,
            tokenizer=tokenizer,
            prefix=prefix,
            continuation=action_text,
            device=device,
        )

        total_nll += action_nll
        total_action_tokens += action_token_count
        turn_losses.append({
            "step": step_idx + 1,
            "action": action,
            "nll": action_nll,
            "num_tokens": action_token_count,
            "avg_nll_per_token": action_nll / max(action_token_count, 1),
        })

        # Add the gold action to history
        messages.append({"role": "assistant", "content": action})

        # Add the environment observation to history, but do not score it
        if observation:
            messages.append({"role": "user", "content": observation})

    num_actions = len(action_observation_pairs)
    avg_nll_per_action = total_nll / max(num_actions, 1)
    avg_nll_per_token = total_nll / max(total_action_tokens, 1)

    return {
        "total_nll": total_nll,
        "num_actions": num_actions,
        "total_action_tokens": total_action_tokens,
        "avg_nll_per_action": avg_nll_per_action,
        "avg_nll_per_token": avg_nll_per_token,
        "turn_losses": turn_losses,
    }


def score_likelihood(model, tokenizer, z: str, rollout_text: str, device):
    """
    Likelihood = sum of action-turn losses only.
    Observations are used as context but are not scored.
    """
    action_observation_pairs = parse_scienceworld_rollout(rollout_text)
    return score_action_trajectory(
        model=model,
        tokenizer=tokenizer,
        task_description=z,
        action_observation_pairs=action_observation_pairs,
        device=device,
    )


def print_score_summary(name: str, prior_loss: float, likelihood_info: dict):
    total_loss = prior_loss + likelihood_info["total_nll"]

    print(f"\n=== {name} ===")
    print(f"Prior loss (-log p(z)):                 {prior_loss:.4f}")
    print(f"Likelihood loss (-log p(actions|ctx)): {likelihood_info['total_nll']:.4f}")
    print(f"Total loss:                            {total_loss:.4f}")
    print(f"Num actions:                           {likelihood_info['num_actions']}")
    print(f"Total action tokens:                   {likelihood_info['total_action_tokens']}")
    print(f"Avg likelihood loss / action:          {likelihood_info['avg_nll_per_action']:.4f}")
    print(f"Avg likelihood loss / token:           {likelihood_info['avg_nll_per_token']:.4f}")


def main():
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    # Replace this with your real trace if you want.
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

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if not torch.cuda.is_available():
        model = model.to(device)

    model.eval()

    observed_task, observed_rollout = split_task_and_rollout(full_trace)
    parsed_pairs = parse_scienceworld_rollout(observed_rollout)

    print("Observed task description:")
    print(observed_task)

    print("\nParsed action-observation pairs:")
    for i, pair in enumerate(parsed_pairs[:5], start=1):
        print(f"\nStep {i}")
        print(f"Action: {pair['action']}")
        print(f"Observation preview: {pair['observation'][:200]}")

    concise_prior = score_prior(model, tokenizer, candidate_concise, device)
    concise_likelihood = score_likelihood(model, tokenizer, candidate_concise, observed_rollout, device)

    dummy_prior = score_prior(model, tokenizer, candidate_dummy, device)
    dummy_likelihood = score_likelihood(model, tokenizer, candidate_dummy, observed_rollout, device)

    false_prior = score_prior(model, tokenizer, candidate_false, device)
    false_likelihood = score_likelihood(model, tokenizer, candidate_false, observed_rollout, device)

    print("\nConcise task description:")
    print(candidate_concise)
    print_score_summary("CONCISE", concise_prior, concise_likelihood)

    print("\nDummy / overfit task description:")
    print(candidate_dummy)
    print_score_summary("DUMMY", dummy_prior, dummy_likelihood)

    print("\nFalse task description:")
    print(candidate_false)
    print_score_summary("FALSE", false_prior, false_likelihood)

    print("\nFirst few per-action losses for CONCISE:")
    for item in concise_likelihood["turn_losses"][:10]:
        print(
            f"step={item['step']:02d} | "
            f"nll={item['nll']:.4f} | "
            f"avg/token={item['avg_nll_per_token']:.4f} | "
            f"action={item['action']}"
        )


if __name__ == "__main__":
    main()