import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def qwen_system_prefix(tokenizer, content: str) -> str:
    """
    Render a Qwen chat-formatted prefix using only a system message,
    followed by an assistant generation prompt.
    """
    messages = [{"role": "system", "content": content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def continuation_nll(model, tokenizer, prefix: str, continuation: str, device) -> float:
    """
    Returns -log p(continuation | prefix)
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

    start = prefix_ids.shape[1] - 1
    continuation_log_prob_sum = token_log_probs[:, start:].sum().item()
    return -float(continuation_log_prob_sum)


def score_prior(model, tokenizer, z: str, device) -> float:
    """
    Prior loss: -log p(z)

    Interprets the prior prompt as a Qwen system instruction and scores
    the assistant continuation z.
    """
    system_content = "Generate a task to test an agent's science skills."
    prefix = qwen_system_prefix(tokenizer, system_content)

    return continuation_nll(
        model=model,
        tokenizer=tokenizer,
        prefix=prefix,
        continuation=z,
        device=device,
    )


def score_likelihood(model, tokenizer, z: str, x: str, device) -> float:
    """
    Likelihood loss: -log p(x | z)

    Here:
      z = candidate task description
      x = observed ScienceWorld rollout after the task description
    """
    system_content = (
        "Solve this ScienceWorld task.\n\n"
        f"Task Description: {z}\n\n"
        "You must follow ScienceWorld interaction format.\n"
        "Formatting expectations:\n"
        "- Output actions one per line.\n"
        "- Each action must begin exactly with \"> \".\n"
        "- After each action, the environment may return an observation.\n"
        "- Continue in the style of a ScienceWorld trajectory.\n"
        "- Do not use JSON, bullet points, or explanations unless they appear as environment text.\n"
    )
    prefix = qwen_system_prefix(tokenizer, system_content)

    return continuation_nll(
        model=model,
        tokenizer=tokenizer,
        prefix=prefix,
        continuation=x,
        device=device,
    )


def split_task_and_rollout(full_trace: str):
    """
    Splits the full ScienceWorld trace into:
      - task description z_obs
      - remainder x_obs
    """
    lines = full_trace.splitlines()
    if not lines or not lines[0].startswith("Task Description:"):
        raise ValueError("Trace must start with 'Task Description:'")

    task_line = lines[0]
    z_obs = task_line.replace("Task Description:", "", 1).strip()
    x_obs = "\n".join(lines[1:]).lstrip()
    return z_obs, x_obs


def print_scores(name: str, prior_loss: float, likelihood_loss: float):
    total_loss = prior_loss + likelihood_loss
    print(f"\n=== {name} ===")
    print(f"Prior loss (-log p(z)):        {prior_loss:.4f}")
    print(f"Likelihood loss (-log p(x|z)): {likelihood_loss:.4f}")
    print(f"Total loss:                    {total_loss:.4f}")


def main():
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    # Put your own trace here.
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

    print("Observed task description:")
    print(observed_task)

    print("\nObserved rollout prefix:")
    print(observed_rollout[:1000] + ("..." if len(observed_rollout) > 1000 else ""))

    concise_prior = score_prior(model, tokenizer, candidate_concise, device)
    concise_likelihood = score_likelihood(model, tokenizer, candidate_concise, observed_rollout, device)

    dummy_prior = score_prior(model, tokenizer, candidate_dummy, device)
    dummy_likelihood = score_likelihood(model, tokenizer, candidate_dummy, observed_rollout, device)

    false_prior = score_prior(model, tokenizer, candidate_false, device)
    false_likelihood = score_likelihood(model, tokenizer, candidate_false, observed_rollout, device)

    print("\nConcise task description:")
    print(candidate_concise)
    print_scores("CONCISE", concise_prior, concise_likelihood)

    print("\nDummy / overfit task description:")
    print(candidate_dummy)
    print_scores("DUMMY", dummy_prior, dummy_likelihood)

    print("\nFalse task description:")
    print(candidate_false)
    print_scores("FALSE", false_prior, false_likelihood)


if __name__ == "__main__":
    main()