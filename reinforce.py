import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model


# =========================
# Config
# =========================

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MAX_NEW_TOKENS = 96
TEMPERATURE = 0.8
LR = 5e-4
LAMBDA_PRIOR = 1.0
SEED = 42

# LoRA config
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# One toy prompt / episode
OBSERVED_SEQUENCE = [1, 1, 2, 5, 12, 27, 58, 121, 248, 503]

PROMPT = """You are given a number sequence.

Observed values:
1, 1, 2, 5, 12, 27, 58, 121, 248, 503

Write a concise Python generator function that produces this sequence.
Return only Python code.
"""


# =========================
# Utilities
# =========================

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_prior_prefix() -> str:
    return (
        "Write a concise Python generator function for a numerical sequence.\n\n"
        "Code:\n"
    )


def build_likelihood_prefix(z: str) -> str:
    return (
        "Given this Python generator code:\n\n"
        f"{z}\n\n"
        "Output the first 10 values, comma-separated:\n"
    )


def continuation_logprob(model, tokenizer, prefix: str, continuation: str, device) -> torch.Tensor:
    """
    Returns log p(continuation | prefix) as a scalar tensor.
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

    outputs = model(full_ids)
    logits = outputs.logits[:, :-1, :].float()
    labels = full_ids[:, 1:]

    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    start = prefix_ids.shape[1] - 1
    return token_log_probs[:, start:].sum()


def compute_reward(eval_model, tokenizer, z: str, observed_sequence, lam: float, device):
    """
    Reward = -J = lam * log p(z) + log p(x | z)
    where:
      J = -lam log p(z) - log p(x|z)
    """
    seq_text = ", ".join(map(str, observed_sequence))

    with torch.no_grad():
        log_p_z = continuation_logprob(
            eval_model,
            tokenizer,
            build_prior_prefix(),
            z,
            device,
        )

        log_p_x_given_z = continuation_logprob(
            eval_model,
            tokenizer,
            build_likelihood_prefix(z),
            seq_text,
            device,
        )

        reward = lam * log_p_z + log_p_x_given_z

    return {
        "reward": reward.detach(),
        "prior_nll": -float(log_p_z.item()),
        "likelihood_nll": -float(log_p_x_given_z.item()),
        "total_objective": -float(reward.item()),
    }


def sample_from_policy(model, tokenizer, prompt: str, device, max_new_tokens: int, temperature: float):
    """
    Sample one completion z from the policy model.
    """
    inputs = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    sampled_ids = generated[:, prompt_len:]
    sampled_text = tokenizer.decode(sampled_ids[0], skip_special_tokens=True)

    return {
        "prompt_ids": inputs["input_ids"],
        "sampled_ids": sampled_ids,
        "text": sampled_text,
    }


def greedy_from_policy(model, tokenizer, prompt: str, device, max_new_tokens: int):
    """
    Greedy decode from the policy model.
    """
    inputs = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = generated[:, prompt_len:]
    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return text


def policy_logprob_of_sample(policy_model, prompt_ids: torch.Tensor, sampled_ids: torch.Tensor) -> torch.Tensor:
    """
    Returns log pi_theta(sampled_ids | prompt) as a scalar tensor.
    This is the differentiable quantity used in REINFORCE.
    """
    full_ids = torch.cat([prompt_ids, sampled_ids], dim=1)

    outputs = policy_model(full_ids)
    logits = outputs.logits[:, :-1, :].float()
    labels = full_ids[:, 1:]

    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    start = prompt_ids.shape[1] - 1
    return token_log_probs[:, start:].sum()


def build_policy_model(model_name: str, dtype, device):
    """
    Load a base model and wrap it with LoRA adapters.
    Only LoRA weights are trainable.
    """
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    ).to(device)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(base, lora_config)
    model.train()
    return model


def build_frozen_eval_model(model_name: str, dtype, device):
    """
    Frozen base evaluator model.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    ).to(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model


def print_trainable_parameters(model):
    trainable = 0
    total = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    pct = 100.0 * trainable / max(total, 1)
    print(f"Trainable params: {trainable:,}")
    print(f"Total params:     {total:,}")
    print(f"Trainable %:      {pct:.4f}%")


# =========================
# Main
# =========================

def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Using device: {device}")
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print(f"\nLoading trainable LoRA policy model: {MODEL_NAME}")
    policy_model = build_policy_model(MODEL_NAME, dtype, device)
    print_trainable_parameters(policy_model)

    print(f"\nLoading frozen evaluator model: {MODEL_NAME}")
    eval_model = build_frozen_eval_model(MODEL_NAME, dtype, device)

    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=LR,
    )

    print("\n=== Prompt ===")
    print(PROMPT)

    print("\n=== Pre-update greedy output ===")
    pre_update_greedy = greedy_from_policy(
        policy_model,
        tokenizer,
        PROMPT,
        device,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print(pre_update_greedy)

    print("\n=== Episode 1: sampled output ===")
    sample = sample_from_policy(
        policy_model,
        tokenizer,
        PROMPT,
        device,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
    z = sample["text"]
    print(z)

    reward_info = compute_reward(
        eval_model=eval_model,
        tokenizer=tokenizer,
        z=z,
        observed_sequence=OBSERVED_SEQUENCE,
        lam=LAMBDA_PRIOR,
        device=device,
    )

    reward = reward_info["reward"]

    logprob = policy_logprob_of_sample(
        policy_model,
        sample["prompt_ids"],
        sample["sampled_ids"],
    )

    baseline = 0.0
    advantage = reward - baseline
    policy_loss = -(advantage * logprob)

    print("\n=== Evaluator scores for sampled output ===")
    print(f"Prior loss (-log p(z)):        {reward_info['prior_nll']:.4f}")
    print(f"Likelihood loss (-log p(x|z)): {reward_info['likelihood_nll']:.4f}")
    print(f"Total objective J:             {reward_info['total_objective']:.4f}")
    print(f"Reward = -J:                   {float(reward.item()):.4f}")

    print("\n=== REINFORCE terms ===")
    print(f"Policy logprob:                {float(logprob.item()):.4f}")
    print(f"Advantage:                     {float(advantage.item()):.4f}")
    print(f"Policy loss:                   {float(policy_loss.item()):.4f}")

    optimizer.zero_grad()
    policy_loss.backward()

    grad_norm_sq = 0.0
    for p in policy_model.parameters():
        if p.grad is not None:
            grad_norm_sq += p.grad.detach().float().pow(2).sum().item()
    grad_norm = grad_norm_sq ** 0.5
    print(f"Gradient norm:                 {grad_norm:.4f}")

    optimizer.step()

    print("\n=== Post-update greedy output ===")
    post_update_greedy = greedy_from_policy(
        policy_model,
        tokenizer,
        PROMPT,
        device,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print(post_update_greedy)

    print("\n=== Post-update sampled output ===")
    post_sample = sample_from_policy(
        policy_model,
        tokenizer,
        PROMPT,
        device,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
    print(post_sample["text"])


if __name__ == "__main__":
    main()