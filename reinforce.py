import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model


# =========================
# Config
# =========================

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MAX_NEW_TOKENS = 96
TEMPERATURE = 0.8
LR = 5e-5
LAMBDA_PRIOR = 1.0
SEED = 42

NUM_EPISODES = 50
NUM_SAMPLES_PER_EPISODE = 4
GRAD_CLIP_NORM = 1.0
BASELINE_MOMENTUM = 0.9
MOVING_AVG_WINDOW = 10

# LoRA config
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

OBSERVED_SEQUENCE = [1, 1, 2, 5, 12, 27, 58, 121, 248, 503]

PROMPT = """# Observed sequence:
# 1, 1, 2, 5, 12, 27, 58, 121, 248, 503
#
# Complete the Python generator function.

def gen_sequence():
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


def moving_average(values, window: int):
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(sum(values[start:i + 1]) / (i - start + 1))
    return out


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
        "total_objective": -float(reward.item()),  # J
    }


def sample_from_policy(model, tokenizer, prompt: str, device, max_new_tokens: int, temperature: float):
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
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


def policy_logprob_of_sample(policy_model, prompt_ids: torch.Tensor, sampled_ids: torch.Tensor) -> torch.Tensor:
    """
    Returns log pi_theta(sampled_ids | prompt).
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
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
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
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
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

    baseline = None
    episode_mean_J = []
    episode_mean_reward = []

    for episode in range(NUM_EPISODES):
        sample_losses = []
        sample_rewards = []
        sample_Js = []

        for _ in range(NUM_SAMPLES_PER_EPISODE):
            sample = sample_from_policy(
                policy_model,
                tokenizer,
                PROMPT,
                device,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
            )
            z = sample["text"]

            reward_info = compute_reward(
                eval_model=eval_model,
                tokenizer=tokenizer,
                z=z,
                observed_sequence=OBSERVED_SEQUENCE,
                lam=LAMBDA_PRIOR,
                device=device,
            )

            reward = reward_info["reward"]
            total_objective = reward_info["total_objective"]  # J

            if baseline is None:
                baseline = float(reward.item())
            else:
                baseline = (
                    BASELINE_MOMENTUM * baseline
                    + (1.0 - BASELINE_MOMENTUM) * float(reward.item())
                )

            logprob = policy_logprob_of_sample(
                policy_model,
                sample["prompt_ids"],
                sample["sampled_ids"],
            )

            advantage = reward - baseline
            policy_loss = -(advantage * logprob)

            sample_losses.append(policy_loss)
            sample_rewards.append(float(reward.item()))
            sample_Js.append(total_objective)

        mean_policy_loss = torch.stack(sample_losses).mean()

        optimizer.zero_grad()
        mean_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        mean_reward = sum(sample_rewards) / len(sample_rewards)
        mean_J = sum(sample_Js) / len(sample_Js)

        episode_mean_reward.append(mean_reward)
        episode_mean_J.append(mean_J)

        if (episode + 1) % 10 == 0 or episode == 0:
            print(
                f"Episode {episode + 1:03d} | "
                f"mean J={mean_J:.4f} | "
                f"mean reward={mean_reward:.4f} | "
                f"baseline={baseline:.4f}"
            )

    print("\n=== Final greedy output ===")
    final_greedy = greedy_from_policy(
        policy_model,
        tokenizer,
        PROMPT,
        device,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print(final_greedy)

    J_mavg = moving_average(episode_mean_J, MOVING_AVG_WINDOW)

    plt.figure(figsize=(9, 5))
    plt.plot(range(1, NUM_EPISODES + 1), episode_mean_J, label="Mean episode J")
    plt.plot(range(1, NUM_EPISODES + 1), J_mavg, label=f"Moving average (window={MOVING_AVG_WINDOW})")
    plt.xlabel("Episode")
    plt.ylabel("Total objective J")
    plt.title("REINFORCE training: total objective over episodes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()