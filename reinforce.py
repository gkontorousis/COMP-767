import ast
import math
import signal
import textwrap
from contextlib import contextmanager

import matplotlib.pyplot as plt
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================
# Config
# =========================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_NEW_TOKENS = 128
TEMPERATURE = 0.8
LR = 5e-5
SEED = 42

EXEC_ALPHA = 0.7          # weight on correlation loss
USE_AFFINE_MSE = True     # False => use plain MSE instead
EXEC_SCALE = 10.0         # optional overall scaling of task loss

NUM_EPISODES = 50
NUM_SAMPLES_PER_EPISODE = 4
GRAD_CLIP_NORM = 1.0
BASELINE_MOMENTUM = 0.9
MOVING_AVG_WINDOW = 10

LAMBDA_PRIOR = 0.1  # keep prior weak so it doesn't dominate execution fit

# LoRA config
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

OBSERVED_SEQUENCE = [1, 1, 2, 5, 12, 27, 58, 121, 248, 503]

SYSTEM_PROMPT = (
    "You are a careful Python assistant. "
    "Return only valid Python code, with no markdown fences, no explanation, and no extra text."
)

USER_PROMPT = """Write a concise Python generator function that produces this number sequence:

1, 1, 2, 5, 12, 27, 58, 121, 248, 503

Requirements:
- Define exactly one function named gen_sequence
- It must have the form def gen_sequence():
- It must contain a while True loop
- It must use yield inside that loop
- Return only Python code
"""


# =========================
# Utility helpers
# =========================

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def moving_average(values, window: int):
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(sum(values[start:i + 1]) / (i - start + 1))
    return out


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
# Chat formatting
# =========================

def render_chat_prefix(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_prior_prefix(tokenizer) -> str:
    system_prompt = (
        "You write concise, valid Python generator functions. "
        "Return only Python code with no markdown fences or explanation."
    )
    user_prompt = (
        "Write a concise Python generator function named gen_sequence that contains "
        "a while True loop and uses yield."
    )
    return render_chat_prefix(tokenizer, system_prompt, user_prompt)


def build_policy_prompt(tokenizer) -> str:
    return render_chat_prefix(tokenizer, SYSTEM_PROMPT, USER_PROMPT)


# =========================
# Logprob scoring
# =========================

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


def prior_nll(eval_model, tokenizer, z: str, device) -> float:
    prefix = build_prior_prefix(tokenizer)
    with torch.no_grad():
        log_p_z = continuation_logprob(eval_model, tokenizer, prefix, z, device)
    return -float(log_p_z.item())


# =========================
# Safe-ish execution helpers
# =========================

class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds: int):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out during execution")

    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def extract_code(text: str) -> str:
    """
    Best-effort extraction of Python code from an assistant reply.
    """
    text = strip_code_fences(text)

    # If the model still emits some prose, keep only from first def gen_sequence onward.
    idx = text.find("def gen_sequence")
    if idx != -1:
        text = text[idx:]

    return text.strip()


def validate_structure(code: str):
    """
    Parse code and verify:
      - defines gen_sequence
      - has while True
      - has yield inside the function
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {
            "ok": False,
            "reason": f"syntax_error: {e}",
            "tree": None,
        }

    gen_func = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "gen_sequence":
            gen_func = node
            break

    if gen_func is None:
        return {
            "ok": False,
            "reason": "missing_gen_sequence",
            "tree": tree,
        }

    has_while_true = False
    has_yield = False

    for node in ast.walk(gen_func):
        if isinstance(node, ast.While):
            if isinstance(node.test, ast.Constant) and node.test.value is True:
                has_while_true = True
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            has_yield = True

    if not has_while_true:
        return {
            "ok": False,
            "reason": "missing_while_true",
            "tree": tree,
        }

    if not has_yield:
        return {
            "ok": False,
            "reason": "missing_yield",
            "tree": tree,
        }

    return {
        "ok": True,
        "reason": "ok",
        "tree": tree,
    }


def safe_execute_and_generate(code: str, n_terms: int):
    """
    Execute the code in a very restricted environment and return first n terms.
    """
    safe_builtins = {
        "range": range,
        "len": len,
        "enumerate": enumerate,
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
    }

    globals_dict = {"__builtins__": safe_builtins}
    locals_dict = {}

    with time_limit(2):
        exec(code, globals_dict, locals_dict)

        fn = locals_dict.get("gen_sequence", globals_dict.get("gen_sequence", None))
        if fn is None or not callable(fn):
            raise RuntimeError("gen_sequence not found after execution")

        gen = fn()
        out = []
        for _ in range(n_terms):
            value = next(gen)
            if not isinstance(value, (int, float)):
                raise RuntimeError(f"non-numeric yield: {value!r}")
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                raise RuntimeError(f"bad float yield: {value!r}")
            out.append(value)

    return out


def pearson_corr_loss(predicted, observed, eps=1e-8):
    x = torch.tensor(predicted, dtype=torch.float32)
    y = torch.tensor(observed, dtype=torch.float32)

    x = x - x.mean()
    y = y - y.mean()

    x_norm = torch.sqrt((x * x).sum() + eps)
    y_norm = torch.sqrt((y * y).sum() + eps)

    corr = (x * y).sum() / (x_norm * y_norm)
    corr = torch.clamp(corr, -1.0, 1.0)

    # 0 is best, 2 is worst
    return 1.0 - corr.item()


def mse_loss(predicted, observed):
    x = torch.tensor(predicted, dtype=torch.float32)
    y = torch.tensor(observed, dtype=torch.float32)
    return torch.mean((x - y) ** 2).item()


def affine_shifted_mse_loss(predicted, observed, eps=1e-8):
    """
    Fit observed ≈ a * predicted + b, then compute MSE after that affine fit.
    """
    x = torch.tensor(predicted, dtype=torch.float32)
    y = torch.tensor(observed, dtype=torch.float32)

    x_mean = x.mean()
    y_mean = y.mean()

    x_centered = x - x_mean
    y_centered = y - y_mean

    var_x = torch.mean(x_centered ** 2)

    if var_x.item() < eps:
        # Degenerate case: predicted is nearly constant
        y_hat = torch.full_like(y, y_mean)
    else:
        a = torch.mean(x_centered * y_centered) / var_x
        b = y_mean - a * x_mean
        y_hat = a * x + b

    return torch.mean((y_hat - y) ** 2).item()

def execution_error(
    code: str,
    observed_sequence,
    alpha: float = EXEC_ALPHA,
    use_affine_mse: bool = USE_AFFINE_MSE,
    scale: float = EXEC_SCALE,
):
    """
    Lower is better.

    Mixed loss:
        exec_error = scale * [ alpha * corr_loss + (1 - alpha) * mse_term ]

    where:
        corr_loss = 1 - PearsonCorr(predicted, observed)
        mse_term  = log1p(MSE) or log1p(affine_shifted_MSE)
    """
    code = extract_code(code)

    if not code:
        return {
            "exec_error": 200.0,
            "status": "empty_output",
            "predicted": None,
            "clean_code": code,
        }

    validation = validate_structure(code)
    if not validation["ok"]:
        return {
            "exec_error": 150.0,
            "status": validation["reason"],
            "predicted": None,
            "clean_code": code,
        }

    try:
        predicted = safe_execute_and_generate(code, len(observed_sequence))
    except TimeoutException:
        return {
            "exec_error": 200.0,
            "status": "timeout",
            "predicted": None,
            "clean_code": code,
        }
    except Exception as e:
        return {
            "exec_error": 180.0,
            "status": f"runtime_error: {type(e).__name__}",
            "predicted": None,
            "clean_code": code,
        }

    corr = pearson_corr_loss(predicted, observed_sequence)

    if use_affine_mse:
        mse = affine_shifted_mse_loss(predicted, observed_sequence)
    else:
        mse = mse_loss(predicted, observed_sequence)

    # Important: raw MSE can be huge, so compress it
    mse_term = math.log1p(mse)

    mixed_loss = alpha * corr + (1.0 - alpha) * mse_term
    exec_error_value = scale * mixed_loss

    return {
        "exec_error": float(exec_error_value),
        "status": "ok",
        "predicted": predicted,
        "clean_code": code,
    }


def compute_reward(eval_model, tokenizer, z: str, observed_sequence, lam: float, device):
    """
    J(z, x) = lam * prior_nll(z) + exec_error(z, x)
    reward = -J
    """
    clean_code = extract_code(z)
    p_nll = prior_nll(eval_model, tokenizer, clean_code, device)
    exec_info = execution_error(clean_code, observed_sequence)

    total_objective = lam * p_nll + exec_info["exec_error"]
    reward = -total_objective

    return {
        "reward": torch.tensor(reward, device=device, dtype=torch.float32),
        "prior_nll": p_nll,
        "exec_error": exec_info["exec_error"],
        "total_objective": total_objective,
        "status": exec_info["status"],
        "predicted": exec_info["predicted"],
        "clean_code": exec_info["clean_code"],
    }


# =========================
# Generation / policy logprob
# =========================

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
    full_ids = torch.cat([prompt_ids, sampled_ids], dim=1)

    outputs = policy_model(full_ids)
    logits = outputs.logits[:, :-1, :].float()
    labels = full_ids[:, 1:]

    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    start = prompt_ids.shape[1] - 1
    return token_log_probs[:, start:].sum()


# =========================
# Model builders
# =========================

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

    prompt = build_policy_prompt(tokenizer)

    print("\n=== Pre-update greedy output ===")
    pre_update_greedy = greedy_from_policy(
        policy_model,
        tokenizer,
        prompt,
        device,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print(pre_update_greedy)

    baseline = None
    episode_mean_J = []

    for episode in range(NUM_EPISODES):
        sample_losses = []
        sample_Js = []

        for _ in range(NUM_SAMPLES_PER_EPISODE):
            sample = sample_from_policy(
                policy_model,
                tokenizer,
                prompt,
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
            J = reward_info["total_objective"]

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
            sample_Js.append(J)

        mean_policy_loss = torch.stack(sample_losses).mean()

        optimizer.zero_grad()
        mean_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        mean_J = sum(sample_Js) / len(sample_Js)
        episode_mean_J.append(mean_J)

        if (episode + 1) % 10 == 0 or episode == 0:
            print(f"Episode {episode + 1:03d} | mean J={mean_J:.4f} | baseline={baseline:.4f}")

    print("\n=== Final greedy output ===")
    final_greedy = greedy_from_policy(
        policy_model,
        tokenizer,
        prompt,
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
    plt.title("REINFORCE with execution-based fit")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


main()