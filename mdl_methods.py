from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_MODEL_CACHE_DIR = None


def hf_pretrained_kwargs():
    kw = {"trust_remote_code": True}
    if HF_MODEL_CACHE_DIR:
        p = Path(str(HF_MODEL_CACHE_DIR)).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        kw["cache_dir"] = str(p)
    return kw


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


DEVICE = pick_device()
DTYPE = pick_dtype(DEVICE)

SYSTEM_PROMPT = (
    "You are a careful pattern-discovery assistant. "
    "Given an observed number sequence, propose a concise natural-language rule "
    "that likely generated it. Return only the rule, with no extra commentary."
)


def load_model_and_tokenizer(model_name: str):
    kw = hf_pretrained_kwargs()
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kw)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        **kw,
    ).to(DEVICE)
    model.eval()
    return model, tokenizer


def render_chat(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def continuation_nll(model, tokenizer, prefix: str, continuation: str) -> float:
    """
    Returns -log p(continuation | prefix)
    """
    with torch.no_grad():
        prefix_ids = tokenizer(
            prefix,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(DEVICE)

        full_ids = tokenizer(
            prefix + continuation,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(DEVICE)

        outputs = model(full_ids)
        logits = outputs.logits[:, :-1, :].float()
        labels = full_ids[:, 1:]

        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

        start = prefix_ids.shape[1] - 1
        logp = token_log_probs[:, start:].sum().item()

    return -logp


def build_conditional_prefix(tokenizer, observed_sequence):
    seq_text = ", ".join(map(str, observed_sequence))
    user_prompt = (
        f"Observed sequence:\n{seq_text}\n\n"
        "What is the most likely concise rule that generated this sequence?"
    )
    return render_chat(tokenizer, SYSTEM_PROMPT, user_prompt)


def build_prior_prefix(tokenizer):
    system_prompt = (
        "You are a careful pattern-discovery assistant. "
        "Write a concise rule for a number sequence. "
        "Return only the rule."
    )
    user_prompt = "Write a rule to generate a number sequence."
    return render_chat(tokenizer, system_prompt, user_prompt)


def build_likelihood_prefix(tokenizer, explanation: str):
    system_prompt = (
        "You are a careful sequence predictor. "
        "Given a rule, continue the number sequence implied by that rule."
    )
    user_prompt = (
        f"Rule:\n{explanation}\n\n"
        "Output the first 10 terms of the sequence as comma-separated integers only."
    )
    return render_chat(tokenizer, system_prompt, user_prompt)


def score_explanation(model, tokenizer, observed_sequence, explanation: str, loss_mode: str, lam=1):
    if loss_mode == "conditional":
        prefix = build_conditional_prefix(tokenizer, observed_sequence)
        loss = continuation_nll(model, tokenizer, prefix, explanation)
        return {
            "mode": "conditional",
            "loss": loss,
            "prior_nll": None,
            "likelihood_nll": None,
        }

    elif loss_mode == "bayes":
        seq_text = ", ".join(map(str, observed_sequence))

        prior_prefix = build_prior_prefix(tokenizer)
        prior_nll = continuation_nll(model, tokenizer, prior_prefix, explanation)

        likelihood_prefix = build_likelihood_prefix(tokenizer, explanation)
        likelihood_nll = continuation_nll(model, tokenizer, likelihood_prefix, seq_text)

        loss = lam * prior_nll + likelihood_nll
        return {
            "mode": "bayes",
            "loss": loss,
            "prior_nll": prior_nll,
            "likelihood_nll": likelihood_nll,
        }

    else:
        raise ValueError("loss_mode must be 'conditional' or 'bayes'")


def build_policy_model(model_name: str):
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        **hf_pretrained_kwargs(),
    ).to(DEVICE)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
    )

    model = get_peft_model(base, lora_config)
    model.train()
    return model


def encode_continuation(tokenizer, text: str):
    return tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(DEVICE)


def sample_explanation(model, tokenizer, observed_sequence, max_new_tokens=64, temperature=1.0):
    prefix = build_conditional_prefix(tokenizer, observed_sequence)

    inputs = tokenizer(
        prefix,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = output[:, prompt_len:]
    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

    return text, inputs["input_ids"], gen_ids


def generate_explanation(model, tokenizer, observed_sequence, max_new_tokens=64):
    prefix = build_conditional_prefix(tokenizer, observed_sequence)

    inputs = tokenizer(
        prefix,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = output[:, prompt_len:]
    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
    return text


def policy_logprob(model, prompt_ids, gen_ids):
    full_ids = torch.cat([prompt_ids, gen_ids], dim=1)

    outputs = model(full_ids)
    logits = outputs.logits[:, :-1, :].float()
    labels = full_ids[:, 1:]

    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    start = prompt_ids.shape[1] - 1
    return token_log_probs[:, start:].sum()
