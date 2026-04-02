import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# =========================
# Your generator
# =========================

def gen_fct():
    i=0
    while True:
        yield 2**i - i
        i+=1

text = """
def gen_sequence():
    i=0
    while True:
        yield 2**i - i
        i+=1
"""

# =========================
# Sequence helper
# =========================

def take_n(gen, n):
    return [next(gen) for _ in range(n)]


# =========================
# Logprob scoring
# =========================

def continuation_nll(model, tokenizer, prefix, continuation, device):
    """
    Returns -log p(continuation | prefix)
    """
    prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(prefix + continuation, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits[:, :-1, :]
        labels = full_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    start = prefix_ids.shape[1] - 1
    continuation_log_prob_sum = token_log_probs[:, start:].sum().item()
    return -float(continuation_log_prob_sum)


def score_prior(model, tokenizer, z, device):
    """
    Prior loss: -log p(z)
    """
    prior_prompt = "Write a short Python function that generates a sequence.\n\n"
    return continuation_nll(model, tokenizer, prior_prompt, z, device)


def score_likelihood(model, tokenizer, z, sequence, device):
    """
    Likelihood loss: -log p(x | z)
    where x is the whole observed sequence.
    """
    prefix = (
        "Given this Python code:\n\n"
        f"{z}\n\n"
        "Output the first 10 values, comma-separated:\n"
    )
    continuation = ", ".join(map(str, sequence))
    return continuation_nll(model, tokenizer, prefix, continuation, device)


# =========================
# Explanations
# =========================

def build_dummy_explanation(sequence):
    return (
        "vals = [" + ", ".join(map(str, sequence)) + "]\n"
        "def gen_sequence():\n"
        "   i=0\n"
        "   while True:\n"
        "       yield vals[i]\n"
        "       i += 1\n"
    )


def build_concise_explanation():
    return text.strip()


def build_false_explanation():
    return (
        "def fib():\n"
        "    a, b = 0, 1\n"
        "    while True:\n"
        "        yield a\n"
        "        a, b = b, a + b\n"
    )

# =========================
# Printing
# =========================

def print_scores(name, prior_loss, likelihood_loss):
    total_loss = prior_loss + likelihood_loss

    print(f"\n=== {name} ===")
    print(f"Prior loss (-log p(z)):        {prior_loss:.4f}")
    print(f"Likelihood loss (-log p(x|z)): {likelihood_loss:.4f}")
    print(f"Total loss:                    {total_loss:.4f}")


# =========================
# Main
# =========================

def main():
    model_name = "Qwen/Qwen2.5-7B-Instruct"

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

    # Generate one sequence
    gen = gen_fct()
    seq = take_n(gen, 10)

    print("Observed sequence:")
    print(seq)

    # Build explanations
    z_dummy = build_dummy_explanation(seq)
    z_concise = build_concise_explanation()
    z_false = build_false_explanation()

    # Score dummy explanation
    dummy_prior = score_prior(model, tokenizer, z_dummy, device)
    dummy_likelihood = score_likelihood(model, tokenizer, z_dummy, seq, device)

    # Score concise explanation
    concise_prior = score_prior(model, tokenizer, z_concise, device)
    concise_likelihood = score_likelihood(model, tokenizer, z_concise, seq, device)


    # score for false explanation (e.g., "def fib(): ...")
    false_prior = score_prior(model, tokenizer, z_false, device)
    false_likelihood = score_likelihood(model, tokenizer, z_false, seq, device)

    # Print results
    print("\nDummy explanation:")
    print(z_dummy)
    print_scores("DUMMY", dummy_prior, dummy_likelihood)

    print("\nConcise explanation:")
    print(z_concise)
    print_scores("CONCISE", concise_prior, concise_likelihood)

    print("\nFalse explanation:")
    print(z_false)
    print_scores("FALSE", false_prior, false_likelihood)

    
if __name__ == "__main__":
    main()