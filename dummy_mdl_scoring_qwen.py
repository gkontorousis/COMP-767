import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================
# Your generator
# =========================

def hard_v1():
    i = 1
    while True:
        if i % 3 == 0:
            yield (2 * i - 1) ** 2 - 1
        elif i % 3 == 1:
            yield 2 * i ** 2 - 1 
        else:
            yield (2 * i + 1) ** 2 - 1
        i += 1


# =========================
# Dataset
# =========================

def take_n(gen, n):
    return [next(gen) for _ in range(n)]


def build_dataset(seq, window=4):
    examples = []
    for i in range(len(seq) - window):
        context = seq[i:i+window]
        target = seq[i+window]
        examples.append((context, target))
    return examples


# =========================
# Logprob scoring
# =========================

def continuation_nll(model, tokenizer, prefix, continuation, device):
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(prefix + continuation, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits[:, :-1, :]
        labels = full_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    start = prefix_ids.shape[1] - 1
    return -token_log_probs[:, start:].sum().item()


def score_z(model, tokenizer, z, device):
    prior_prompt = "Write a short Python function that generates a sequence.\n\n"
    return continuation_nll(model, tokenizer, prior_prompt, z, device)


def score_dataset(model, tokenizer, z, dataset, device):
    total = 0.0

    for context, target in dataset:
        prefix = (
            "Given this Python code:\n\n"
            f"{z}\n\n"
            f"Sequence: {', '.join(map(str, context))}\n"
            "Next number:"
        )

        continuation = f" {target}"
        total += continuation_nll(model, tokenizer, prefix, continuation, device)

    return total


# =========================
# Explanations
# =========================

def build_dummy_explanation(train_seq):
    return (
        "vals = [" + ", ".join(map(str, train_seq)) + "]\n"
        "def next_value(n):\n"
        "    return vals[n]\n"
    )


def build_concise_explanation():
    return (
        "def next_value(i):\n"
        "    if i % 3 == 0:\n"
        "        return (2 * i - 1) ** 2 - 1\n"
        "    elif i % 3 == 1:\n"
        "        return 2 * i ** 2 - 1\n"
        "    else:\n"
        "        return (2 * i + 1) ** 2 - 1\n"
    )


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

    # Generate sequence
    gen = hard_v1()
    seq = take_n(gen, 12)

    # Split
    train_seq = seq[:8]
    test_seq = seq[8:]

    train_data = build_dataset(train_seq)
    test_data = build_dataset(seq)

    print("Sequence:", seq)

    # Build explanations
    z_dummy = build_dummy_explanation(train_seq)
    z_concise = build_concise_explanation()

    # Score dummy
    dummy_z = score_z(model, tokenizer, z_dummy, device)
    dummy_train = score_dataset(model, tokenizer, z_dummy, train_data, device)
    dummy_test = score_dataset(model, tokenizer, z_dummy, test_data, device)

    # Score concise
    concise_z = score_z(model, tokenizer, z_concise, device)
    concise_train = score_dataset(model, tokenizer, z_concise, train_data, device)
    concise_test = score_dataset(model, tokenizer, z_concise, test_data, device)

    # Print
    def print_scores(name, z, train, test, n_train, n_test):
        map_loss = z + train
        avg_train = train / max(n_train, 1)
        avg_test = test / max(n_test, 1)

        print(f"\n=== {name} ===")
        print(f"-log p(z):       {z:.2f}")
        print(f"train NLL:       {train:.2f}")
        print(f"test NLL:        {test:.2f}")
        print(f"MDL score:       {map_loss:.2f}")
        print(f"MAP loss:        {map_loss:.2f}")
        print(f"avg train NLL:   {avg_train:.2f}")
        print(f"avg test NLL:    {avg_test:.2f}")

    print_scores("DUMMY", dummy_z, dummy_train, dummy_test, len(train_data), len(test_data))
    print_scores("CONCISE", concise_z, concise_train, concise_test, len(train_data), len(test_data))


if __name__ == "__main__":
    main()