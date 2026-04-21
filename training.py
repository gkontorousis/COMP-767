#!/usr/bin/env python
# coding: utf-8
# USE OF JUPYTER LAB TO 'SAVE AND EXPORT AS EXECUTABLE SCRIPT' FROM training.ipynb notebook file

# ## Environment Setup
# Load dependencies and initialize shared components.

# In[1]:


from pathlib import Path
import os
import torch
from dotenv import load_dotenv

import mdl_methods

load_dotenv(".env.local")

HF_MODEL_CACHE_DIR = os.getenv("HF_MODEL_CACHE_DIR", None)

mdl_methods.HF_MODEL_CACHE_DIR = HF_MODEL_CACHE_DIR

DEVICE = mdl_methods.DEVICE
DTYPE = mdl_methods.DTYPE
print(f"DEVICE={DEVICE}  DTYPE={DTYPE}  HF_MODEL_CACHE_DIR={HF_MODEL_CACHE_DIR!r}")

# =========================
# Config
# =========================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# "conditional"  => score = -log p(z | x)
# "bayes"        => score = -log p(z) - log p(x | z)
LOSS_MODE = "conditional"

OBSERVED_SEQUENCE = [1, 1, 2, 5, 12, 27, 58, 121, 248, 503]

SYSTEM_PROMPT = (
    "You are a careful pattern-discovery assistant. "
    "Given an observed number sequence, propose a concise natural-language rule "
    "that likely generated it. Return only the rule, with no extra commentary."
)

mdl_methods.SYSTEM_PROMPT = SYSTEM_PROMPT

load_model_and_tokenizer = mdl_methods.load_model_and_tokenizer
render_chat = mdl_methods.render_chat
continuation_nll = mdl_methods.continuation_nll
build_conditional_prefix = mdl_methods.build_conditional_prefix
build_prior_prefix = mdl_methods.build_prior_prefix
build_likelihood_prefix = mdl_methods.build_likelihood_prefix
generate_explanation = mdl_methods.generate_explanation
score_explanation = mdl_methods.score_explanation


# ## Single-Sequence Generation
# Generate an explanation for one observed sequence.

# In[2]:


model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

explanation = generate_explanation(model, tokenizer, OBSERVED_SEQUENCE)
score_conditional = score_explanation(model, tokenizer, OBSERVED_SEQUENCE, explanation, "conditional")
score_bayesian = score_explanation(model, tokenizer, OBSERVED_SEQUENCE, explanation, "bayes")

print("=== Observed sequence ===")
print(OBSERVED_SEQUENCE)

print("\n=== Generated explanation ===")
print(explanation)

print("\n=== Score ===")
print(f"mode: conditional")
print(f"loss: {score_conditional['loss']:.4f}")
print(f"mode: bayes")
print(f"loss: {score_bayesian['loss']:.4f}")
print(f"prior_nll: {score_bayesian['prior_nll']:.4f}")
print(f"likelihood_nll: {score_bayesian['likelihood_nll']:.4f}")



# In[3]:


OBSERVED_SEQUENCE = [1, 1, 2, 5, 12, 27, 58, 121, 248, 503]
concise_explanation = "2^n - n"
concise_explanation_ling = "Each term is 2 to the power of its index, minus its index."
dummy_explanation = f"the pattern is {OBSERVED_SEQUENCE}"
false_explanation = "3*n + 1"
max_likelihood_loss = explanation


# ## Scoring and Baselines
# Compare concise vs. generated explanations and inspect scoring behavior.

# In[ ]:


explanations_to_score = {
    "Concise explanation": concise_explanation,
    "Linguistic explanation": concise_explanation_ling,
    "Dummy explanation": dummy_explanation,
    "False explanation": false_explanation,
    "Max Likelihood": max_likelihood_loss,
}

for name, explanation in explanations_to_score.items():
    score_conditional = score_explanation(model, tokenizer, OBSERVED_SEQUENCE, explanation, "conditional")
    score_bayesian = score_explanation(model, tokenizer, OBSERVED_SEQUENCE, explanation, "bayes", 0.25)

    print(f"\n=== {name} ===")
    print(f"Explanation: {explanation}")
    print("\n=== Score ===")
    print(f"mode: conditional")
    print(f"loss: {score_conditional['loss']:.4f}\n")
    print(f"mode: bayes")
    print(f"loss: {score_bayesian['loss']:.4f}")
    print(f"prior_nll: {score_bayesian['prior_nll']:.4f}")
    print(f"likelihood_nll: {score_bayesian['likelihood_nll']:.4f}")


# In[ ]:


import torch
import torch.nn as nn
import random
from tqdm.auto import tqdm

from mdl_methods import build_policy_model, encode_continuation, policy_logprob, sample_explanation

# =========================
# REINFORCE loop
# =========================

policy_model = build_policy_model(MODEL_NAME)
evaluator_model = model

optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-5)

baseline = None

NUM_EPISODES = 50
SAMPLES_PER_EPISODE = 4

# add before the loop
reward_history = []

_episode_pbar = tqdm(range(NUM_EPISODES), desc="REINFORCE", unit="ep")
for episode in _episode_pbar:

    rewards = []
    logprobs = []

    for i in tqdm(range(SAMPLES_PER_EPISODE), desc="  samples", leave=False):

        z, prompt_ids, gen_ids = sample_explanation(
            policy_model, tokenizer, OBSERVED_SEQUENCE, temperature=0.8
        )

        if i == 1: # injection lol
          z = concise_explanation
          gen_ids = encode_continuation(tokenizer, z)

        score = score_explanation(
            evaluator_model, tokenizer, OBSERVED_SEQUENCE, z, "bayes", 0.30
        )

        reward = -score["loss"]  # IMPORTANT

        logprob = policy_logprob(policy_model, prompt_ids, gen_ids)

        rewards.append(reward)
        logprobs.append(logprob)

    rewards = torch.tensor(rewards, device=DEVICE)
    logprobs = torch.stack(logprobs)

    # baseline (batch mean)
    baseline = rewards.mean()
    reward_history.append(baseline)

    advantages = rewards - baseline

    loss = -(advantages.detach() * logprobs).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _episode_pbar.set_postfix(
        mean_r=f"{rewards.mean().item():.3f}",
        loss=f"{float(loss.detach().item()):.4f}",
    )
    # for j, (r, lp) in enumerate(zip(rewards.tolist(), logprobs)):
    #     print(j, r)
    # print("baseline:", baseline.item())
    # print("advantages:", (rewards - baseline).tolist())


# In[ ]:


import matplotlib.pyplot as plt

plt.figure()
# Convert CUDA tensors to CPU and then to a list of numbers for plotting
plt.plot([r.cpu().item() for r in reward_history])
plt.xlabel("Episode")
plt.ylabel("Mean Reward")
plt.title("REINFORCE Reward Curve")
plt.grid(True)
plt.savefig("reward_curve.png")
plt.close()


# ## Policy Model Check
# Run policy-model generation and conditional scoring checks.

# In[ ]:


# base_model: original unfine-tuned model
# policy_model: fine-tuned LoRA model
# evaluator_model: frozen scorer
# tokenizer: shared tokenizer

base_model = model
base_model.eval()
policy_model.eval()


base_explanation = generate_explanation(base_model, tokenizer, OBSERVED_SEQUENCE)
ft_explanation = generate_explanation(policy_model, tokenizer, OBSERVED_SEQUENCE)

base_score = score_explanation(
    evaluator_model, tokenizer, OBSERVED_SEQUENCE, base_explanation, "bayes", 0.3
)
c_base_score = score_explanation(
    policy_model, tokenizer, OBSERVED_SEQUENCE, base_explanation, "conditional"
)

ft_score = score_explanation(
    evaluator_model, tokenizer, OBSERVED_SEQUENCE, ft_explanation, "bayes", 0.3
)
c_ft_score = score_explanation(
    policy_model, tokenizer, OBSERVED_SEQUENCE, ft_explanation, "conditional"
)

concise_score = score_explanation(
    evaluator_model, tokenizer, OBSERVED_SEQUENCE, concise_explanation, "bayes", 0.3
)
c_concise_score = score_explanation(
    policy_model, tokenizer, OBSERVED_SEQUENCE, concise_explanation, "conditional"
)

dummy_score = score_explanation(
    evaluator_model, tokenizer, OBSERVED_SEQUENCE, dummy_explanation, "bayes", 0.3
)
c_dummy_score = score_explanation(
    policy_model, tokenizer, OBSERVED_SEQUENCE, dummy_explanation, "conditional"
)

print("=== Observed sequence ===")
print(OBSERVED_SEQUENCE)

print("\n=== Base model explanation ===")
print(base_explanation)
print(f"Conditional loss: {c_base_score['loss']}\n")
print(f"Bayes loss: {base_score['loss']:.4f}")
print(f"prior_nll: {base_score['prior_nll']:.4f}")
print(f"likelihood_nll: {base_score['likelihood_nll']:.4f}")

print("\n=== Fine-tuned model explanation ===")
print(ft_explanation)
print(f"Conditional loss: {c_ft_score['loss']}\n")
print(f"Bayes loss: {ft_score['loss']:.4f}")
print(f"prior_nll: {ft_score['prior_nll']:.4f}")
print(f"likelihood_nll: {ft_score['likelihood_nll']:.4f}")


print("\n=== Correct explanation ===")
print(concise_explanation)
print(f"Conditional loss: {c_concise_score['loss']}\n")
print(f"Bayes loss: {concise_score['loss']:.4f}")
print(f"prior_nll: {concise_score['prior_nll']:.4f}")
print(f"likelihood_nll: {concise_score['likelihood_nll']:.4f}")

print("\n=== Dummy explanation ===")
print(dummy_explanation)
print(f"Conditional loss: {c_dummy_score['loss']}\n")
print(f"Bayes loss: {dummy_score['loss']:.4f}")
print(f"prior_nll: {dummy_score['prior_nll']:.4f}")
print(f"likelihood_nll: {dummy_score['likelihood_nll']:.4f}")



# In[ ]:


explanation = generate_explanation(policy_model, tokenizer, OBSERVED_SEQUENCE)
score_conditional = score_explanation(policy_model, tokenizer, OBSERVED_SEQUENCE, explanation, "conditional")
score_bayesian = score_explanation(model, tokenizer, OBSERVED_SEQUENCE, explanation, "bayes", 0.3)

print("=== Observed sequence ===")
print(OBSERVED_SEQUENCE)
print("[User]: What is the most likely concise rule that generated this data?")
print("\n=== Generated explanation ===")
print(explanation)

print("\n=== Score ===")
print(f"mode: conditional")
print(f"loss: {score_conditional['loss']:.4f}")
print(f"mode: bayes")
print(f"loss: {score_bayesian['loss']:.4f}")
print(f"prior_nll: {score_bayesian['prior_nll']:.4f}")
print(f"likelihood_nll: {score_bayesian['likelihood_nll']:.4f}")


# In[ ]:


import gc
import torch

try:
    del model
except:
    pass

try:
  del base_model
except:
  pass

try:
    del policy_model
except:
    pass

try:
    del evaluator_model
except:
    pass

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

if torch.backends.mps.is_available():
    torch.mps.empty_cache()


# In[ ]:


# ## Multi-Data Training
# Train/evaluate on multiple generated datasets.

# In[ ]:


import random
import torch
from tqdm.auto import tqdm

from generator_functions import build_train_data
from mdl_methods import (
    build_policy_model,
    encode_continuation,
    generate_explanation,
    policy_logprob,
    sample_explanation,
)

# =========================
# Config
# =========================
LR = 5e-5
NUM_EPOCHS = 50
SAMPLES_PER_EXAMPLE = 4
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 64

LAMBDA_PRIOR = 0.30
INJECTION_PROB = 0.25   # probability of replacing one sampled explanation with gold rule
GRAD_CLIP_NORM = 1.0

# =========================
# Training data from generator functions
# Each item is: (observed_sequence, correct_rule)
# =========================
TRAIN_DATA = build_train_data(
    samples_per_rule=2,
    sequence_length=10,
    min_start_index=0,
    max_start_index=15,
    seed=42,
)

# Quick manual sanity check of generated training data format.
print(f"TRAIN_DATA size: {len(TRAIN_DATA)}")
if TRAIN_DATA:
    sample_sequence, sample_rule = TRAIN_DATA[0]
    print(f"TRAIN_DATA sample types: sequence={type(sample_sequence).__name__}, rule={type(sample_rule).__name__}")
    print(f"TRAIN_DATA sample: sequence={sample_sequence}, rule={sample_rule}")



# =========================
# Assumes you already have these from your earlier code:
# - load_model_and_tokenizer
# - render_chat
# - continuation_nll
# - build_conditional_prefix
# - build_prior_prefix
# - build_likelihood_prefix
# - score_explanation(model, tokenizer, observed_sequence, explanation, "bayes", lambda_prior)
# =========================

# =========================
# Init models
# =========================
policy_model = build_policy_model(MODEL_NAME)
evaluator_model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
optimizer = torch.optim.AdamW(policy_model.parameters(), lr=LR)

# =========================
# Training loop
# =========================
for epoch in range(NUM_EPOCHS):  # one full pass over TRAIN_DATA
    policy_model.train()

    all_rewards = []
    all_policy_losses = []

    sampled_examples = random.sample(TRAIN_DATA, k=len(TRAIN_DATA))

    for observed_sequence, gold_rule in sampled_examples:
        inject_index = None
        if random.random() < INJECTION_PROB:
            inject_index = random.randrange(SAMPLES_PER_EXAMPLE)

        local_rewards = []
        rollout_prompt_ids = []
        rollout_gen_ids = []

        for sample_idx in range(SAMPLES_PER_EXAMPLE):
            z, prompt_ids, gen_ids = sample_explanation(
                policy_model,
                tokenizer,
                observed_sequence,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
            )

            if inject_index is not None and sample_idx == inject_index:
                z = gold_rule
                gen_ids = encode_continuation(tokenizer, z)

            # Keep evaluator forward pass out of autograd to reduce memory pressure.
            with torch.no_grad():
                score = score_explanation(
                    evaluator_model,
                    tokenizer,
                    observed_sequence,
                    z,
                    "bayes",
                    LAMBDA_PRIOR,
                )

            reward = -score["loss"]
            local_rewards.append(reward)
            rollout_prompt_ids.append(prompt_ids)
            rollout_gen_ids.append(gen_ids)

        local_rewards = torch.tensor(local_rewards, device=DEVICE)

        baseline = local_rewards.mean()
        advantages = local_rewards - baseline

        optimizer.zero_grad(set_to_none=True)
        policy_loss = torch.tensor(0.0, device=DEVICE)
        for adv, prompt_ids, gen_ids in zip(advantages.detach(), rollout_prompt_ids, rollout_gen_ids):
            logprob = policy_logprob(policy_model, prompt_ids, gen_ids)
            sample_loss = -(adv * logprob) / SAMPLES_PER_EXAMPLE
            sample_loss.backward()
            policy_loss = policy_loss + sample_loss.detach()

        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        all_rewards.append(local_rewards.mean().item())
        all_policy_losses.append(float(policy_loss.detach().item()))

    if (epoch + 1) % 5 == 0 or epoch == 0:
        avg_reward = sum(all_rewards) / len(all_rewards)
        avg_policy_loss = sum(all_policy_losses) / len(all_policy_losses)

        print(f"\nEpoch {epoch+1:03d}")
        print(f"Average reward: {avg_reward:.4f}")
        print(f"Average policy loss: {avg_policy_loss:.4f}")

        # quick qualitative check on 2 examples
        for seq, gold in TRAIN_DATA[:2]:
            pred = generate_explanation(policy_model, tokenizer, seq)
            with torch.no_grad():
                pred_score = score_explanation(
                    evaluator_model,
                    tokenizer,
                    seq,
                    pred,
                    "bayes",
                    LAMBDA_PRIOR,
                )
            print(f"Sequence: {seq}")
            print(f"Gold: {gold}")
            print(f"Pred: {pred}")
            print(f"Bayes loss: {pred_score['loss']:.4f}")
            print("-" * 50)

