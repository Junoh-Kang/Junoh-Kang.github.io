---
layout: distill
title: "Post-Training of Modern LLMs"
date: 2026-06-19
description: "A review of modern LLM post-training, from RLHF and DPO to verifier-based reinforcement learning, GRPO, and the DeepSeek-R1 pipeline."
categories: deep-learning # deep-learning, finance,
tags: paper-review, llm, post-training

giscus_comments: true
related_posts: false
featured: false

authors:
  - name: Junoh Kang
    url: https://junoh-kang.github.io/
    affiliations:
      name: Seoul National University

toc:
  - name: Overview
  - name: From Next-token Prediction to RLHF
  - name: From Preference Rewards to Verifier Rewards
  - name: GRPO and DeepSeekMath
  - name: R1-Zero and DeepSeek-R1
  - name: GRPO Normalization Bias
  - name: Takeaway

---

## Overview

Modern LLMs are not finished after pre-training. Pre-training gives the model broad next-token prediction capability, but the model still needs a separate stage that shapes how it answers instructions, handles preference tradeoffs, and spends computation on reasoning. This later stage is usually called post-training.

The useful way to read recent post-training work is not as a list of unrelated tricks. The methods form a chain. The first shift is from next-token prediction to preference-based post-training. Inside that shift, PPO-based RLHF and DPO-based optimization are two ways to use preference data. RLVR then changes the reward source again: when the final answer can be checked by a rule, it uses verifier rewards instead of learned preference rewards.

This post follows that chain. The main examples are InstructGPT-style RLHF, Direct Preference Optimization, DeepSeekMath's GRPO, R1-Zero, and the final DeepSeek-R1 pipeline.

---

## From Next-token Prediction to RLHF

The limitation of next-token prediction is that it imitates text rather than directly optimizing assistant behavior. A base model can learn fluent language from web-scale data, but helpfulness, truthfulness, and harmlessness are not direct training targets. More importantly, next-token prediction does not provide direct negative feedback on what behavior to avoid. Bad or low-quality text can still be imitated if it appears in the data.

RLHF adds a preference signal after supervised instruction tuning. The InstructGPT-style pipeline has three stages.

| Stage | Role |
| --- | --- |
| SFT | Imitate labeler-written answers |
| RM | Train reward model from ranked answers |
| PPO | Optimize the policy against the reward model |

### PPO-based RLHF

PPO-based RLHF is the explicit reward-model route. The RL objective is not just "maximize reward." It also constrains the updated policy to stay close to a reference policy. In simplified form, RLHF optimizes a reward term plus a KL penalty:

$$
\max_{\pi}\;
\mathbb{E}_{x,y}
\left[
r(x,y)
- \beta \log
\frac{\pi(y \mid x)}
{\pi_{\mathrm{ref}}(y \mid x)}
\right].
$$

This KL term matters. Without it, the policy can exploit reward-model errors and drift away from language behavior that the reference model already handles well.

### DPO-based preference optimization

DPO is the direct preference-optimization route inside the same preference-based framing. It starts from the same KL-regularized preference view, but removes the explicit reward-model and PPO stages. Its key observation is that the optimal reward can be represented by a policy/reference log-ratio:

$$
\hat r_\theta(x,y)
=
\beta \log
\frac{\pi_\theta(y \mid x)}
{\pi_{\mathrm{ref}}(y \mid x)}.
$$

The resulting loss directly compares a preferred answer $y_w$ and a dispreferred answer $y_l$:

$$
L_{\mathrm{DPO}}(\pi_\theta;\pi_{\mathrm{ref}})
=
-\mathbb{E}_{(x,y_w,y_l)}
\left[
\log \sigma\left(
\beta \log \frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-
\beta \log \frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right)
\right].
$$

This is why DPO is sometimes described as RL-free. That phrase is easy to misunderstand. DPO removes the explicit reward model and PPO optimization loop, but it still uses preference pairs, a reference policy, and a KL-regularized objective.

---

## From Preference Rewards to Verifier Rewards

Preference-based RLHF is useful, but the reward signal is expensive and subjective. Human labelers or reward models must judge completed responses. The learned reward model can also be over-optimized, especially when the policy discovers answers that score well under the reward model without being genuinely useful.

Verifier-based RL changes the reward source. In math, code, and some reasoning tasks, the final answer can be checked. A math answer can be compared against the known answer. A program can be run against tests. In these settings, the training loop does not need a learned preference model for every completed answer.

| Setting | Reward signal | Natural domain |
| --- | --- | --- |
| RLHF / DPO | Human or model preference | Chat, helpfulness, safety |
| RLVR | Rule-based outcome reward | Math, code, verifiable reasoning |

This distinction is the bridge from preference alignment to reasoning RL. RLVR is attractive because the reward can be less subjective. It is also limited because outcome reward does not explain which reasoning step was wrong. A failed long solution receives a low final reward, but the training signal does not automatically identify the local mistake.

---

## GRPO and DeepSeekMath

DeepSeekMath introduces Group Relative Policy Optimization, or GRPO, as a PPO-style method for mathematical reasoning. The main engineering problem is the learned value model. PPO normally uses a value function to estimate advantages. For LLM reasoning, that value model can be another large model, and math rewards often arrive only after a full solution is generated.

GRPO removes the learned critic. For each question, it samples a group of outputs, scores them, normalizes the rewards inside the group, and uses those group-relative values as advantages.

The GRPO objective can be remembered as a grouped PPO clipped objective with a reference-policy KL penalty:

$$
\begin{aligned}
J_{\mathrm{GRPO}}(\theta)
&=
\mathbb{E}_{q,\,\{o_i\}_{i=1}^{G}}
\Bigg[
\frac{1}{G}\sum_{i=1}^{G}
\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
\left(
\min\left(
\rho_{i,t}(\theta)A_i,\,
\operatorname{clip}\left(\rho_{i,t}(\theta),1-\epsilon,1+\epsilon\right)A_i
\right)
-\beta\mathbb{D}_{\mathrm{KL}}\left(\pi_\theta\Vert\pi_{\mathrm{ref}}\right)
\right)
\Bigg].
\end{aligned}
$$

The group-relative advantage is:

$$
A_i
=
\frac{
r_i-\operatorname{mean}(\{r_1,r_2,\ldots,r_G\})
}{
\operatorname{std}(\{r_1,r_2,\ldots,r_G\})
}.
$$

The core move is critic removal, not reward removal. DeepSeekMath-RL still scores sampled outputs and still uses a reference policy for KL regularization. GRPO should therefore be read as a PPO variant, not as a wholly separate reinforcement learning family.

The reported DeepSeekMath-RL results are also scoped. The RL stage improves DeepSeekMath-Instruct 7B on GSM8K and MATH, but it sits on top of a strong base model, a math corpus, and SFT. The result is evidence that RL can reshape the output distribution toward better sampled answers. It is not evidence that GRPO alone created the whole mathematical ability of the model.

---

## R1-Zero and DeepSeek-R1

DeepSeek-R1 is often summarized too loosely. The important split is between R1-Zero and the final DeepSeek-R1 pipeline.

R1-Zero is the cleaner RLVR case. It applies RL directly to DeepSeek-V3-Base with rule-based rewards for verifiable reasoning tasks. On AIME 2024, the paper reports pass@1 rising from 15.6% to 77.9%, and self-consistency with 16 samples reaching 86.7%. It also reports longer generated reasoning during training.

The final DeepSeek-R1 model is not a pure RLVR pipeline. It adds several additional stages:

1. Cold-start SFT gives the base model a readable long-CoT format before RL.
2. A first RL stage improves reasoning while reducing language mixing.
3. Rejection sampling and general-purpose SFT broaden the model beyond math and code.
4. Distillation transfers behavior into smaller models.
5. A final RL stage mixes verifiable rewards with helpfulness and safety reward models.

This distinction changes the conclusion. RLVR is load-bearing in the reasoning story, but the final DeepSeek-R1 behavior is produced by a hybrid post-training pipeline. Calling final R1 "pure RLVR" hides the SFT, rejection sampling, distillation, and preference-reward components that make the model usable as a general assistant.

---

## GRPO Normalization Bias

GRPO removes the learned critic, but it does not remove all design choices from the objective. Two normalization terms are especially important.

$$
\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}L^{\mathrm{PPO}}_{i,t}
\qquad
A_i =
\frac{r_i-\operatorname{mean}(r)}
{\operatorname{std}(r)}.
$$

The first term is length normalization. Because the token-level loss is averaged by output length, a long incorrect output can receive a weaker per-token penalty. If a model expects to be wrong, lengthening the answer can dilute the penalty signal.

The second term is reward standard-deviation normalization. Dividing by $\operatorname{std}(r)$ can give larger update weight to groups with low reward variance. This can overweight questions where sampled answers are all similarly correct or all similarly wrong. The practical severity of this bias depends on the training setup, but it is a real design issue.

Dr. GRPO is motivated by this caveat. It removes both the length normalization and the reward standard-deviation normalization terms. The point is not that GRPO is unusable. The point is that critic-free PPO still has normalization choices, and those choices can change what the policy update emphasizes.

---

## Takeaway

Modern LLM post-training is moving along a clear axis. The field starts with imitation, adds preference optimization, and then uses verifiable rewards where the task permits it.

PPO-based RLHF is the classic preference-reward pipeline. DPO-based preference optimization keeps the preference framing but turns the reward-model view into a direct policy loss. GRPO removes the learned critic from PPO-style LLM RL by using group-relative rewards. R1-Zero shows that rule-based RLVR can strongly improve reasoning behavior in a verifiable domain. Final DeepSeek-R1 shows that a usable reasoning assistant is still a hybrid system, not a pure RLVR artifact.

The clean summary is this: post-training is no longer only about making a model sound helpful. It is increasingly about choosing the right feedback signal for the behavior we want to reinforce.

## References

- Ouyang et al., *Training language models to follow instructions with human feedback* (2022).
- Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (2023).
- Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (2024).
- Guo et al., *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (2025).
- Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective* (2025).
