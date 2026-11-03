---
title: "System One Models and Calibrated Confidence"
description: "Typed decisions and how Jev's confidence score is computed."
publishDate: '2026-10-16'
tags: [llm]
draft: true
---

## Overview

This post reads TypeSafe's launch post<sup class="cite">[<a href="#ref-1">1</a>]</sup> for *System One models* and its first model, Jev, together with its documentation page on confidence<sup class="cite">[<a href="#ref-2">2</a>]</sup>. A System One model gives up free-form text and returns typed decisions with a probability distribution over outcomes declared in advance.

My takeaway is that **the interesting part is the interface, not the benchmark numbers**. A fixed output space rules out type errors by construction and lets every answer carry a full distribution. The `confidence` field is a deterministic summary of that distribution, **not a separate estimate of correctness**, so its value depends on whether the probabilities are calibrated. The two sources do not yet show direct evidence that they are.

---

## What Is a System One Model?

TypeSafe defines System One models as frontier models built to make fast, structured decisions that software can use directly, and summarizes its first model, Jev, in one line<sup class="cite">[<a href="#ref-1">1</a>]</sup>:

> Think of Jev as a frontier-intelligence function call: unstructured state in, typed probabilistic decisions out.

The name comes from Kahneman's fast, intuitive System 1 thinking. A call takes unstructured **state**, such as text, and a set of **questions** built from typed primitives<sup class="cite">[<a href="#ref-2">2</a>]</sup>:

- **Choice**: one of $n$ unordered options, with probabilities over the options.
- **Score**: one of $n$ ordered levels, with probabilities over the levels.
- **Noul**: a yes/no question, answered with a single probability $p$ of yes.

The launch post contrasts this with LLMs. The table condenses its main claims<sup class="cite">[<a href="#ref-1">1</a>]</sup>; all entries are TypeSafe's.

| | Existing LLMs | Jev |
| --- | --- | --- |
| Output | Strings to parse and validate | Predefined typed values with probabilities |
| Sampling | Sequential, token by token | All outputs in a single parallel query |
| Latency | 3 to 329 s | 70 to 500 ms |
| Input price | \$0.20 to \$10 per million tokens | \$0.042 per million tokens, output free |
| Training | RLHF / RLVR | RLCD (Reinforcement Learning for Calibrated Decisions) |

TypeSafe's argument for giving up strings is that type safety is "table stakes for automation": a hallucinated tool call is tolerable in a chat agent but not deep inside a dependency chain<sup class="cite">[<a href="#ref-1">1</a>]</sup>. Jev's 0% schema error rate is guaranteed by design rather than measured. The architecture, the parallel sampler, and RLCD are named but not described.

### How strong is the evidence?

The launch post attaches its own caveats to each result<sup class="cite">[<a href="#ref-1">1</a>]</sup>:

- **Workflow evaluations** compare each model's answers with reference probabilities averaged from GPT-6 Astra and Fable 5.1. The headline figures of 193.6x faster and 444.6x cheaper come from these; the authors expect them to be at the high end of real-world gains.
- **The side-by-side demo** against GPT-5.6 Terra uses a simplified query that, by the authors' account, favors Jev.
- **No public benchmarks** are reported, by deliberate choice.

My reading is that the workflow evaluations measure **agreement with frontier LLMs**, which supports the claim of similar intelligence at much lower cost. They cannot test calibration, because the reference models are the ones the same post calls overconfident.

---

## How Is Confidence Computed?

Every Choice and Score answer returns `probabilities`, and `confidence` maps their shape to a number between 0 (uniform) and 1 (all mass on one outcome)<sup class="cite">[<a href="#ref-2">2</a>]</sup>.

For a **Choice** with $n$ options and top probability $p_{\max}$, confidence is how far $p_{\max}$ sits above uniform, rescaled to $[0, 1]$:

$$
\text{confidence}_{\text{Choice}} = \frac{p_{\max} - \frac{1}{n}}{1 - \frac{1}{n}}.
$$

A **Noul** returns no confidence field, but the page suggests $|2p - 1|$, which is the Choice formula with $n = 2$.

A **Score** has ordered levels $i = 0, \dots, n-1$, so probability on a distant level should cost more than probability on a neighbor. With most likely level $m$,

$$
\text{confidence}_{\text{Score}} = \max\left(0,\ 1 - \frac{\sum_i p_i \, |i - m|}{\text{MAD}_{\text{unif}}}\right),
\qquad
\text{MAD}_{\text{unif}} = \frac{1}{n} \sum_i \left| i - \frac{n - 1}{2} \right|,
$$

where the numerator is the expected distance from the mode and $\text{MAD}_{\text{unif}}$ is the same quantity for a uniform distribution. The page's three-level examples show the difference from the Choice formula<sup class="cite">[<a href="#ref-2">2</a>]</sup>:

| Probabilities | Score confidence | Choice confidence |
| --- | ---: | ---: |
| $(0, 0.5, 0.5)$: adjacent levels | 0.25 | 0.25 |
| $(0.5, 0, 0.5)$: opposite ends | 0 | 0.25 |

The page recommends gating actions in three tiers (act, confirm, or route to a human) with stricter thresholds for riskier actions. In its example, anything below 0.5 goes to a human, and a money transfer needs confidence above 0.9 to proceed without extra verification<sup class="cite">[<a href="#ref-2">2</a>]</sup>.

---

## What Does Confidence Guarantee?

This section is my interpretation.

**Confidence adds no information beyond the probabilities.** Calibration means that among answers given probability $p$, a fraction $p$ are correct. If Jev's probabilities are calibrated, $p_{\max}$ is the chance the selected option is right, and thresholding confidence is a sound risk control. If they are not, no threshold fixes that. The launch post describes calibration as "higher confidence means higher accuracy"<sup class="cite">[<a href="#ref-1">1</a>]</sup>, which is a ranking property and weaker than calibration: a model can order its answers correctly while every probability is too high. Neither source shows reliability diagrams or accuracy binned by confidence on labeled data, and **this is the evidence I would most want to see**.

**One threshold means different things for different $n$.** Thresholding Choice confidence at $c$ is the same as requiring $p_{\max} > \frac{1}{n} + c\left(1 - \frac{1}{n}\right)$. At $c = 0.5$, that is $p_{\max} > 2/3$ for three options but $p_{\max} > 0.55$ for ten. The normalization makes one threshold reusable across questions, but not one error rate, which is why the documentation's advice to tune thresholds per question on one's own data matters.

Such per-question evaluations can also measure calibration directly, because every answer returns its full distribution rather than only the `confidence` summary.

---

## References

<ol class="references">
<li id="ref-1">

[Diogo Almeida, *Introducing System One Models & Jev* (TypeSafe AI Blog, Sep 15, 2026)](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

</li>
<li id="ref-2">

[TypeSafe, *Confidence* (TypeSafe Documentation, accessed Oct 16, 2026)](https://docs.typesafe.ai/confidence)

</li>
</ol>
