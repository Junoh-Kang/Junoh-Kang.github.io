---
layout: distill
title: "LOB-Bench: Orderbook Generative Model Evaluation Benchmark"
date: 2026-06-24
description: "A review of LOB-Bench, focusing on why generated order book models need rollout-level evaluation beyond one-step loss and stylized facts."
categories: [paper-review]
tags: [finance, generative]

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
  - name: LOB Generation Needs a Rollout Benchmark
  - name: What LOB-Bench Measures
    subsections:
      - name: "Group 1: Score-distribution Metrics"
      - name: "Group 2: Market-impact Response Functions"
      - name: "Group 3: Adversarial Measurement"
  - name: Positioning Current LOB Generators
  - name: References
---

## Overview

LOB-Bench starts from a critique of how generative limit order book models are usually evaluated. One-step prediction loss and a few stylized facts can be useful local checks, but they do not show whether a model remains realistic after it samples a trajectory. A model can pass those narrower checks and still drift away from realistic market behavior once it generates a sequence autoregressively.

The benchmark therefore evaluates generated LOBSTER-compatible order book rollouts by comparing real and generated score distributions, conditional behavior, market-impact response curves, and discriminator separability. This makes it useful for diagnosing where a generated market trajectory fails: quote state, event timing, cancellation behavior, event placement, order flow, price response, or trajectory-level artifacts.

The reported results make the benchmark's role concrete. LOBS5 is the strongest tested model overall, but LOB-Bench still exposes growing horizon error, conditional microstructure failures, market-impact response gaps, and discriminator separability. The useful output is therefore not a single leaderboard score, but a failure profile for generated order book trajectories.

---

## LOB Generation Needs a Rollout Benchmark

Limit order book data is hard to generate because it mixes discrete events, continuous prices and quantities, irregular timing, strategic interaction, and market microstructure constraints. A generator that emits message-level data must keep the book coherent over time. A generator that emits book states directly must still preserve spread, depth, liquidity, return, and order-flow behavior across a rollout.

The evaluation problem is that common checks answer narrower questions. Held-out next-token cross-entropy tests whether the model predicts the next event under real history, but it does not test the distribution of sampled trajectories. Stylized facts test selected marginal patterns, but they can miss failures in timing, cancellation behavior, event placement, or response dynamics. Fragmented realism metrics can show isolated failures, but they do not by themselves produce a benchmark-level failure profile.

LOB-Bench is the paper's answer to that gap. It evaluates generated samples after rollout, when the conditioning history has started to contain model-generated outputs. The target question is not just "did the model predict the next event?" The target question is "does the generated market trajectory still look like real LOB data after the model runs?"

---

## What LOB-Bench Measures

LOB-Bench starts from generated, real, and conditioning sequences stored in LOBSTER-compatible CSV format. This input contract matters. If a model generates messages, those messages need to be exported or replayed into comparable message and book-state sequences. If a model generates another representation, it needs a conversion step before the default benchmark metrics apply.

At a high level, the benchmark has three measurement groups.

| Group | Flow | What it catches |
| --- | --- | --- |
| Group 1: score-distribution metrics | Sequence $d$ -> scalar score $\Phi(d)$ -> real/generated distribution comparison | Broad microstructure realism across book state, timing, lifecycle, event placement, and order flow. |
| Group 2: market-impact response functions | Event class $\pi$ -> sign-adjusted mid-price response curve $R_{\pi}(l)$ -> real/generated curve comparison | Whether generated trajectories preserve average price response after market events. |
| Group 3: adversarial measurement | Orderbook-state trajectory -> compact state-change representation -> discriminator score | Whether a learned classifier can still separate generated trajectories from real ones. |

### Group 1: Score-distribution Metrics

Group 1 is the main distributional evaluation layer.

1. **Project a sequence to a scalar.** A scoring function $\Phi$ maps a generated or real sequence $d$ into a scalar score $\Phi(d)$.

2. **Compare real and generated score distributions.** LOB-Bench uses histogram L1 distance for binned distribution mismatch and Wasserstein-1 distance for how far score values move.

3. **Repeat the comparison conditionally.** The benchmark can bucket by a context variable or conditioning score, compare the target score distribution inside each bucket, and average the bucket-level errors.

4. **Track the same error by rollout horizon.** Conditioning the comparison on rollout step shows whether a model stays realistic as it samples farther away from the real conditioning history.

#### Default Score Families

The concrete $\Phi$ functions are best read as a menu of probes. They are not the whole benchmark; they are the scalar projections used by Group 1.

| Metric family | Examples | What it checks |
| --- | --- | --- |
| Quote and book state | Bid-ask spread, order-book imbalance, bid and ask volume, best-level volume | Whether top-of-book tightness, balance, and liquidity look realistic. |
| Message timing and lifecycle | Inter-arrival time, time-to-cancel | Whether event timing and cancellation lifetimes match real data. |
| Event placement | Limit-order depth, cancellation depth, limit-order level, cancellation level | Whether new limits and cancellations occur at realistic distances from the mid-price or book levels. |
| Trading and order flow | Volume per minute, order-flow imbalance, OFI conditional on next mid-price move | Whether trading pressure and directional order-flow patterns are preserved. |

### Group 2: Market-impact Response Functions

Group 2 measures whether generated trajectories preserve average price responses around market events.

1. **Choose an event class.** LOB-Bench groups events such as market orders, limit orders, and cancellations, and separates them by whether they change the mid-price.

2. **Compute a sign-adjusted response curve.** For each event class, it tracks the future mid-price response across a lag grid and aligns the sign with the event's expected price-pressure direction. If $p_t$ is the mid-price at event time $t$ and $\epsilon_t$ is this event-aligned sign, the response curve for event class $\pi$ is

   $$
   R_{\pi}(l) = \left\langle (p_{t+l} - p_t)\epsilon_t \mid \pi_t = \pi \right\rangle_T.
   $$

3. **Compare real and generated curves.** The benchmark measures the gap between the real response curve and the generated response curve. For one event class, this is the mean absolute curve gap across lags:

   $$
   \Delta R_{\pi} = \frac{1}{L}\sum_l \left|R_{\pi}^{\mathrm{real}}(l)-R_{\pi}^{\mathrm{gen}}(l)\right|.
   $$

This is a stronger test than a marginal statistic. A model that matches spreads and volumes can still fail if it does not reproduce how prices react after order-flow shocks. The paper reports that LOBS5 reproduces GOOG response curves much better than a stochastic baseline in the discussed comparison.

### Group 3: Adversarial Measurement

Group 3 asks whether a learned classifier can still find trajectory-level artifacts that hand-written metrics miss.

1. **Start from orderbook-state trajectories.** The adversarial check uses real and generated book-state sequences.

2. **Map state changes into a compact representation.** LOB-Bench represents sparse book updates through features such as mid-price change, relative price level, and quantity change.

3. **Train a discriminator.** The discriminator tries to classify whether a trajectory is real or generated.

4. **Read separability as an artifact signal.** If the discriminator separates real and generated samples well, then the generated data still contains detectable structure that the interpretable metrics may not fully describe.

---

## Positioning Current LOB Generators

LOB-Bench is most useful as a map of where current LOB generators stand. LOBS5 is the strongest tested model overall, but it is not treated as solved market simulation: the benchmark still exposes horizon error, conditional microstructure failures, response-curve gaps, and discriminator separability.

The comparisons also show why the generated object matters. Message-only RWKV variants diverge quickly on price levels and book-volume-related statistics, which suggests that plausible event tokens are not enough when the book state drifts. Hand-coded or parametric baselines can match some placement statistics, such as depths and levels, while still missing imbalance, volume, timing, and market-impact behavior.

This makes LOB-Bench a positioning tool rather than only a leaderboard. It helps say whether a model is mainly good at static score distributions, conditional behavior, event-response dynamics, or adversarial trajectory realism. It still does not validate a single inserted action, queue position, fill probability, order identity, or matching-engine correctness; those claims need simulator- and execution-specific checks.

---

## References

- Nagy et al., *LOB-Bench: Benchmarking Generative AI for Finance -- an Application to Limit Order Book Markets* (2025), arXiv:2502.09172.
