---
title: PinT algorithms for Diffusion Models
description: A review of researches that accelerate diffusion models in wall clock
  time by parallelization.
publishDate: '2025-08-28'
tags:
- paper-review
- generative
slides: presentation.pdf
---

## Overview

Diffusion models require heavy computation resource due to iterative sampling strategy. 
Due to their sequential sampling strategy, many accleration algorithms trade **sample quality** for **efficiency**.
However, there are a group of researches which trade **compute** for **time**. 
In this post, I review two papers accelerating sampling in time, which are motivated from `PinT (Parallel in Time)` algorithms: 

- Parallel Sampling of Diffusion Models
- Self-Refining Diffusion Samplers: Enabling Paralleization via Parareal Iterations
