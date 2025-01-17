---
layout: distill
title: Video Diffusion Models as World Simulators
date: 2025-01-17
description: "A review of Oasis: A Universe in a transformer, and Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion" 

categories: deep-learning # deep-learning, finance, 
tags: paper-review, # paper-review, survey, 
attachments: https://docs.google.com/presentation/d/1anuY_SaTFu5gXceB_qtCFLKER0aAlHsJq6CWN9KAGhE/edit?usp=sharing

giscus_comments: true
related_posts: false
featured: false

authors:
  - name: Junoh Kang
    url: https://junoh-kang.github.io/
    affiliations:
      name: Seoul National University

bibliography: diffusion.bib

toc:
  - name: Overview 
  - name: Conventional Long Video Generation
  - name: Long Sequence Generation
---

## Overview

**World simulators** are explorable and interactive systems or models that can mimic real world. 
Advanced video generation models can function as world simulators, and to achieve it, they should have **low latency** for input actions, and capable of **long sequence generation**. 
Long sequence generation includes **<u>capability of long generation itself</u>**, **<u>preventing error accumulation</u>**, and **<u>long term context preservation</u>**.
This post mainly focuses on how related project **Oasis: A Universe in a transformer**<d-cite key="decart2024oasis"></d-cite> deals with **long sequence generation**.

---

## Conventional Long Video Generation are Inappropriate for World Simulator!

### Video Diffusion Models (VDMs)

|![]({{ "/blog/post/20250117/1.vdm.png" | relative_url }}){:style="margin:auto; display:block;width:90%; height:auto;"}| 
|:--| 
| Training and sampling of video diffusion models. Darker tokens has higher noise levels. |

**<u>Videos are sequential data</u>**.
However, video diffusion models are trained and inferenced to denoise tokens of same noise levels, interpreting each video clip as a single object. 
This section reviews approaches to generate long videos using aforementioned VDMs.


### Chunked Autoregressive Methods

|![]({{ "/blog/post/20250117/2.chunked.png" | relative_url }}){:style="margin:auto; display:block;width:90%; height:auto;"}| 

- Small $$k$$ (*e.g.* $$k=1$$) results in high latency for each action since $$f-k$$ frames are output for each action. Also it tends to lose contexts.
- Large $$k$$ (*e.g.* $$k=f-1$$) results in ineifficient training and inference since models learn only $$f-k$$ tokens, while models calculate for $$f$$ tokens.
- Chunked autoregressive methods suffers from **<u>quality degradation originated from error accumulation</u>**.

|![]({{ "/blog/post/20250117/3.chunked_erroraccumulate.png" | relative_url }}){:style="margin:auto; display:block;width:90%; height:auto;"}| 

### Hierarchical Methods (Multi-stage Generation)

|![]({{ "/blog/post/20250117/4.hierarchy.png" | relative_url }}){:style="margin:auto; display:block;width:90%; height:auto;"}| 

- It does not fit to interactive generation since the end of the video is already determined.

 **Conventional approaches are not appropriate for wolrd simulator!**

|![]({{ "/blog/post/20250117/5.convention.png" | relative_url }}){:style="margin:auto; display:block;width:90%; height:auto;"}| 


---

## Long Sequence Generation in Oasis

### Capability of Long Sequence Generation

**Oasis**<d-cite key="decart2024oasis"></d-cite> follows **Diffusion Forcing**<d-cite key="chen2024diffusionforcing"></d-cite> to train models for long video generation.
**Diffusion Forcing** inherits advantages of Teacher Forcing and Diffusion Models: **<u>flexible time horizon</u>** from Teacher Forcing, **<u>guidance at sampling</u>** from Diffusion Models.

**Diffusion Forcing** trains models to denoise **<u>tokens with independent noise levels</u>**, and sampling noise schedules are carefully chosen depending on the purpose.
The training offers cheaper training than next-token prediction in video domain, and the complexity added by independent noise level is not excessive since the complexity is only in temporal dimension.

|![]({{ "/blog/post/20250117/6.df_train.png" | relative_url }}){:style="margin:auto; display:block;width:50%; height:auto;"}| 
|:--:| 
| Training in Diffusion Forcing |

|![]({{ "/blog/post/20250117/7.df_sample.png" | relative_url }}){:style="margin:auto; display:block;width:90%; height:auto;"}| 
|:--:| 
| Sampling in Diffusion Forcing |

### Preventing Error Accumulation

#### The reason of Error Accumulation

|![]({{ "/blog/post/20250117/8.vanilla.png" | relative_url }}){:style="margin:auto; display:block;width:90%; height:auto;"}| 

**Oasis**<d-cite key="decart2024oasis"></d-cite> and **Diffusion Forcing**<d-cite key="chen2024diffusionforcing"></d-cite> hypothesize that the error accumulation stems from the model erroneously treating generated noisy frames as grount truth (GT), despite their inherent inaccuracies.
They interpret **input noise levels to the models as inversely proportional to the confidence** in the corresponding input tokens.


#### Stable Rollout in Diffusion Forcing

|![]({{ "/blog/post/20250117/9.stable_rollout.png" | relative_url }}){:style="margin:auto; display:block;width:90%; height:auto;"}| 

**Diffusion Forcing** suggests to deceive models that generated clean tokens are little noisy, preventing models from believing generated tokens as GT.
However, this approach is out of distribution (OOD) inference, and there is no rule of thumb for "little noisy".

#### Stable Rollout (Another Option)

|![]({{ "/blog/post/20250117/10.stable_rollout.png" | relative_url }}){:style="margin:auto; display:block;width:90%; height:auto;"}| 

To avoid OOD, one may suggest add little noise to generated tokens and tell models that the tokens are noisy. 
However, this approach may dilute details in generated tokens.

#### Dynamic Noise Augmentation (DNA)

|![]({{ "/blog/post/20250117/11.DNA.png" | relative_url }}){:style="margin:auto; display:block;width:90%; height:auto;"}| 

**Oasis**<d-cite key="decart2024oasis"></d-cite> suggests Dynamic Noise Augmentation (DNA) to mitigate error accumulation. 
- For initial denoising steps, conditioning tokens (generated tokens) are moderately noised since models tend to generate low-frequency features during initial steps.
- For last denoising steps, noise levels of conditioning tokens gradually decreases.


### Long Tern Context Preservation


|![]({{ "/blog/post/20250117/12.video.gif" | relative_url }}){:style="margin:auto; display:block;width:50%; height:auto;"}| 

Through above approaches, **Oasis** can autoregressively generate long videos without much quality degradation. 
However, models do not have long time horizon memory, leading to inconsistent videos.
While there is no innovative breakthrough yet, I believe that video models with long-term memory is an important next step.


