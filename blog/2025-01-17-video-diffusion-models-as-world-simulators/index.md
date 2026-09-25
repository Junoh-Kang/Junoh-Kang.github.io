---
title: Video Diffusion Models as World Simulators
description: 'A review of Oasis: A Universe in a transformer, and Diffusion Forcing:
  Next-token Prediction Meets Full-Sequence Diffusion'
publishDate: '2025-01-17'
tags:
- paper-review
- generative
- video
---

**Slides:** [PDF](/blog/post/20250117/presentation.pdf)

## Overview

**World simulators** are explorable and interactive systems or models that can mimic real world. 
Advanced video generation models can function as world simulators, and to achieve it, they should have **low latency** for input actions, and capable of **long sequence generation**. 
Long sequence generation includes **<u>capability of long generation itself</u>**, **<u>preventing error accumulation</u>**, and **<u>long term context preservation</u>**.
This post mainly focuses on how related project **Oasis: A Universe in a transformer**<sup class="cite">[<a href="#ref-1">1</a>]</sup> deals with **long sequence generation**.

---

## Conventional Long Video Generation are inappropriate for World Simulator!

### Video Diffusion Models (VDMs)

|<img src="/blog/post/20250117/1.vdm.png" alt="" class="zoomable" style="width:90%" />| 
|:--| 
| Training and sampling of video diffusion models. Darker tokens has higher noise levels. |

**<u>Videos are sequential data</u>**.
However, video diffusion models are trained and inferenced to denoise tokens of same noise levels, interpreting each video clip as a single object. 
This section reviews approaches to generate long videos using aforementioned VDMs.

### Chunked autoregressive methods

|<img src="/blog/post/20250117/2.chunked.png" alt="" class="zoomable" style="width:90%" />| 

- Small $k$ (*e.g.* $k=1$) results in high latency for each action since $f-k$ frames are output for each action. Also it tends to lose contexts.
- Large $k$ (*e.g.* $k=f-1$) results in ineifficient training and inference since models learn only $f-k$ tokens, while models calculate for $f$ tokens.
- Chunked autoregressive methods suffers from **<u>quality degradation originated from error accumulation</u>**.

|<img src="/blog/post/20250117/3.chunked_erroraccumulate.png" alt="" class="zoomable" style="width:90%" />| 

### Hierarchical methods (Multi-stage generation)

|<img src="/blog/post/20250117/4.hierarchy.png" alt="" class="zoomable" style="width:90%" />| 

- It does not fit to interactive generation since the end of the video is already determined.

 **Conventional approaches are not appropriate for wolrd simulator!**

|<img src="/blog/post/20250117/5.convention.png" alt="" class="zoomable" style="width:90%" />| 

---

## Long Sequence Generation in Oasis

### Capability of long sequence generation

**Oasis**<sup class="cite">[<a href="#ref-1">1</a>]</sup> follows **Diffusion Forcing**<sup class="cite">[<a href="#ref-2">2</a>]</sup> to train models for long video generation.
**Diffusion Forcing** inherits advantages of Teacher Forcing and Diffusion Models: **<u>flexible time horizon</u>** from Teacher Forcing, **<u>guidance at sampling</u>** from Diffusion Models.

**Diffusion Forcing** trains models to denoise **<u>tokens with independent noise levels</u>**, and sampling noise schedules are carefully chosen depending on the purpose.
The training offers cheaper training than next-token prediction in video domain, and the complexity added by independent noise level is not excessive since the complexity is only in temporal dimension.

|<img src="/blog/post/20250117/6.df_train.png" alt="" class="zoomable" style="width:50%" />| 
|:--:| 
| Training in Diffusion Forcing |

|<img src="/blog/post/20250117/7.df_sample.png" alt="" class="zoomable" style="width:90%" />| 
|:--:| 
| Sampling in Diffusion Forcing |

### Preventing error accumulation

#### The reason of error accumulation

|<img src="/blog/post/20250117/8.vanilla.png" alt="" class="zoomable" style="width:90%" />| 

**Oasis**<sup class="cite">[<a href="#ref-1">1</a>]</sup> and **Diffusion Forcing**<sup class="cite">[<a href="#ref-2">2</a>]</sup> hypothesize that the error accumulation stems from the model erroneously treating generated noisy frames as grount truth (GT), despite their inherent inaccuracies.
They interpret **input noise levels to the models as inversely proportional to the confidence** in the corresponding input tokens.

#### Stable rollout in Diffusion Forcing

|<img src="/blog/post/20250117/9.stable_rollout.png" alt="" class="zoomable" style="width:90%" />| 

**Diffusion Forcing** suggests to deceive models that generated clean tokens are little noisy, preventing models from believing generated tokens as GT.
However, this approach is out of distribution (OOD) inference, and there is no rule of thumb for "little noisy".

#### Stable rollout (Another option)

|<img src="/blog/post/20250117/10.stable_rollout.png" alt="" class="zoomable" style="width:90%" />| 

To avoid OOD, one may suggest add little noise to generated tokens and tell models that the tokens are noisy. 
However, this approach may dilute details in generated tokens.

#### Dynamic Noise Augmentation (DNA)

|<img src="/blog/post/20250117/11.DNA.png" alt="" class="zoomable" style="width:90%" />| 

**Oasis**<sup class="cite">[<a href="#ref-1">1</a>]</sup> suggests Dynamic Noise Augmentation (DNA) to mitigate error accumulation. 
- For initial denoising steps, conditioning tokens (generated tokens) are moderately noised since models tend to generate low-frequency features during initial steps.
- For last denoising steps, noise levels of conditioning tokens gradually decreases.

### Long term context preservation

|<img src="/blog/post/20250117/12.video.gif" alt="" class="zoomable" style="width:50%" />| 

Through above approaches, **Oasis** can autoregressively generate long videos without much quality degradation. 
However, models do not have long time horizon memory, leading to inconsistent videos.
While there is no innovative breakthrough yet, I believe that video models with long-term memory is an important next step.

<iframe src="/blog/post/20250117/presentation.pdf" width="100%" height="600" style="border: none;">
  This browser does not support PDFs. Please download the PDF to view it: <a href="/blog/post/20250117/presentation.pdf">Download PDF</a>
</iframe>

## References

<ol class="references">
<li id="ref-1">

[Etched Decard, *Oasis: A Universe in a Transformer* (2024)](https://oasis-model.github.io/)

</li>
<li id="ref-2">

[Boyuan Chen et al., *Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion* (NeurIPS, 2024)](https://boyuan.space/diffusion-forcing/)

</li>
</ol>
