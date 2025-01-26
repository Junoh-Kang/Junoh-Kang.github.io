---
layout: distill
title: Guiding a Diffusion Model with a Bad Version of Itself
date: 2025-01-20
description: A paper review on a paper, Guiding a Diffusion Model with a Bad Version of Itself.
categories: deep-learning # deep-learning, finance, 
tags: paper-review, # paper-review, survey, 
attachments: /blog/post/

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
  - name: Section
    subsections:
      - name: Subsection 1
      - name: Subsection 2
---

<!-- 
# Image
|![]({{ "/blog/post/path_to_image.png" | relative_url }}){:style="margin:auto; display:block;width:90%; height:auto;"}| 
|:--:| 
|  | 
# Cite
**DDPM**<d-cite key="ho2020denoising"></d-cite>
-->

## Overview

**Classifier-Free Diffusion Guidance (CFG)**<d-cite key="ho2021cfg"></d-cite> is widely used in text-to-image models to generate high-quality images which align well with texts. 
While **CFG** was initially proposed to enhance image-prompt alignment, it has been found to improve fidelity as well.
**Guiding a Diffusion Model with a Bad Version of Itself**<d-cite key="karras2021guiding"></d-cite> analyzes why CFG improves image quality and suggests **Autoguidance**, which can enhance image quality without sacrificing diversity, unlike **CFG**.

## Why does CFG improve image quality?

### Classifier-Free Guidance

![]({{ "/blog/post/20250120/teaser.png" | relative_url }}){:style="margin:auto; display:block;width:90%; height:auto;"}

**CFG** is a method to sample $$\textbf{x}|\textbf{c}$$ with only text-to-image models $$\epsilon_\theta(x_t, c)$$ without additional models such as classifier.
The method is to update latents with 
$$
\begin{align}
  \epsilon_\theta(\textbf{x}_t, \textbf{c}) + w (\epsilon_\theta(\textbf{x}_t, \textbf{c}) - \epsilon_\theta(\textbf{x}_t, \phi)),
\end{align}
$$
and common value for the guidance is $$w>1$$.
The higher **CFG** results in better text alignment and enhanced image qualities. 
However, it suffers from low diversity as shown in the leftmost column of the following figure.


### Observing sample behaviors in toy experiments

|![]({{ "/blog/post/20250120/toy_sample.png" | relative_url }}){:style="margin:auto; display:block;width:90%; height:auto;"}|
|:--|
| Fig 1. **Guiding a Diffusion Model with a Bad Version of Itself**<d-cite key="karras2021guiding"></d-cite> conducts experiment on 2D toy distribution, which is anisotropic and has narrow support.|

#### Score matching leads to outliers

KL-divergence incurs extreme penalities if the model underestimates the likelikhood of training sample.
Consequantly, diffusion models estimate conservative fit of data distribution as shown in Fig 1(b).

#### CFG eliminates outliers

We can observe in Fig 1(c) that CFG samples avoid class boundaries, and are pulled towards the center of the manifold. 
Authors argue that the outlier elimination attributes to the image quality improvement.
However,they also argue that the concentration cannot be solely explained by increase in log-likelihood.

#### Quality difference between the conditional and unconditional denoisers causes the elimination

|![]({{ "/blog/post/20250120/toy_field.png" | relative_url }}){:style="margin:auto; display:block;width:90%; height:auto;"}|
|:--|
| Fig 2. Visualization of probability density and score functions. $$p_1$$ and $$p_0$$ are those of conditional and unconditional models, respectively.|

CFG can be interpreted as pushing samples to higher value of $$p_1 / p_0$$.
Note that 
- $$p_1$$ (conditional model) has tighter estimation on target distribution boundary,
- $$p_0$$ (unconditional model) has loose estimation on target distribution boundary,

as shown in Fig 2(a) and (b); unconditional model is solving harder problem than the conditional model.
As a result, the landscape of ratio $$p_1 / p_0$$ becomes steeper, and samples concentrates.


<!-- Moreover, the following graph explains some samples 
notion of why some samples are concentrated to boundary in Fig 2(c).
![]({{ "/blog/post/20250120/toy_graph.png" | relative_url }}){:style="margin:auto; display:block;width:50%; height:auto;"} -->

## Isolating Sample Quality Improvements 

The above experiments 