---
layout: distill
title: Elucidating the Design Space of Diffusion-Based Generative Models
date: 2023-01-01
description: 
categories: deep-learning # deep-learning, finance, 
tags: paper-review, # paper-review, survey, 
attachments: 20230903.pdf

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
    # subsections:
    #   - name: Subsection 1
    #   - name: Subsection 2
---

## Overview 

<!-- ---------------------------------------------------------------------- -->
## Generalizing Formulations

#### ODE formulation
**Score-Based Model**<d-cite key="song2021scorebased"></d-cite> defines the flow ODE by its evolution:
$$
  \begin{align}
    d\mathrm{x}_t = \left[ f(\mathrm{x}_t,t)dt - \frac{1}{2} g^2(t) \nabla \log p_t(\mathrm{x}_t)  \right]dt
  \end{align}
$$

where $$f:\mathbb{R}^{h\times w \times 3}\times \mathbb{R} \rightarrow \mathbb{R}^{h\times w \times 3}$$ and $$g:\mathbb{R} \rightarrow \mathbb{R}$$ are predefined functions and $$\mathrm{w}_t$$ is Brownian process. 

On the other hand, **EDM**<d-cite key="karras2022elucidating"></d-cite> defines the marginal distribution first.
$$
  \begin{align}
    p_t(\mathrm{x}_t) 
    &= \int p_{data}(\mathrm{x}_0)p_{t|0}(\mathrm{x}_t | \mathrm{x}_0) d\mathrm{x}_0 \\
    &= \int p_{data}(\mathrm{x}_0)\left[\mathcal{N}(\mathrm{x}_t;s(t)\mathrm{x}_0, s(t)^2\sigma(t)^2\mathrm{I})\right] d\mathrm{x}_0 \\
    &= s(t)^{-d} \left[p_{data} * \mathcal{N}(\mathrm{0}, \sigma(t)^2 \mathrm{I})\right](\mathrm{x}_t/s(t))
  \end{align}
$$

Then, the flow ODE in forms of differential equation is 
$$
  \begin{align}
    d\mathrm{x}_t = \left[
      \dot{s}(t)\mathrm{x}_t / s(t) - s(t)^2 \dot{\sigma}(t)\sigma(t) \nabla_{\mathrm{x}_t}\log p_t(\mathrm{x_t})
    \right] dt.
  \end{align}
$$

**EDM**<d-cite key="karras2022elucidating"></d-cite> considers the case when $$s(t)=1$$:
$$
  \begin{align}
    d\mathrm{x}_t = - \dot{\sigma}(t)\sigma(t) \nabla_{\mathrm{x}_t}\log p_t(\mathrm{x_t}) dt. \label{eq:edm_ode}
  \end{align}
$$

#### SDE formulation
$$
  \begin{align}
    d\mathrm{x}_{t\pm} = 
    \underbrace{-\dot{\sigma}(t)\sigma(t) \nabla_{\mathrm{x}_t}\log p_t(\mathrm{x_t}) dt}_{\text{probability flow ODE}~(\ref{eq:edm_ode})} 
    \pm \underbrace{\underbrace{\beta(t)\sigma(t)^2 \nabla_{\mathrm{x}_t}\log p_t(\mathrm{x_t})dt}_{\text{deterministic noise decay}}  
    + \underbrace{\sqrt{2\beta(t)}\sigma(t) d\mathrm{w}_t}_{\text{noise infection}}}_{\text{Langevin diffusion SDE}},
    \label{eq:edm_sde}
  \end{align}
$$
where '+' denotes forward SDE and $$-$$ denotes backward SDE.

<!-- ---------------------------------------------------------------------- -->
## Enhancements in Sampling

#### Discretization

#### Higher order Runge-Kutta method


<!-- ---------------------------------------------------------------------- -->

## Enhancements in Training

#### Preconditioning 
