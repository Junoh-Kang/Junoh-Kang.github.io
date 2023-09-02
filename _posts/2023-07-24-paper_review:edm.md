---
layout: post
title: Elucidating the Design Space of Diffusion-Based Generative Models
date: 2023-07-24 
# description: an example of a blog post with jupyter notebook
# tags: formatting jupyter
categories: paper-review
giscus_comments: false
related_posts: false
---

#### Contribution Summary
- The paper entangles independent components in diffusion model
- Higher-order Runge-Kutta method
- Third contribution : training of the score-modelig neural network, noise level, non-leaking augmentation


#### Disentangle components of diffusion models 

#### **Training Denoising Network**

#### Forward process formulation

[Song et al.](https://arxiv.org/abs/2011.13456) define the forward SDE/ODE as 
$$
\begin{align}
    dx = f(x,t)dt + g(t)dw. 
    \nonumber
\end{align}
$$
However, instead of defining $f(\cdot,\cdot)$ and $g(\cdot)$, EDM formulates forward SDE/ODE to follow the marginal distribution, $$p_{t}(x(t)|x(0)) = \mathcal{N}(x(t);s(t)x_0, s^2(t) \sigma^2(t) \mathrm{I})$$.
Then the corresponding ODE is 
$$
\begin{align}
    dx = [\dot{s}(t)x/s(t) - s^2(t)\dot{\sigma}(t)\sigma(t) \nabla_x \log p(x/s(t); \sigma(t))]dt, 
    \nonumber
\end{align}
$$
where $$p(x;\sigma)=p_{data} * \mathcal{N}(0, \sigma^2(t)\mathrm{I})$$ and $$p_t(x)=s^{-d}(t) p(x/s(t); \sigma(t))$$.

#### Parametrization

#### Preconditioning
Previous Method : $$D_\theta(x;\sigma) = x - \sigma F_\theta(\cdot)$$ \
EDM : $$ D_\theta(x;\sigma) = c_{skip}(\sigma) x + c_{out}(\sigma) F_\theta(c_{in}(\sigma)x;c_{noise}(\sigma))$$


#### **Sampling**

#### ODE Solver
Euler's method -> Heun's second order method
Write Algorithm later

#### Discretization
Step size should decrease monotonically with decreasing $$\sigma$$.
\begin{align}
    \sigma_{i<N} = ({\sigma_{max}}^{\frac{1}{\rho}} + \frac{i}{N-1}({\sigma_{min}}^\frac{1}{\rho} - {\sigma_{max}}^\frac{1}{\rho}))^{\rho}, \sigma_{N} = 0
    \nonumber
\end{align}
Larger $$\rho$$ results in shorter steps near $$\sigma_{min}$$ and larger steps near $$\sigma_{max}$$.
EDM uses $$\rho=7$$, while $$\rho=3$$ nearly eualizes the truncation error at each timestep.

#### **Results**

|제목 셀1|제목 셀2|제목 셀3|제목 셀4|
|---|---|---|---|
|내용 1|내용 2|내용 3|내용 4|
|내용 5|내용 6|내용 7|내용 8|
|내용 9|내용 10|내용 11|내용 12|