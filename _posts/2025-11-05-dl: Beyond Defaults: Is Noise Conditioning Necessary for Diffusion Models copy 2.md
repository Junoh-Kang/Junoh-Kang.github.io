---
layout: distill
title: "Beyond Defaults: Is Noise Conditioning Necessary for Diffusion Models?"
date: 2025-11-05
description: "A review of recent research that challenges the necessity of noise level conditioning in generative models, exploring alternative approaches to denoising and flow matching." 
categories: deep-learning # deep-learning, finance, 
tags: paper-review, # paper-review, survey, 
attachments: /blog/post/20251105/presentation.pdf

giscus_comments: true
related_posts: false
featured: false

authors:
  - name: Junoh Kang
    url: https://junoh-kang.github.io/
    affiliations:
      name: Seoul National University

bibliography: diffusion.bib

---

## Overview

Traditional diffusion models explicitly specify noise level to neural network to model score function. However, recent research has begun to question whether this conditioning is truly necessary. In this post, I review two papers that challenge this fundamental assumption and propose alternative approaches:

**"Is Noise Conditioning Necessary for Denoising Generative Models?"**
  - This paper challenges the convention by demonstrating that denoising networks can perform effectively without explicit noise level conditioning, suggesting that models may inherently learn to estimate noise levels from the input data itself.
  
**"Equilibrium Matching: Generative Modeling with Implicit Energy-based Models"**
  - This work reformulates the generative modeling problem as learning an energy landscape, providing a theoretical foundation for noise-unconditioning approaches.

---

<iframe src="/blog/post/20251105/presentation.pdf" width="800" height="600" style="border: none;">
  This browser does not support PDFs. Please download the PDF to view it: <a href="/blog/post/20251105/presentation.pdf">Download PDF</a>
</iframe>




