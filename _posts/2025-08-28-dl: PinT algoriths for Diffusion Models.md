---
layout: distill
title: PinT algorithms for Diffusion Models
date: 2025-08-28
description: "A review of researches that accelerate diffusion models in wall clock time by parallelization." 
categories: [survey]
tags: [generative]
attachments: /blog/post/20250828/presentation.pdf

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

Diffusion models require heavy computation resource due to iterative sampling strategy. 
Due to their sequential sampling strategy, many accleration algorithms trade **sample quality** for **efficiency**.
However, there are a group of researches which trade **compute** for **time**. 
In this post, I review two papers accelerating sampling in time, which are motivated from `PinT (Parallel in Time)` algorithms: 

- Parallel Sampling of Diffusion Models
- Self-Refining Diffusion Samplers: Enabling Paralleization via Parareal Iterations

---

<iframe src="/blog/post/20250828/presentation.pdf" width="800" height="600" style="border: none;">
  This browser does not support PDFs. Please download the PDF to view it: <a href="/blog/post/20250828/presentation.pdf">Download PDF</a>
</iframe>

