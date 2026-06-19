---
layout: distill
title: "One-step Generation in the Post Diffusion Era"
date: 2026-03-12
description: " A review of one-step generative modeling beyond diffusion-time iteration, covering Consistency Models, CTM, MeanFlow, DMD, and the 2026 Drifting Models framework." 
categories: [survey]
tags: [generative]
attachments: /blog/post/20260312/presentation.pdf

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

This post reviews the shift from iterative diffusion and flow models toward one-step generation. \
After summarizing prior approaches such as Consistency Models, CTM, MeanFlow, and DMD, it focuses on Drifting Models, which replace inference-time dynamics with training-time distribution evolution and learn an explicit drift field to align generated and data distributions.
---

<iframe src="/blog/post/20260312/presentation.pdf" width="800" height="600" style="border: none;">
  This browser does not support PDFs. Please download the PDF to view it: <a href="/blog/post/20260312/presentation.pdf">Download PDF</a>
</iframe>




