---
title: "Learning to tokenize (1): The bitter lesson meets tokenization"
publishDate: '2026-11-16'
tags: [representation]
language: en
series: "Learning to tokenize"
seriesOrder: 1
draft: false
---

<!--
Series scope: My understanding of H-Net and the philosophy and claims in
Cartesia's blog, including Parts 1–3. Compare the design choices first, then
explore the behaviors they enable, without technical implementation details.

Naming convention: Following Cartesia's blog, call the original paper's model
LH-Net and the improved follow-up model H-Net. Explain this convention at
first mention so that the paper and the blog results remain distinct.

Sources:
- H-Net paper: https://www.alphaxiv.org/pdf/2507.07955
- Cartesia blog: https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1
-->

## Modern LLMs through the lens of the bitter lesson

> **general methods that leverage computation** are ultimately the most effective, and by a large margin.
>
> — Richard Sutton, [*The bitter lesson* (2019)](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf)

Modern LLMs illustrate the bitter lesson through training methods that obtain learning signals at scale:

- **Next-token prediction** turns existing text into training examples: the next token is already there to serve as the target. This makes large text corpora usable for learning without separately labeling each example, as demonstrated by [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
- **Reinforcement learning with verifiable rewards (RLVR)** uses automatically checkable outcomes, such as correct answers or passing tests, to reward generated solutions. [DeepSeek-R1-Zero](https://arxiv.org/html/2501.12948v1#S2.SS2) showed how this can improve a pretrained model’s reasoning without requiring human-written step-by-step solutions for its RL training.

However, a tokenizer still determines the units these models learn from. Chunking bytes into larger tokens shortens sequences and reduces the computational cost of standard Transformers. These merge rules remain fixed during language-model training: the model learns patterns within and across tokens, but cannot change how the text is divided.

The example below shows how GPT-2's tokenizer represents `hello world` as two tokens instead of eleven byte positions.

<figure>
<iframe src="/blog/post/20261116/hello-world-tokenization.html" title="Hello world: bytes and BPE tokens" width="100%" height="172" style="display: block; width: 100%; border: 0; margin: 0 auto;"></iframe>
  <figcaption style="font-size: 0.8rem; line-height: 1.5; font-weight: 400; color: hsl(var(--muted-foreground, 0 0% 45%)); margin: 0.4rem 0 0; text-align: center;">GPT-2 encodes <code>hello world</code> as two tokens; · marks a space. <a style="color: inherit; font-size: inherit; font-weight: inherit; text-underline-offset: 2px;" href="https://github.com/openai/gpt-2/blob/master/src/encoder.py">Source: OpenAI</a>.</figcaption>
</figure>

<details>
<summary>How byte pair encoding (BPE) tokenizes text</summary>

Byte-level BPE builds a vocabulary by repeatedly merging the most frequent adjacent pair of units in a training corpus, starting from individual byte values. These merge rules are learned before language-model training and remain fixed throughout it. The interactive example below steps through how such merge rules are learned on a toy corpus, one merge per click.

<figure>
<iframe src="/blog/post/20261116/bpe-demo.html" title="Learning BPE merge rules" width="100%" height="460" style="display: block; width: 100%; border: 0; margin: 0 auto;"></iframe>
<figcaption style="font-size: 0.8rem; line-height: 1.5; font-weight: 400; color: hsl(var(--muted-foreground, 0 0% 45%)); margin: 0.4rem 0 0; text-align: center;">BPE merges on a toy character corpus from Sennrich et al.<sup class="cite">[<a style="color: inherit; font-size: inherit; font-weight: inherit; text-underline-offset: 2px;" href="#ref-2">2</a>]</sup></figcaption>
</figure>

</details>

This raises a question through the lens of the bitter lesson: **should the model also learn how to form the units it processes?**

---
## Hierarchical networks (H-Nets)

Cartesia's answer is yes. In its view, intelligence means learning from raw data, and that includes **learning how to group the data into useful units**. This is similar to how people learn: as psychologist [George A. Miller](https://psychclassics.yorku.ca/Miller/) illustrated with Morse code, a learner first hears individual dots and dashes, and later recognizes whole letters and words.

H-Nets embody this perspective through three key features:

- **End-to-end learning:** The model learns how to form units jointly with the prediction objective, rather than in a separate preprocessing step.
- **Learning from bytes:** The model reads raw bytes directly, so its learning signal comes from the data itself rather than through a fixed tokenizer.
- **Hierarchical abstraction:** The model can recursively chunk these units to build representations at multiple levels of abstraction.

### Observed chunking behaviors

The examples below come from Cartesia's blog post on H-Nets.

#### In-context tokenization

Unlike BPE, H-Net can adapt its chunk boundaries based on the preceding context. Thus, the same word can be chunked differently depending on what came before it.

<figure>
<iframe src="/blog/post/20261116/repeated-formats.html" title="Repeated formats: H-Net stage 1 and Llama 3 BPE" width="100%" height="300" style="display: block; width: 100%; border: 0; margin: 0 auto;"></iframe>
<figcaption style="font-size: 0.8rem; line-height: 1.5; font-weight: 400; color: hsl(var(--muted-foreground, 0 0% 45%)); margin: 0.4rem 0 0; text-align: center;">Chunking repeated formats. <a style="color: inherit; font-size: inherit; font-weight: inherit; text-underline-offset: 2px;" href="https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1#fig-11">Source: Cartesia, Figure 11</a>.</figcaption>
</figure>

<figure>
<iframe src="/blog/post/20261116/digit-blocks.html" title="Digit blocks: H-Net stage 0 and Llama 3 BPE" width="100%" height="300" style="display: block; width: 100%; border: 0; margin: 0 auto;"></iframe>
<figcaption style="font-size: 0.8rem; line-height: 1.5; font-weight: 400; color: hsl(var(--muted-foreground, 0 0% 45%)); margin: 0.4rem 0 0; text-align: center;">Chunking recurring four-digit blocks. <a style="color: inherit; font-size: inherit; font-weight: inherit; text-underline-offset: 2px;" href="https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1#fig-11">Source: Cartesia, Figure 11</a>.</figcaption>
</figure>

#### Semantic chunking

H-Net tends to chunk input into semantically meaningful units. These units can cross boundaries between words, whitespace, and punctuation, as illustrated by the Python and Korean examples below.

<figure>
<iframe src="/blog/post/20261116/python-mlp.html" title="Python MLP: H-Net stage 1 and Llama 3 BPE" width="100%" height="180" style="display: block; width: 100%; border: 0; margin: 0 auto;"></iframe>
<figcaption style="font-size: 0.8rem; line-height: 1.5; font-weight: 400; color: hsl(var(--muted-foreground, 0 0% 45%)); margin: 0.4rem 0 0; text-align: center;">Chunking Python code. <a style="color: inherit; font-size: inherit; font-weight: inherit; text-underline-offset: 2px;" href="https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1#fig-10b">Source: Cartesia, Figure 10b</a>.</figcaption>
</figure>

<figure>
<iframe src="/blog/post/20261116/korean-chunks.html" title="Korean: H-Net stage 1 and Llama 3 BPE" width="100%" height="180" style="display: block; width: 100%; border: 0; margin: 0 auto;"></iframe>
<figcaption style="font-size: 0.8rem; line-height: 1.5; font-weight: 400; color: hsl(var(--muted-foreground, 0 0% 45%)); margin: 0.4rem 0 0; text-align: center;">Chunking Korean text (each character is 3 bytes). <a style="color: inherit; font-size: inherit; font-weight: inherit; text-underline-offset: 2px;" href="https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1#fig-10a">Source: Cartesia, Figure 10a</a>.</figcaption>
</figure>


## References

<ol class="references">
<li id="ref-1">

[Cartesia, *From tokens to bytes I: Scaling hierarchical networks*](https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1)

</li>
<li id="ref-2">

[Rico Sennrich, Barry Haddow, and Alexandra Birch, *Neural Machine Translation of Rare Words with Subword Units* (ACL 2016)](https://aclanthology.org/P16-1162/)

</li>
</ol>
