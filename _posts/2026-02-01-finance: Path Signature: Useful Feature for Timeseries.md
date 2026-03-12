---
layout: distill
title: "Path Signature: Useful Feature for Timeseries"
date: 2026-02-01
description: "Study note on path signature in the literature of rough path theory. This post introduces the definition, algebraic structure, and probabilistic interpretation of signatures, bridging rough path theory and modern machine learning, based primarily on *A Primer on the Signature Method in Machine Learning*." 
categories: deep-learning, timeseries, finance # deep-learning, finance, 
tags: study-notes # paper-review, survey, 
attachments: 

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


### (Motivation) Random Variable: Polynomial Features

#### Definition (Words / Multi-Indices)
A word (or multi-index) is a finite sequence $I:=i_1 \ldots i_k$ with $i_1,\ldots,i_k \in \{1,\ldots,d\}$. The length of $I$ is $|I|:=k$. The set $\{1,\ldots,d\}$ is the alphabet. Denote $\mathcal{W}$ by the set of all words.

#### Definition (Polynomials for RV)
For a vector $x=(x^1,\ldots,x^d)$ and a multi-index $I=(i_1,\ldots,i_k)$, define $x^{I}=x^{(i_1,\ldots,i_k)}:=x^{i_1}\ldots x^{i_k}$. The collection $\{x^I\}_{I\in \mathcal{W}}$ is called the polynomials of $x$.

#### Theorem (Stone-Weierstrass, Carleman)
Under suitable integrability conditions on $\mathbb{P}$, $$\mathbb{E}_{X \sim \mathbb{P}}[X^I] = \mathbb{E}_{X \sim \mathbb{Q}}[Y^I] ~~~\forall I \in \mathcal{W} ~~~~~\Leftrightarrow~~~~~ \mathbb{P} = \mathbb{Q}$$

---

### Paths: Iterated Integrals and Signature

#### Definition (Path-space)
Let $\mathcal{P}([a,b], \mathbb{R}^d)$ denote the space of continuous bounded variation paths $$X=(X^1,\ldots,X^d):[a,b] \rightarrow \mathbb{R}^d.$$

#### Definition (Iterated integrals)
For $X \in \mathcal{P}([a,b], \mathbb{R}^d)$ and word $I=i_1 \ldots i_k$, define the iterated integral $S(X)_{a,\cdot}^I \in \mathcal{P}([a,b], \mathbb{R})$ inductively by 

$$S(X)_{a,t}^{I}=\int_a^t S(X)_{a,s}^{i_1 \ldots i_{k-1}} dX_s^{i_k},$$

with base case $k=0$ as $S(X)_{a,t}^{\phi}=1$. In other expression, 

$$S(X)_{a,t}^{I} = \int_{a<t_1<\ldots<t_k<t} dX_{t_1}^{i_1} \ldots dX_{t_k}^{i_k}.$$

#### Definition (Signature)
The signature $S(X)_{a,b}$ of $X \in \mathcal{P}([a,b], \mathbb{R}^d)$ is the collection of real numbers indexed by words $$\{ S(X)_{a,b}^I\}_{I\in\mathcal{W}}=(S(X)_{a,b}^{\phi},S(X)_{a,b}^1,\ldots,S(X)_{a,b}^d,S(X)_{a,b}^{11}, \ldots S(X)_{a,b}^{dd},\ldots).$$##### Definition 
$$ S(X)_{a,b}^{(k)} = \{ S(X)_{a,b}^I \}_{|I|=k}$$

#### Exercise (Linear path)
Let $x \in \mathbb{R}^d$ and define $X(t) = tx$ for $t\in[0,1]$. Then $$S(X)_{0,1}^{i_1 \ldots i_k} = \frac{x^{i_1}\ldots x_{i_k}}{k!}.$$ sol) 

$$ \begin{align*} 
S(X)_{0,1}^{i_1 \ldots i_k} 
&= \int_{0}^{1} S(X)_{0,t}^{i_1 \ldots i_{k-1}}dX_t^{i_k} \\
 &= \int_0^1 \frac{x^{i_1}\ldots x_{i_{k-1}}}{(k-1)!}t^{k-1}x^{i_k} dt ~~(\because induction) \\
 &= \frac{x^{i_1}\ldots x_{i_k}}{k!} 
\end{align*}_\square$$

##### Exercise (1D path)
> For $X \in \mathcal{P}([a,b],\mathbb{R})$, $$S(X)_{a,b}^{1 \ldots 1} = \frac{(X_b-X_a)^k}{k!}.$$sol) $$\begin{align*} S(X)_{a,b}^{1^{\otimes k}} 
&= \int_a^b S(X)_{a,t}^{1^{\otimes {k-1}}} dX_t \\
&= \int_a^b \frac{(X_t - X_a)^{k-1}}{(k-1)!} \dot{X}_t dt \\
&= [\frac{(X_t - X_a)^{k}}{k!}]_a^t 
= \frac{(X_b - X_a)^{k}}{k!}
\end{align*}_\square $$
