---
layout: post
title: Advances in Financial Machine Learning Chapter 2.
date: 2023-08-10
description: Advances in Financial Machine Learning Chapter 2.
# tags: formatting jupyter
categories: Finance, Advances in Financial Machine Learning
giscus_comments: false
related_posts: false
---
<!-- {::nomarkdown} -->
<!-- {% jupyter_notebook "test.ipynb" %} -->
<!-- {:/nomarkdown} -->
## 2.3 바 (Bar)

바는 수집한 데이터 테이블의 한 행을 말한다. 
즉, 이는 우리가 어떻게 데이터를 수집하는가에 대한 것이다.
<!-- Bar is a row of collected data table. This is about how we collect the data. -->

### 2.3.1 표준 바 (Standard Bar)
#### 2.3.1.1 시간 바 (time bar)
- 거래량이 적은 시간의 정보를 과대반영 함
- Serial correlation, heteroscedasticity, non-normality 등 통계적 성질이 좋지 않음
#### 2.3.1.2 틱 바 (tick bar)
- 수익률이 IID 정규분포에 가까움
- 많은 거래량이 단일 틱으로 기록될 수 있음 (대량 장외가 거래)
#### 2.3.1.3 거래량 바 (volume bar)
- IID 정규분포에 더욱더 가까움
#### 2.4.1.4 달러 바 (dollar bar) : 정해진 시장가치가 거래될 때마다 추출
- 틱 바와 거래량 바에 비해서 일별 바의 수가 안정적임

### 2.3.2 정보 주도 바 (Information-driven bar)
시장에 새로운 정보가 도달했을 때 추출하는 방법이다. 이는 시장 외부의 이벤트로 균형 가격에 변동이 생기고 이를 시장 내에서 포착했을 때 데이터를 추출한다.
틱의 시퀀스 $\{(p_t, v_t)\}_{t=1,2,...}$가 있다고 하자. 
이 때, signed tick $\{b_t\}_{t=1,2,...}$를 다음과 같이 정의하자.
$$
b_t := 
\begin{cases}
    b_{t-1} & \text{if } \Delta p_t = 0 \\ 
    sign(\Delta p_t) & \text{otherwise.}
\end{cases}
$$
일반적으로 $\Delta p_t>0$은 매수자에 의한 거래 체결을 의미하고 $\Delta p_t<0$은 매도자에 의한 거래 체결을 의미한다.

#### 2.3.2.1 틱 불균형바 (TIB, Tick Imbalance Bar) / 거래량 불균형 바 (VIB, Volumne Imbalance Bar) / 달러 불균형 바 (DIB, Dollar Imbalance Bar)
$$
\begin{align*}
\theta_T &:= \sum_{t=1}^T b_tv_t \\
\mathbb{E}_0[\theta_T] &:= \mathbb{E}_0[T](\mathrm{P}[b_t=1]\mathbb{E}[v_t|b_t=1]-\mathrm{P}[b_t=-1]\mathbb{E}[v_t|b_t=-1]) \\
&:= \mathbb{E}_0[T](v^+ - v^-) = \mathbb{E}_0[T](2v^+ - \mathbb{E}[v_t]) \\
T^* &:= \argmin_{T} \{|\theta_T| \geq \mathbb{E}_0[\theta_T]\}
\end{align*}
$$
이고 $\mathbb{E}_0[T]$, $v^+$와 $\mathbb{E}[v_t]$는 이전 바들로부터의 EMA 추정값을 사용한다. \
$v_t$ = 1이면 TIB, $v_t$가 거래량이나 달러이면 각각 VIB와 DIB이다.

#### 2.3.2.3 틱 런 바 (TRB, Tick Run Bar) / 거래량 런 바 (VRB, Volumne Run Bar) / 달러 런 바 (DRB, Dollar Run Bar)
불균형 바는 매수와 매도의 불균형을 감지했을 때마다 데이터를 수집한다. 
한편, 대규모 거래자들은 거래내역을 감추기 위해서 소량씩 분할하여 거래하기도하는데 이와 같은 run을 감지하기 위함이다. 
통계학에서 run은 연속된 같은 데이터를 의미한다.
$$
\begin{align*}
\theta_T &:= \max(\sum_{t|b_t=1} b_tv_t, -\sum_{t|b_t=-1} b_tv_t) \\
\mathbb{E}_0[\theta_T] &:= \mathbb{E}[T]\max(\mathrm{P}[b_t=1]\mathbb{E}[v_t|b_t=1], \mathrm{P}[b_t=-1]\mathbb{E}[v_t|b_t=-1]) \\
&= \mathbb{E}[T]\max(\mathrm{P}[b_t=1]\mathbb{E}[v_t|b_t=1], (1-\mathrm{P}[b_t=1])\mathbb{E}[v_t|b_t=-1]) \\
T^* &:= \argmin_{T} \{|\theta_T| \geq \mathbb{E}_0[\theta_T]\}
\end{align*}
$$
이고 $\mathbb{E}_0[T]$, $\mathrm{P}[b_t=1]$, $\mathbb{E}[v_t|b_t=1]$와 $\mathbb{E}[v_t|b_t=-1]$는 이전 바들로부터의 EMA 추정값을 사용한다.

##