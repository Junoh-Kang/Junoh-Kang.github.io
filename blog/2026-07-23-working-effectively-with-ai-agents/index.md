---
title: Working effectively with AI agents
description: More verified, useful outcomes—for less total human attention.
publishDate: '2026-07-23'
updatedDate: '2026-10-16'
category: Tech
slides: working-with-ai-agents.html
---

## Overview

**Productivity with agents means more verified, useful outcomes for less total human attention**—not just more output. This post works through three questions:

1. **What should productivity mean?**
2. **What should we keep, and what should we delegate?**
3. **How do we work with agents in practice?**

As agents accelerate execution, the human role is shifting toward planning the task and verifying the outcome. To do both well, we should stay in the loop. My own practice comes down to three habits:

- **Don't rush into Do.** Spend more time on Plan and Verify before delegating execution.
- **Stay in the loop.** Ask the agent to quiz me, so I stay in sync at minimal attention cost.
- **Write for others.** When my reasoning feels unclear, I write something to explain or persuade other people. An agent can help structure and edit it, but I own the claims and reasoning.

---

## What is an agent?

OpenAI defines agents as "systems that independently accomplish tasks on your behalf"<sup class="cite">[<a href="#ref-1">1</a>]</sup>. Anthropic defines an agent as "an AI model that directs its own processes and tool use when accomplishing a task"<sup class="cite">[<a href="#ref-2">2</a>]</sup>. Both definitions center on the same thing: an agent **accomplishes a task**, rather than returning a single response.

Anthropic also describes an agent by its parts<sup class="cite">[<a href="#ref-2">2</a>]</sup>: **Agent = Model + Harness + Tools + Environment**. The harness is the set of instructions and guardrails around the model.

<figure>
  <img class="zoomable" src="/blog/post/20260723/diagram-agent-anatomy.svg" alt="The model operates within a harness, calls tools, and receives observations from the environment." width="1680" height="470" loading="lazy" style="width: 100%; max-width: 100%; height: auto; margin: 0; padding: 12px; border-radius: 8px; background: #fff;" />
  <figcaption>Agent = Model + Harness + Tools + Environment. Adapted from Anthropic<sup class="cite">[<a href="#ref-2">2</a>]</sup>.</figcaption>
</figure>

---

## What counts as productivity?

### Throughput is not productivity

OpenAI's harness-engineering write-up reports roughly 10× faster implementation in an agent-first internal project, but **human QA became the new bottleneck**<sup class="cite">[<a href="#ref-3">3</a>]</sup>. Faster execution moved the bottleneck to verification; it did not by itself produce more verified outcomes.

### Productivity depends on the whole loop

<figure>
  <img class="zoomable" src="/blog/post/20260723/diagram-productivity-whole-loop.svg" alt="Task N runs Plan, Do, Verify and produces a Verified Outcome, which starts Task N+1's Plan, Do, Verify." width="1680" height="190" loading="lazy" style="width: 100%; max-width: 100%; height: auto; margin: 0; padding: 12px; border-radius: 8px; background: #fff;" />
  <figcaption>A simplified model of work.</figcaption>
</figure>

**Work is a sequence of interdependent tasks.** Each task starts from the previous task's *verified outcome*, and its Plan → Do → Verify loop updates the work state into a new one. Productivity improves only when the whole cycle speeds up—not just Do, which is what agents mainly accelerate. The goal, then, is **more verified outcomes for less total human attention**, including planning, review, and recovering from mistakes.

---

## What should we keep, and what should we delegate?

### The human must maintain an updated mental model

<figure>
  <img class="zoomable" src="/blog/post/20260723/diagram-mental-model-loop.svg" alt="Verified outcomes update the work state, which must sync with human understanding to guide the next Plan and Verify." width="1680" height="530" loading="lazy" style="width: 100%; max-width: 100%; height: auto; margin: 0; padding: 12px; border-radius: 8px; background: #fff;" />
  <figcaption>A successful task yields a verified outcome, which updates the work state. Once that update syncs to the human, they can decide what the next task should be.</figcaption>
</figure>

**The human must maintain an updated mental model.** In Anthropic's study of agentic coding<sup class="cite">[<a href="#ref-4">4</a>]</sup>, domain expertise—understanding of the work—mattered in how people used agents. More expert users reached verified success more often, recovered from trouble more often, and delegated longer tasks per instruction. Their expertise showed up in three ways: concrete goals, constraints, and invariants; clear criteria for what and how to verify; and corrections of the model's domain errors.

### When the human mental model falls behind

<figure>
  <img class="zoomable" src="/blog/post/20260723/diagram-mental-model-falls-behind.svg" alt="The same loop with the sync between work state and human state broken by comprehension debt, and the human state itself eroded by cognitive debt." width="1680" height="500" loading="lazy" style="width: 100%; max-width: 100%; height: auto; margin: 0; padding: 12px; border-radius: 8px; background: #fff;" />
  <figcaption>Comprehension debt breaks the sync; cognitive debt weakens the human state itself.</figcaption>
</figure>

The human mental model can fall behind in two ways.

- **Comprehension debt**<sup class="cite">[<a href="#ref-5">5</a>]</sup> is a state-sync failure. The work changes, but my mental model does not, so the deferred attention returns later as review, rework, and recovery.
- **Cognitive debt**<sup class="cite">[<a href="#ref-6">6</a>]</sup> is capability degradation. Reasoning and debugging are delegated without engagement, so skills, recall, and verification ability weaken.

**Failure case 1: Cognitive debt.** In Anthropic's study of 52 developers learning an unfamiliar Python library<sup class="cite">[<a href="#ref-7">7</a>]</sup>, the AI-assisted group finished about two minutes faster—a difference that was not statistically significant—but scored lower than the hand-coding group on a subsequent quiz (50% versus 67%), especially on debugging questions.

**Failure case 2: Comprehension debt.** I had an agent refactor a research codebase and accepted the result once the baseline training reproduced under a fixed seed—without first understanding the original pipeline. When I later tried to change the token order, tokenization turned out to be entangled with encoding, preprocessing, and one-off scripts. I had to trace the pipeline and refactor it again, which delayed my experiments. Reproducing the baseline verified the code's behavior, not my understanding of it.

### Three gates before delegation

Before delegating a task, we ask one question: **can we frame the Plan, define how to Verify, and Reverse a failure?**<sup class="cite">[<a href="#ref-8">8</a>, <a href="#ref-9">9</a>, <a href="#ref-10">10</a>]</sup>

- **Plan gate.** Can we specify the goal, context, constraints, and scope—without prescribing every execution step?
- **Verify gate.** Can we define success before execution and check the outcome independently, rather than trusting the agent's claim?
- **Reversibility gate.** Can we restore the previous state, and afford the recovery cost if the task fails?

When a gate is blocked, we either strengthen it and re-evaluate, or reduce the agent's role.

<figure>
  <figcaption>A blocked gate does not mean no AI. It means less agency.</figcaption>

| Blocked gate | Strengthen | Reduce |
|---|---|---|
| **Plan** | Clarify and reframe the intent. | Narrow the scope. |
| **Verify** | Add repeatable, goal-aligned criteria. | The human owns acceptance. |
| **Reversibility** | Add a sandbox or dry run. | The agent prepares; the human acts. |

</figure>

---

## Working with agents in practice

### Protect the bio-token budget

I call human attention *bio-tokens*: the human equivalent of AI tokens. Unlike AI tokens, bio-tokens cannot be stored, and only time refills them.

Three things drain it quickly:

- **Reading.** Low-readability output costs more, and other people's code and writing cost the most.
- **Switching.** Working memory is small<sup class="cite">[<a href="#ref-11">11</a>]</sup>, and task switching has measurable costs<sup class="cite">[<a href="#ref-12">12</a>]</sup>. OpenAI reports that most of its engineers ran three to five concurrent Codex sessions before it started to hurt<sup class="cite">[<a href="#ref-13">13</a>]</sup>.
- **Generation.** Answering an open-ended question costs more than choosing among concrete options.

The practices below are largely about spending this budget well.

### Plan gate: externalize the work state

```text
docs/
├── GOAL.md          goal and non-goals
├── STATE.md         current state, completed work
├── DECISIONS.md     key decisions, open questions
├── plan/            implementation plans and history
│   └── implementation-001.md
└── experiment/      results, evidence, uncertainty
    └── experiment-001.md
```

Structure the project so that the agent can collect context from accumulated plans, decisions, and experiment records<sup class="cite">[<a href="#ref-14">14</a>]</sup>. The agent can write and update these records; we decide what to record and how, and check that the records stay accurate.

### Plan gate: make intent explicit

<figure>
  <blockquote><p>A lot of times, people don't know what they want until you show it to them.</p></blockquote>
  <figcaption>— Steve Jobs, <em>BusinessWeek</em> (1998)<sup class="cite">[<a href="#ref-15">15</a>]</sup></figcaption>
</figure>

The same holds when we delegate to an agent: we often do not know exactly what we want until we see a result, so we may not even know what to ask for. **Rather than writing a complete brief up front, I let the agent draw the intent out of me.**

A standard planning flow asks a few questions, drafts a plan, and leaves me to review it open-endedly. I then have to discover what is missing or wrong, and explain it. I prefer an interview pattern, such as the [grill-me skill](https://github.com/mattpocock/skills/blob/66898f6/skills/productivity/grill-me/SKILL.md): the agent asks **one decision question at a time, each with a recommended answer**, and I accept or correct each one until we are aligned.

This turns generation into selection, which costs fewer bio-tokens. It also leaves explicit decisions behind, which makes it clearer what should be verified.

### Verify gate: define it first, read less

Define verification before delegating Do, as in test-driven development. Good verification is<sup class="cite">[<a href="#ref-9">9</a>]</sup>:

- **Outcome-based:** it checks the actual result, not the agent's claim.
- **Repeatable:** the same condition gives the same result.
- **Goal-aligned:** passing it is evidence for the actual goal.
- **Failure-sensitive:** it fails for plausible wrong outcomes.

Verification does not require rereading the whole implementation:

1. **Automated checks first.** Let tests, builds, and output comparisons screen out concrete failures.
2. **Compact review.** Ask the agent to present the rest: what changed, what evidence was collected, and what remains uncertain.
3. **Quiz.** When we need to check our own understanding, ask the agent to quiz us on the result.

### Recover from failure

When a result misses the goal, a simple sequence helps:

1. **Revert and redo once.** This is cheap when the work is reversible, and because agent outputs are non-deterministic, a second attempt may succeed.
2. **If it fails again, stop repeating the same attempt.** The same input and conditions will not produce a more reliable process.
3. **Change what caused the failure.** If execution departed from a sound plan, write a failure report and repair execution. If the plan itself was wrong, re-plan and redo. If the problem keeps recurring, update the context, rules, skills, or tests.

---

## References

<ol class="references">
<li id="ref-1">

[OpenAI, *A practical guide to building agents* (2025)](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)

</li>
<li id="ref-2">

[Anthropic, *Trustworthy agents in practice* (Apr 2026)](https://www.anthropic.com/research/trustworthy-agents)

</li>
<li id="ref-3">

[OpenAI, *Harness engineering* (Feb 2026)](https://openai.com/index/harness-engineering/)

</li>
<li id="ref-4">

[Anthropic, *Agentic coding and persistent returns to expertise* (Jun 2026)](https://www.anthropic.com/research/claude-code-expertise)

</li>
<li id="ref-5">

[Addy Osmani, *Comprehension debt: the hidden cost of AI-generated code* (Mar 2026)](https://medium.com/@addyosmani/comprehension-debt-the-hidden-cost-of-ai-generated-code-285a25dac57e)

</li>
<li id="ref-6">

[Nataliya Kosmyna et al., *Your Brain on ChatGPT* (arXiv:2506.08872, 2025)](https://arxiv.org/abs/2506.08872)

</li>
<li id="ref-7">

[Anthropic, *How AI assistance impacts the formation of coding skills* (Jan 2026)](https://www.anthropic.com/research/AI-assistance-coding-skills)

</li>
<li id="ref-8">

[OpenAI Academy, *Workspace agents* (Apr 2026)](https://openai.com/academy/workspace-agents/)

</li>
<li id="ref-9">

[Anthropic, *Demystifying evals for AI agents* (Jan 2026)](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)

</li>
<li id="ref-10">

[Anthropic, *Measuring agent autonomy* (Feb 2026)](https://www.anthropic.com/research/measuring-agent-autonomy)

</li>
<li id="ref-11">

[Nelson Cowan, *The magical number 4 in short-term memory* (Behavioral and Brain Sciences, 2001)](https://pubmed.ncbi.nlm.nih.gov/11515286/)

</li>
<li id="ref-12">

[Joshua S. Rubinstein, David E. Meyer, and Jeffrey E. Evans, *Executive control of cognitive processes in task switching* (Journal of Experimental Psychology: Human Perception and Performance, 2001)](https://pubmed.ncbi.nlm.nih.gov/11518143/)

</li>
<li id="ref-13">

[OpenAI, *Codex orchestration* (Apr 2026)](https://openai.com/index/open-source-codex-orchestration-symphony/)

</li>
<li id="ref-14">

[Anthropic, *Effective context engineering for AI agents* (Sep 2025)](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

</li>
<li id="ref-15">

[Steve Jobs, interview in *BusinessWeek* (1998), quoted in WIRED](https://www.wired.com/story/unit-109/)

</li>
</ol>
