---
layout: post
title: "Reasoning with LLMS"
date: 2025-01-05
categories: reasoning llm dl
---

# Introduction

As cliché as it may sound, LLMs incorporate the knowledge of the entire internet. This poses a significant challenge, for that also means that is becoming particularly hard to improve these models. There is hardly any text data left around (true, there are images and videos, but we've yet to figure out how to properly make use of them). For this reason, we have shifted our focus to other areas to explore. At the core of all these areas lies the fundamental idea of prompting the model to do CoT (Chain-of-Thought). Similarly to the way humans think, LLMs can also benefit from writing things down in order to condition and so, improve their "thoughts". In its most vanilla form, CoT entails prompting the model via some instructions to make it output its thinking process. This automatically improves the performance of the model solely by the virtue of better conditioning (the likelihood of the final answer - whatever it may be - increases after outputting some tokens that express some form of reasoning).

A secondary area encompasses various RL-based approaches similar to RLHF for alignment to human preferences. In particular, having a reward model that can accurately rank various CoT traces (based on some training procedure meant to imitate human rankings) could lead to some form of free-flow reasoning. During training, the base model is allowed to output whatever it may want and the reward model rewards or punishes it by virtue of a reward. The base model is trained with PPO with the respective reward signal and incentivized to produce better reasoning.

A third  area shifts the focus from training compute to inference compute. This idea has its origins in psychology, in the distinction between System 1 and System 2. Current autoregressive LLMs behave like the human System 1; it's the impulse we have when we attempt to answer something simple. It's that first answer the math teacher gives a child while multitasking. It does not require slow thinking, it's instinct. System 2 is slow and requires deliberation. Current autoregressive LLMs spend essentially the same amount of time answering any question, up to the number of output tokens. They do not exhibit the skills of a potential System 2. Inference-time compute is meant to emulate this deliberate thinking. The premise of using inference-time compute lies in the idea that LLMs already contain all the knowledge they need in order to solve a task at hand. They just do not know how to use it.

To sum up, the three approaches are:

- SFT (supervised fine-tuning with reasoning traces)
- RLHF (using a reasoning-trained reward model to train a base model with PPO to improve reasoning)
- Using a verifier (do some smart search in the space of tokens by searching against another model that can correct the potential mistakes in the output)

Let us discuss in detail these three approaches.

# SFT

Naturally, the first approach entails training a model to reproduce reasoning traces for a specific task token by token. This could mean 

# RLHF

As with Reinforcement Learning with Human Feedback for alignment to human preferences, this approach can also be used for reasoning.

# Verifier-based search

# Self-correction
