---
layout: post
title: "Reasoning with LLMS"
date: 2025-01-05
categories: reasoning llm dl
---

# Introduction

As cliché as it may sound, LLMs incorporate the knowledge of the entire internet. This poses a significant challenge, for that also means that is becoming particularly hard to improve these models. There is hardly any text data left around (true, there are images and videos, but we've yet to figure out how to properly make use of them). For this reason, we have shifted our focus to other areas to explore. One of these new areas shifts the focus from training compute to inference compute. This idea has its origins in psychology, in the distinction between System 1 and System 2.

Current autoregressive LLMs behave like the human System 1; it's the impulse we have when we attempt to answer something simple. It's that first answer the math teacher gives a child while multitasking. It does not require slow thinking, it's instinct.  System 2 is slow and requires deliberation. Current autoregressive LLMs spend essentially the same amount of time answering any question, up to the number of output tokens. They do not exhibit the skills of a potential System 2. Inference-time compute is meant to emulate this deliberate thinking.


# How to make use of inference-time compute?

The premise of using inference-time compute lies in the idea that LLMs already contain all the knowledge they need in order to solve a task at hand. They just do not know how to use it. Thus, we could exploit this knowledge in two ways:

- incentivizing the LLMs to rethink their solution
- do some smart search in the space of tokens by searching against another model that can correct the potential mistakes in the output

Let us discuss in detail these two approaches.

# Verifier-based search

# Self-correction
