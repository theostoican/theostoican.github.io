---
layout: post
title: "Folding space: why depth beats width in neural networks"
date: 2026-11-21
categories: neural-networks deep-learning
---

There's a tempting one-liner about why neural networks work: *they transform inputs into a space where classes become linearly separable.* It's not wrong. But it satisfies you for about ten seconds before you start asking follow-up questions — because it doesn't say what the transformation looks like, why it has to be nonlinear, or what a "layer" is actually doing geometrically.

I think the cleanest way to understand neural networks is as **machines that fold space**. Each layer is a fold. The folds aren't arbitrary — they're chosen so that, after enough of them, the question you care about becomes trivial. In this post I'll build up to that picture through the smallest problem that breaks linear classifiers, the smallest fix that makes it work, and then the surprising fact that depth isn't about expressive power at all — it's about how cheaply a network can express structured functions.

# XOR: the smallest hard problem

Take four points in the plane: $$(0,0)$$, $$(1,0)$$, $$(0,1)$$, $$(1,1)$$. Label the diagonal pairs as classes: $$(0,0)$$ and $$(1,1)$$ are red, $$(1,0)$$ and $$(0,1)$$ are blue. No straight line in the plane separates them — whichever line you draw, one diagonal pair gets split across it.

<div style="text-align: center;">
{% include figure.html path="assets/img/folding-space/xor-input-space.svg" class="img-fluid rounded z-depth-1" %}
</div>

Minsky and Papert proved this in 1969, and it stalled neural-network research for a decade. You can see the obstruction by trying: every orientation of a straight line puts at least one point in the wrong class.

<div style="text-align: center;">
{% include figure.html path="assets/img/folding-space/xor-lines-fail.svg" class="img-fluid rounded z-depth-1" %}
</div>

The reason it's hard is actually a one-dimensional fact in disguise. If you project the four points onto any line in the plane — call the projection $$s = w_1 x_1 + w_2 x_2$$ — the two reds always end up at the extremes of the four projected values, and the two blues end up in the middle. The two reds are the corners with sums $$0$$ and $$w_1 + w_2$$; the two blues are $$w_1$$ and $$w_2$$, which always lie between.

<div style="text-align: center;">
{% include figure.html path="assets/img/folding-space/xor-projection.svg" class="img-fluid rounded z-depth-1" %}
</div>

So no monotonic function of $$s$$ can separate them. A sigmoid, a ReLU, a tanh — they all preserve the order of their inputs. The reds stay at the extremes, the blues stay in the middle. You cannot pick a threshold that puts both reds on one side and both blues on the other.

# Breaking the symmetry: two ways out

There are two escape hatches, and they correspond to the two superpowers neural networks use.

**Option one: use a non-monotonic nonlinearity.** Squaring works. The map $$\phi(x_1, x_2) = (x_1 + x_2 - 1)^2$$ sends both blues to $$0$$ and both reds to $$1$$. Geometrically, the line $$x_1 + x_2 = 1$$ passes exactly through the two blue points; squaring measures how far you are from that line, and the reds sit at the same distance on opposite sides.

<div style="text-align: center;">
{% include figure.html path="assets/img/folding-space/xor-squaring-fold.svg" class="img-fluid rounded z-depth-1" %}
</div>

This works, but real networks don't use squaring. Sigmoids, ReLUs, and tanh are all monotonic. So how do they handle XOR?

**Option two: use more than one neuron.** Two sigmoid neurons can build features that say "did we pass threshold A?" and "did we pass threshold B?". Set one threshold low (acting like OR) and one high (acting like AND). In the two-dimensional $$(h_1, h_2)$$ feature space:

- $$(0,0)$$ → neither threshold passed → $$(0, 0)$$
- $$(1,0)$$ and $$(0,1)$$ → only the low threshold → $$(1, 0)$$
- $$(1,1)$$ → both thresholds → $$(1, 1)$$

The two blue points collapse onto each other, and the line $$h_1 - h_2 = 0.5$$ — "exactly one threshold passed" — separates the classes. That's XOR by definition.

<div style="text-align: center;">
{% include figure.html path="assets/img/folding-space/xor-two-sigmoids.svg" class="img-fluid rounded z-depth-1" %}
</div>

Both options do the same thing in spirit: they break the order-preserving property of a single monotonic projection. Option one uses a nonlinearity that folds; option two uses two projections that are allowed to disagree. Hidden layers fixed XOR by *learning* a feature — like $$x_1 x_2$$ or the OR-AND pair — rather than requiring us to hand-engineer it.

# From XOR to the real question: does depth give you new power?

A natural reading of the XOR story is: "more layers let a network do things fewer layers can't." But this is where it's easy to go wrong.

The universal approximation theorem (Cybenko 1989, Hornik 1991) says that **a single hidden layer of sufficient width can approximate any continuous function on a compact set**. That includes any classifier you care about, any wild decision boundary, any Boolean combination of regions. One hidden layer is already universal.

So depth doesn't add expressive power. What does it add?

**Efficiency.** For a large and important class of functions, a deep network can represent the function with exponentially fewer neurons than any single-hidden-layer network. This is a provable, quantitative gap — and the cleanest witness for it is a function so simple you can draw it by hand.

# The sawtooth: an exponential separation on one page

Consider a sawtooth function on $$[0, 1]$$ with $$2^L$$ triangular teeth. It goes up, down, up, down, $$2^L$$ times.

**A deep ReLU network builds this sawtooth using $$2L$$ neurons. A single-hidden-layer network needs $$\sim 2^L$$ neurons to match it.**

For $$L = 10$$: 20 neurons versus 2048. The deep network has twenty units; the shallow one needs thousands. Same function, exponential gap. This is Telgarsky's 2016 depth-separation result, and the sawtooth is the simplest function anyone has found that proves it rigorously.

The construction is almost comically simple. Define the "triangle wave":

$$g(x) = 2 \cdot \mathrm{ReLU}(x) - 4 \cdot \mathrm{ReLU}(x - 0.5)$$

Check it: on $$[0, 0.5]$$, $$g(x) = 2x$$, rising from $$0$$ to $$1$$. On $$[0.5, 1]$$, $$g(x) = -2x + 2$$, falling from $$1$$ to $$0$$. That's a single tent, built from two ReLUs.

Now compose. $$g(g(x))$$ has 2 teeth. $$g(g(g(x)))$$ has 4. $$g$$ composed with itself $$L$$ times has $$2^L$$ teeth. Each composition is one extra layer, which uses 2 ReLUs — hence $$2L$$ neurons for $$2^L$$ teeth.

# Why composition gives exponential growth: the paper-fold picture

Here's the intuition that finally made this click for me.

$$g$$ is equivalent to **folding a paper strip in half**. I mean this literally. Take a strip of paper marked $$0$$ to $$1$$. Fold the right end onto the left. The point at $$x = 0.75$$ now sits physically on top of the point at $$x = 0.25$$. In math, $$g$$ does the same thing: $$g(0.25) = g(0.75) = 0.5$$. Both inputs produce the same output. The function has glued them together.

Every tent-shaped function is a fold, because a tent maps two inputs — $$x$$ and $$1 - x$$ — to the same output. That's the signature of a fold expressed as a function. The factor of 2 in $$g$$'s formula is just rescaling so that the output range matches the input range, which is what lets us compose $$g$$ with itself.

Now here's the magic:

1. Take a paper strip $$[0, 1]$$. Fold it in half. The strip is now two layers thick, covering $$[0, 0.5]$$ physically.
2. Fold the folded strip in half again. Four layers thick.
3. Fold again. Eight layers thick.

After $$L$$ folds, the strip is $$2^L$$ layers thick. Now take a pen and draw a single triangle on the top face. The ink soaks through all $$2^L$$ layers. Unfold the paper, and you see $$2^L$$ triangles along the strip.

**One pen stroke. $$2^L$$ triangles.**

That's what the deep network is doing. Each layer is a fold — the same simple operation, applied again. But because every fold operates on the already-folded paper, not on the original flat strip, a single action by layer $$k$$ corresponds to $$2^{k-1}$$ actions in the original coordinate.

A wide one-layer network has no folded paper to work with. It only ever sees the raw input. So if you want $$2^L$$ teeth, you need $$2^L$$ separate pen strokes — one per tooth, each drawn by its own neuron. No reuse, no doubling, no exponential payoff.

# Why composition needs nonlinearity

This matters, because it's easy to miss. If $$g$$ were linear — say $$g(x) = ax + b$$ — then composing $$g$$ with itself just gives another linear function:

$$g(g(x)) = a(ax + b) + b = a^2 x + ab + b$$

Still linear. No matter how many times you compose, you get a linear function. Composition buys you nothing in the linear world. This is why pure matrix multiplication doesn't benefit from depth: $$A \cdot B \cdot C \cdot x$$ collapses to $$(ABC) \cdot x$$ — all the depth disappears into a single matrix.

The nonlinearity is what makes composition non-trivial. When $$g$$ has a kink in it, $$g(g(x))$$ has more kinks than $$g(x)$$, because the outer $$g$$'s kink gets placed at every preimage point of the inner $$g$$. Nonlinearity makes kinks multiply rather than collapse. **Composition + nonlinearity = exponential gain. Either one alone gives you nothing.**

# What the general picture looks like

ReLU networks compute piecewise-linear functions. Exactly. So does the sawtooth. So does any thresholded classifier boundary. The universal approximation theorem for ReLU networks is just the fact that piecewise-linear functions are dense in the continuous functions: cut the domain into small pieces, fit a line on each piece, and as the pieces get smaller the approximation gets better.

The question "how good is my network?" becomes "how many linear pieces can it produce, and are they placed well?" And here's where depth earns its keep:

- A 1-hidden-layer ReLU network needs $$\Omega(K)$$ neurons to produce $$K$$ linear pieces — one neuron per kink.
- A deep ReLU network can produce $$K$$ pieces with $$O(\log K)$$ neurons per layer and $$O(\log K)$$ layers, *if* the pieces have a structure that composition exploits — self-similarity, hierarchy, periodicity.

Montúfar, Pascanu, Cho and Bengio (2014) showed the general version: depth-$$L$$ ReLU networks carve input space into a number of linear regions that grows exponentially with $$L$$, versus polynomially with width. Depth isn't more expressive than width; it's exponentially more economical when the function being learned has structure.

# Why this matters beyond toy sawtooths

Real data has repeating and hierarchical structure at many scales. An edge detector in an image is useful at every position. A syntactic pattern in a sentence is useful at every token. A face is made of parts, which are made of textures, which are made of edges.

Deep architectures in practice exploit this in three main ways:

- **CNNs** make spatial repetition explicit through weight sharing — a hard-coded version of the "fold once, reuse everywhere" trick.
- **RNNs and Transformers** make sequence repetition explicit through parameter sharing across time or position.
- **Deep feedforward networks** exploit structure implicitly through composition, the way the sawtooth does.

In all three, depth is how the network gets to say "build the feature once, apply it everywhere." The sawtooth is the cleanest 1D version of that idea — and the one that lets you prove, rather than just gesture at, the exponential gap between depth and width.

# The one-sentence takeaway

Each layer is a fold. One fold, applied to an already-folded stack, does the work of many folds on the original paper — because the paper underneath is already $$2^{k-1}$$ layers thick. That's the entire mechanism. Everything else — the exponential tooth count, the depth-vs-width separation, the compositional efficiency of real deep networks — is a consequence of this one fact.

Depth doesn't add what a network can do. It adds how cheaply the network can do it, by letting each layer act on the structured output of previous layers rather than on the raw input. For problems with repeating or self-similar structure — which is most problems worth solving — this gives exponential savings.

---

**References**

- Minsky and Papert, *Perceptrons* (1969)
- Cybenko, "Approximation by superpositions of a sigmoidal function" (1989)
- Hornik, "Approximation capabilities of multilayer feedforward networks" (1991)
- Montúfar, Pascanu, Cho, Bengio, "On the Number of Linear Regions of Deep Neural Networks" (2014)
- Telgarsky, "Benefits of depth in neural networks" (2016)
