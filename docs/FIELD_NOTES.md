---
title: "Buzzwords & Misdemeanors: one 1B model runs a whole courtroom"
thumbnail: /blog/assets/buzzwords/thumbnail.png
authors:
  - user: BastienHot
---

# Buzzwords & Misdemeanors: one 1B model runs a whole courtroom

You wake up in a courtroom. A judge, a prosecutor, and a defense lawyer are tearing into
your case, and every word out of their mouths is dense aviation lingo: flight plans,
go-arounds, a scrubbed black box, a deviation nobody briefed. There is only one problem.
You are not a pilot. The jargon is a smokescreen you picked yourself at the start of the
game, and what you are actually accused of has nothing to do with aviation. Your job is to
listen past the buzzwords, work out your real profession and the real charge, and enter a
plea. A model then scores how close you got and reveals the truth.

We built Buzzwords & Misdemeanors for the Hugging Face Build Small hackathon, under three
self-imposed constraints. Every model had to be genuinely small, comfortably under 32B. The
whole game had to run locally, with no cloud inference in the loop. And all of the text had
to pass through the `llama.cpp` runtime. By the end we had pushed those constraints further
than we expected: the *entire* game — the director that writes and runs the trial, and the
three actors who argue it — is a **single 1B model**, MiniCPM5-1B, wearing different LoRA
adapters, running on a **free two-vCPU CPU**. No GPU anywhere at play time.

This post is the honest version of how we got there. The design and the distillation are the
parts we are proud of; the part we think is genuinely worth reading is the long, embarrassing
detour we took *evaluating* a grammar-constrained orchestrator, and the methodology we landed
on after being fooled by our own benchmarks three times in a row.

<!-- 📷 Figure slot: a gameplay screenshot (the courtroom stage with a speech bubble). -->

## The idea: a director and its actors

The obvious design is to let one model improvise the entire trial. With small models that
collapses quickly. They wander off topic, they forget who is speaking, and worst of all they
blurt out the answer the player is supposed to guess. So instead of one improviser we
borrowed the structure of a film set: a director and a cast.

A Game Master owns the truth and the pacing. From a seed it writes a hidden Case File — your
real profession, the fault in plain English, and a handful of oblique facts. After that it
never speaks in character. Every turn it makes one small decision about who should talk next
and how. The three actors — judge, prosecutor, defense — deliver the lines. They only ever
see the Game Master's stage direction, never the plain truth, which means they physically
cannot leak the answer even when they get dramatic.

The trick that makes this work on small models is to stop the Game Master from writing prose
at all. A 1B-class model invents badly but classifies well, so we constrain its every move to
a tiny JSON object through a `llama.cpp` GBNF grammar. "Direct the scene" becomes "pick from a
finite deck," which is something a small model is reliably good at:

```gbnf
root ::= "{" ws "\"next_speaker\":" ws speaker ws "," ws "\"beat_type\":" ws beat ws "," ws "\"intensity\":" ws intensity ws "," ws "\"stage_direction\":" ws string ws "," ws "\"wrap_up\":" ws bool ws "}"
speaker ::= "\"judge\"" | "\"prosecutor\"" | "\"defense\""
```

The smokescreen is the part that turns a text generator into an actual game. Because the
jargon has nothing to do with the hidden profession, you cannot pattern-match your way out.
You have to decode metaphor. When the prosecutor accuses the defendant of "scrubbing the
black box to mask a deviation from the flight plan," that is how an aviation-flavored
courtroom describes a museum curator who quietly altered a record.

## The architecture: one base, two kinds of hats

At play time exactly one model is ever loaded: **MiniCPM5-1B**, a clean `LlamaForCausalLM`
checkpoint, in a 4-bit GGUF. Everything else is a small adapter on top of it.

- The **Game Master** is that base plus a single **director LoRA** — one multitask adapter
  that writes the Case File, makes the per-turn beat decisions, and scores the final guess.
- The **cast** is the same base plus one **per-style actor LoRA** for the chosen jargon. The
  judge, the prosecutor, and the defense are the same base and the same adapter; only the
  system prompt changes between them.

So the whole game is *one ~1B base plus two ~90 MB adapters*, swapped by reference. Adding a
role costs nothing because it is a prompt; adding a jargon costs one adapter; and the
director — the hardest job in the game — costs one more. We were careful about a thing that
is easy to get wrong here: there is not a separate base per style, and there is no longer a
separate, larger model for the director. It is all the same 1B.

### How the director got into the base

That last point is the headline, and it is not where we started. Our first design used a
**4B** model as the Game Master — run vanilla behind the grammar — and reserved the 1B for
the actors. On paper that is reasonable: directing is the reasoning-heavy job, so give it the
bigger brain. In practice it was the single thing standing between us and a free-tier demo,
for a reason that only showed up when we benchmarked honestly.

We were determined to keep text inference off the GPU (it is what earns the off-grid angle),
so the real question was: how fast is a small model through `llama.cpp` on a free **two-vCPU**
CPU? The answer for the 1B actor was "fine." The answer for the 4B director was brutal:
roughly **1 token/second**, about **four minutes** to emit a single 256-token decision. A
twelve-turn trial would have taken over an hour. The 4B was simply not viable on the hardware
we wanted to ship on, and no amount of prompt engineering fixes a throughput wall.

Two options followed. Keep the 4B and pay for a GPU — which we had already ruled out — or
*teach the 1B to direct*. We chose the second, and most of the interesting work in the
project lives in that decision: distilling the director's whole job down into a LoRA on the
same 1B the actors already use. When it worked, the architecture collapsed from "two models
on a GPU" to "one model on a CPU," and the demo became something a stranger could actually
run for free.

## Making a 1B fast enough on a 2-vCPU CPU

Before any of that was worth attempting we had to establish that a 1B could be *fast* on a
free CPU, not just present. Three findings did the heavy lifting, and all three were invisible
until we measured them on the real hardware.

**The build matters more than the model.** Our first CPU benchmark, using a stock prebuilt
`llama-cpp-python` wheel, reported a prompt-eval rate of **1.8 tokens/second** — and, bizarrely,
that was *slower* than generation, which is backwards. The tell was clear: the wheel had no
SIMD. Rebuilding `llama.cpp` with AVX2/FMA from source took prompt-eval to **60–80 tok/s** and
generation to **~12 tok/s** on the same two-vCPU box — a 30–40× swing on prompt processing from
nothing but compile flags. This is why the deployed Space is a **Docker** Space: not for the UI,
which is still Gradio, but to control the build so the AVX2 compile is baked in.

**Thinking mode is a trap.** MiniCPM5 is a hybrid-reasoning model with a `<think>` channel.
Left on, it spends the entire token budget reasoning and returns an *empty* answer — the case
file comes back blank, the actor's line never arrives. The fix is to disable thinking, and for
the grammar-constrained director it is free: a grammar whose `root ::= "{"` forces the first
generated token to be `{`, which makes a thinking preamble structurally impossible. Turn
thinking off and the actor settles its line in ~24 tokens and the director its JSON in ~80 —
seconds, not minutes.

**The transcript is the expensive part, so cache it.** A courtroom transcript grows every turn,
and re-evaluating a cold 2,600-token context costs ~42 seconds at CPU prompt-eval rates. Running
the loop through `llama-server` with prompt caching, the second turn re-evaluates only the new
tokens — about **80× cheaper**. Without it, every turn re-eats the whole growing transcript;
with it, per-turn latency stays flat. None of this is novel, but on a 2-vCPU box it is the
difference between a playable game and a slideshow.

## Teaching a 1B to direct

The approach is distillation, the same muscle we already used for the actors: a large teacher
writes the training data, and the 1B student learns from it. Our teacher is Gemma 4 31B-it in
FP8, running offline on Modal through vLLM. The student is the MiniCPM5-1B base, trained with
4-bit QLoRA (rank 32) into one multitask **director** adapter — case file, beat decisions, and
scoring all in a single LoRA, because they share one "case world-model" and a 1B has little
capacity to spare on three separate ones.

The most important design choice is that **the smokescreen is injected by construction, not
hoped for.** A weaker pipeline asks the teacher to invent a profession unrelated to the jargon
and trusts it to comply. We instead draw the profession from a fixed pool with the jargon's own
domain *excluded*, so every training target is provably profession-⟂-jargon. The teacher only
writes the oblique phrasing and the directing around a truth we already control. This is exactly
the lesson the actors taught us a stage earlier — inject the hidden facts, do not pray for them.

Getting the *content* clean took a tighter loop than we expected, and it is worth being concrete
because the numbers moved a lot:

- Our first smoke test rejected **50%** of generated games — every rejection was the teacher
  leaking the profession word into a "clue." Three changes fixed it: a small bank of hand-written
  exemplars that we **sample from** per prompt (so the model learns the *pattern* without copying
  a fixed example), an explicit banned-word list, and a corrective **retry** that re-prompts any
  leak. Reject rate went **50% → 0%**, and held at zero when we scaled to 300 games (~4,700
  training examples).
- We also discovered a subtle prompt bug in our own runtime: `fault_plain` was being written as a
  full sentence ("The defendant skipped a check"), but the director's brief template reads "a
  plumber who `{fault_plain}`," producing "a plumber who The defendant skipped a check." We
  switched the field to a **verb phrase** in both the data and the runtime prompt — a one-line fix
  that the binary metrics would never have flagged, and that only reading the raw output revealed.

Trained two epochs, the director reached a held-out perplexity around **2.0**, comparable to the
actor adapters. But perplexity is not coherence, and confirming that the model could actually
*run a trial* is where the real story is — see the next section.

## The actor pipeline (and a mismatch we had been shipping)

The actors are trained as a two-stage curriculum, a fork rather than a merge. Stage one trains a
single LoRA on a generic courtroom dataset to learn the judicial register (`legal_base`); stage
two loads that adapter as the *starting weights*, resets the optimizer, and runs a fresh pass per
style. The output is one `actor_<style>` adapter per jargon, each a delta on the same untouched
base. Diversity comes from per-sample seeded parameters — profession, fault, tone, severity, and
a seeded subset of a curated ~80-term jargon bank injected into every prompt — so a small run is
reproducible and the teacher pulls from the whole vocabulary instead of recycling five phrases.

While wiring the new director we caught a train/runtime mismatch we had quietly been shipping in
the *actor* pipeline. The actor examples had been built with the hidden brief in the system prompt
and the previous line as the user turn — but at runtime the actor receives no brief and a *stage
direction* instead. We rebuilt the actor data to match the runtime call shape exactly (role+style
system, "Stage direction: … / Intensity: …" user, no brief), which also meant teaching the teacher
to emit a stage direction per line. The lesson, again: the training distribution has to match the
inference distribution, or the model is being graded on a test it never sat.

## How to (not) evaluate a grammar-constrained orchestrator

This is the section we would actually send to someone building something similar, because we got
it wrong three times and the wrong answers were *convincing*.

For the actors, evaluation was easy and we knew why: the jargon banks are a free, objective
yardstick. Feed the controlled runtime prompts, measure term coverage and density base-vs-LoRA
over many independent samples, check the adapter peaks on its own bank. Done. Every one of those
properties — controlled in-distribution prompts, an objective label-free metric, many independent
samples, a base-vs-LoRA contrast — turned out to matter for the director too, and our first
attempts violated them.

**First fooling: a single greedy game.** Our instinct was to *play* a full trial with the director
and read the trace. It looked broken: the model never once handed a turn to the defense, and it
produced an incoherent case file. We nearly went back to retrain. Two tells saved us. The case
file came out *byte-identical* across runs — meaning the harness was effectively greedy, judging a
single trajectory, not the model's distribution. And that trajectory contained `prosecutor →
prosecutor` transitions, which we would later learn occur about **1%** of the time in the training
data. The model was not following its training; the harness was.

**Second fooling: out-of-distribution context.** A statistical eval over many sampled prompts still
showed the defense barely appearing — but those prompts fed the director a *hand-written* transcript
and turn positions skewed toward opening/closing beats. The director conditions heavily on the
transcript, and we were handing it courtroom contexts unlike anything it trained on. We were grading
a chef on a dish that was never on the menu.

**The check that settled it: audit the data, then test in-distribution.** Before touching the model
we reconstructed the per-game speaker structure from the training set. It was not unbalanced at all —
*every* game contained the defense, the rhythm was a clean judge → prosecutor → defense, and the
`prosecutor → defense` response transition sat at **0.73**. The model had been taught the right thing.
So we built the benchmark the data deserved: a fresh, disjoint-seed **held-out** slice of teacher
games, whose *real* transcript contexts became the prompts, with the teacher's own next-speaker choice
as the reference label — the director's equivalent of the jargon banks. Run base-vs-LoRA over 180 such
contexts, no grammar (so we measure what the model *learned*, since the runtime grammar guarantees the
JSON anyway):

| metric (held-out, in-distribution) | teacher ref | base 1B | base+director |
|---|---|---|---|
| speakers used (judge/prosecutor/defense) | .33 / .32 / .34 | none valid | **.31 / .31 / .38** |
| `prosecutor → defense` rate | 0.65 | — | **0.70** |
| top-1 agreement with teacher | — | 0 | **0.80** |
| distinct intensities, profession leaks | — | — | **5, 0** |
| scoring: spot-on / unrelated / separation | — | — | **100 / 14 / 86** |

The "no defense" problem evaporated. On the contexts it was actually trained for, the director
reproduces the teacher's speaker balance, the prosecutor-then-defense rhythm, agrees with the teacher
80% of the time, and scores guesses in clean calibration. The earlier failures were entirely our
harness.

Three lessons fall out of this, and they generalize past courtroom games. **Grammar guarantees
structure, so gate on content, not on JSON validity** — measuring un-grammared parse rates penalizes
the model for a mode it never runs in. **A single greedy trajectory lies**; an orchestrator is a
distribution and has to be measured as one. And **evaluate in-distribution**: for a model that
conditions on context, the teacher's own held-out contexts are both the fairest prompts and a free
reference. We wasted a day before we believed our own data; reading the data, not the model, is what
broke the deadlock.

## Deployment: one 1B, pure CPU, no grid

The project divides cleanly. Everything expensive happens offline on Modal — teacher data
generation, the curriculum and director training, GGUF conversion, and the benchmarks. None of it
reaches a player. The game itself does all of its inference locally through `llama.cpp` on CPU.

The Space is a **Docker** Space on the free `cpu-basic` (two-vCPU) tier. The Dockerfile exists for one
reason: to compile `llama-cpp-python` with AVX2 so the 1B runs at playable speed, while the app inside
is the same Gradio UI as ever. The weights are not committed — a startup hook pulls the base GGUF, the
director LoRA, and the eight style LoRAs from the Hub — and a GitHub Action mirrors `main` to the
Space, so a normal push redeploys. There is no GPU in the loop at any point.

## What we would tell ourselves at the start

The highest-leverage design decision was refusing to let the small models write freely: forcing the
director to pick from a grammar-constrained deck is what kept a 1B coherent over a dozen turns. The
second was being willing to collapse the architecture — the 4B GM felt necessary right up until a CPU
benchmark proved it was the only thing keeping us off the free tier, and distilling its job into the
1B is what made the whole game fit on a two-vCPU box.

But the theme, the same one as last time, is that almost every real problem was invisible in the
metrics and obvious the moment we read the actual output: the teacher recycling phrases, a filter
over-rejecting good transcripts, thinking-mode returning empty answers, `fault_plain` producing
ungrammatical briefs, and three different benchmarks confidently reporting a "broken" model that was
fine. Reading the outputs — and, the new entry in the list, *reading the training data* — was
consistently more useful than staring at the numbers.

## How this was built

A note on how this came together, because we would rather be transparent than impressive. The
architecture and the important calls were human judgment; the execution leaned heavily on AI pair
programming with Claude Code. The word that matters is *pair*: the agent moved fast — it wrote the
distillation pipeline, ran the Modal jobs, and built the benchmarks — but the decisive moments needed
a human watching closely. The smokescreen concept, the insistence on one base with swappable adapters,
the call to kill the 4B, and above all the refusal to trust a green benchmark until the evaluation
methodology was sound, were steering, not delegation. The genuinely interesting part was learning
where fast execution helps and where careful direction is the thing that counts.

## Try it

The game is live on Spaces, the adapters are on the Hub, and the code is open. Pick a jargon, sit
through the hearing, and tell the court what you think you actually did.

- Play it: [`build-small-hackathon/BuzzwordsMisdemeanors`](https://huggingface.co/spaces/build-small-hackathon/BuzzwordsMisdemeanors)
- The director LoRA: [`BastienHot/buzzwords-director-lora`](https://huggingface.co/BastienHot/buzzwords-director-lora)
- The style LoRAs: [`BastienHot/buzzwords-style-loras`](https://huggingface.co/BastienHot/buzzwords-style-loras)
- The code: [`BastienHot/BuildSmallHackathon`](https://github.com/BastienHot/BuildSmallHackathon)

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/6.0.1/gradio.js"></script>
<gradio-app src="https://build-small-hackathon-buzzwordsmisdemeanors.hf.space"></gradio-app>
