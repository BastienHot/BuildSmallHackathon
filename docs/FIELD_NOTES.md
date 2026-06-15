---
title: "Buzzwords & Misdemeanors: a field log of teaching one 1B model to run a courtroom"
thumbnail: /blog/assets/buzzwords/thumbnail.png
authors:
  - user: BastienHot
---

# Buzzwords & Misdemeanors: a field log of teaching one 1B model to run a courtroom

You wake up in a courtroom. A judge, a prosecutor, and a defense counsel are tearing
into your case, and every word out of their mouths is dense aviation lingo — flight
plans, holding patterns, a manifest missing its flight-signature verification. There is
one problem: you are not a pilot. The jargon is a smokescreen you picked yourself at the
start of the game, and what you actually did has nothing to do with aviation. Pinned to
an evidence board under the stage, oblique exhibits accumulate as the hearing unfolds:
*"No signature on the waiver was found." "The patient was unresponsive to standard
dosing."* Your job is to listen past the buzzwords, read the exhibits, work out your
real profession and your real charge, and enter a plea. A model scores how close you
got, and the truth is revealed. (In that game, you were a tattoo artist who worked on a
sedated client who had never signed the consent form. The court never said so. The
court said you "flew blind in the absence of actual airspace.")

We built this for the Hugging Face **Build Small** hackathon under three self-imposed
constraints: every model genuinely small; all play-time inference local through
`llama.cpp`; and the whole thing running on the **free `cpu-basic` Space tier — two
vCPUs, no GPU, ever**. The shipped game is a single **MiniCPM5-1B** base in 4-bit GGUF
wearing small LoRA adapters: one distilled *director* adapter that writes the case,
runs the trial beat by beat, and grades your plea; and one adapter per jargon style for
the actors. One ~0.8 GB base, nine ~50 MB hats.

<!-- 📷 IMAGE SLOT: docs/assets/img/hero_hearing.png — the hearing screen mid-game:
     courtroom backdrop, a prosecutor speech bubble, and 2-3 exhibits pinned on the
     evidence board below. This is THE shot; take it in the aviation court. -->
<p align="center">
  <img src="https://huggingface.co/spaces/build-small-hackathon/BuzzwordsMisdemeanors/resolve/main/docs/assets/img/hero_hearing.png" alt="The hearing: jargon theater above, the evidence board below" width="460">
</p>

This document is the project's field log, written the way we wish more project
write-ups were written: not the sanitized version, but the actual sequence — including
the night we shipped a game that passed every benchmark and was mathematically
unwinnable, the morning a laptop going to sleep silently killed six hours of data
generation, and the evaluation harness that fooled us three separate times before we
learned to distrust it properly. Everything described here is open: the code, the
adapters, the data-generation pipeline with full manifests, and a pool of 64 complete
agent traces. Links at the end.

## The team, and how the work actually happened

Three of us, with very unequal and very deliberate roles, and we would rather describe
them honestly than imply a balanced division of labor that didn't exist.

**Bastien** (writing this) did the bulk of the build: the game design, the architecture
calls, the training pipeline, the evaluation methodology, the deployment, and the long
tail of debugging that is most of any real project. **Alexandre**, a classmate, joined not because
the project needed more hands but because it was a teaching opportunity — he is earlier
in his ML journey, and walking someone through *why* a train/inference shape mismatch
silently poisons a fine-tune, or why a single greedy trajectory cannot evaluate a
policy, turns out to be the best way to find the holes in your own understanding.
Explaining the project sharpened it. Our third teammate, **Fatih** owns everything you see:
the pixel-art courtroom backdrops — sixteen variants for each of seven court themes,
generated and curated for the game's eight jargon styles — which give the game its
character far more than any line of Python does.

And the fourth presence in the room was an AI pair programmer. We used **Claude Code**
throughout, and we want to be precise about what that meant, because "AI-assisted" can
mean anything. The agent wrote most of the code volume: the contracts module, the
llama-server engine, the Modal data-generation and training scripts, the gate suite,
the Dockerfile. It also ran the operations — launching GPU jobs, watching logs,
diagnosing a TRL API that had churned under us at two in the morning. What it did not
do is decide what the project was. The smokescreen concept, the one-base-many-adapters
architecture, the decision to kill a 4B model rather than pay for a GPU, the refusal to
accept a green benchmark until the methodology was sound, the call that an unwinnable
game needed a game-design fix and not another training run — those were human
judgments, and several of them were made *against* the grain of what the metrics were
saying at the time. The honest summary: the agent multiplied our execution speed by
something like an order of magnitude, and that speed would have been worthless — or
actively dangerous — without someone reading every transcript, every training example,
and every eval table with the intent of catching it (and ourselves) being wrong. Two of
the most important bugs in this log were caught by a human reading model output that
all automated checks had passed.

## Part I — The design: a director and a cast, because small models can't improvise

The naive way to build this game is to hand one model the premise and let it improvise
a trial. With small models this collapses within three turns: they wander, they forget
who is speaking, and — fatally for this game — they blurt out the secret the player is
supposed to deduce. So we borrowed the structure of a film set instead.

A **Game Master** owns the truth and the pacing, and never speaks in character. Each
turn it emits one small structured decision — who speaks next, what kind of beat it is
(an objection, a piece of evidence, a plea…), which hidden clue to surface (or none),
how intense the line should be, whether the hearing should start wrapping up — and,
reacting to the transcript so far, **it writes that speaker's line itself, in plain
courtroom English**. Three **actors** — judge, prosecutor, defense — do not invent
dialogue; they are **translators**: each receives only the director's one plain line and
rewrites it into its assigned jargon style. An actor never sees the profession, the
charge, or even the rest of the transcript — only the public line it is handed. **An
actor cannot leak a secret it does not have.** That guarantee is structural, not
behavioral, and it is the first instance of the design rule this whole project runs on:

> **Code enforces invariants; models provide quality.** Anything the game cannot
> survive losing — the smokescreen, the clue economy, courtroom turn order, "never name
> the profession" — is enforced deterministically in code. The models make it *good*;
> they are never the only thing making it *correct*.

That maxim cuts two ways. The *actor* side is the can't-leak guarantee just described:
structural, not learned. The *director* side — keeping the Game Master on the rails —
is enforced by three code mechanisms:

**Grammars.** Every Game Master output is constrained by a `llama.cpp` GBNF grammar.
A 1B-class model invents badly but classifies well, so "direct the scene" is reduced to
"pick from a finite deck":

```gbnf
root ::= "{" ws "\"next_speaker\":" ws speaker ws "," ws "\"beat_type\":" ws beat
         ws "," ws "\"fact_index\":" ws factidx ws "," ws "\"intensity\":" ws intensity
         ws "," ws "\"line\":" ws string ws "," ws "\"wrap_up\":" ws bool ws "}"
speaker ::= "\"judge\"" | "\"prosecutor\"" | "\"defense\""
```

The grammar also buys a free structural bonus on this hybrid-reasoning model: forcing
`{` as the first generated token makes a `<think>` preamble impossible, which on a CPU
token budget is the difference between an answer and an empty, exhausted generation.

**Sampled truth.** The hidden case is not generated at all. Code samples a profession
and a *domain-matched* fault from a curated pool — 66 professions across 14 domains,
each hand-paired with three faults that belong to that job's world (a wedding
photographer's faults are photography faults, never aviation ones) — with the chosen
jargon's own domain excluded. The smokescreen orthogonality and the
profession-fault coherence are therefore guarantees, not learned behaviors. The
director model writes only the oblique clue *facts* around a truth the code already
controls, and every fact passes a leak filter (with one retry and a curated fallback
bank) before the player can ever see its consequences.

**Deterministic guards.** After every director decision, before any actor speaks, a
guard layer checks the courtroom invariants: the speaker must be allowed to deliver
that beat type (only the defense pleads; the charge belongs to the prosecution; an
opening may come from the judge or the prosecution — more on how the model taught *us*
that rule later); no speaker talks three times in a row; the defense must be heard by
the midpoint of the hearing; and a fact-forcing rule guarantees every clue is released
before the budget runs out, even against a director that never volunteers one. The
guard remaps the decision when violated and otherwise does nothing. It is a seatbelt,
not a co-driver — and in the final shipped system it engages on about **2% of beats**,
because the trained director has mostly internalized the rules. We ship it anyway. The
seatbelt's job is the other 2%.

All of these contracts — the grammars, every system prompt, every prompt-builder, the
guard functions, the enums, the turn budget — live in **one module**,
`buzzwords/contracts.py`, stamped with a `SHAPE_VERSION`. The runtime imports it. The
data-generation scripts import it. The training scripts refuse any dataset whose
manifest carries a different shape version. This sounds like bureaucracy until you read
Part III, where the absence of exactly this mechanism is how we shipped actors trained
on a distribution the game never produces.

<!-- 📷 IMAGE SLOT: docs/assets/img/jargon_picker.png — the Charges screen: title hero
     + the 4×2 icon grid of jargon cards (one selected, gold). -->
<p align="center">
  <img src="https://huggingface.co/spaces/build-small-hackathon/BuzzwordsMisdemeanors/resolve/main/docs/assets/img/jargon_picker.png" alt="Choosing your smokescreen: the jargon picker" width="480">
</p>

With those pieces in place — the director, the actors, and the code that fences them in —
here is one whole game end to end: pick a jargon, the hearing pre-generates behind a
progress bar, then the player steps through the cached beats and enters a plea. The two
right-hand boxes (director and actors) are the same 1B base wearing different hats. The
full turn-by-turn sequence — from jargon pick to verdict — is in the appendix at the end
of this post.

## Part II — One base, many hats: the architecture

At play time exactly one model is resident: **MiniCPM5-1B** (a clean
`LlamaForCausalLM`), quantized to `Q4_K_M` GGUF, served by a **`llama-server`**
subprocess that the app starts, health-checks, and supervises. Every persona is an
adapter or a prompt on top of it:

- **The director** is the base + a single multitask LoRA (`director.lora.gguf`,
  rank 32) covering all three of its jobs — writing the clue facts, the per-beat
  decisions, and scoring the final plea. One adapter rather than three because the
  jobs share a single "case world-model" and a 1B has no capacity to spare on
  redundant copies.
- **The actors** are the base + one LoRA per jargon style (eight of them). Judge,
  prosecutor, and defense are *the same adapter* with three different system prompts —
  a role is a prompt; a vocabulary is an adapter.

The whole path — training and gating offline on Modal, publishing to the Hub, and one
resident base wearing every hat in the Space — is diagrammed in the appendix at the end
of this post.

The server holds the base once, registers all nine adapters at startup
(`--lora-init-without-apply`), and switches them **per request by scale** — no merges,
no reloads, no second copy of the base in RAM. Two server slots are pinned, one for the
director and one for the actor path, so each keeps its own warm prompt cache. The
director's prompt is deliberately **append-only**: system + case brief + the full
transcript so far + one status line. Because the prefix never mutates, each new beat
re-evaluates only the lines added since the last one — on a CPU where prompt
evaluation runs at 60–80 tokens/second, this is the difference between a per-beat cost
of one new line and re-eating a 2,600-token transcript (~42 seconds) every single turn.


The numbers that make the free tier work, all measured on the actual 2-vCPU box:

| What | Measured |
|---|---|
| Prebuilt CPU wheel, prompt eval | **1.8 tok/s** (no SIMD — slower than decode, which was the tell) |
| AVX2 source build, prompt eval | **60–80 tok/s** |
| AVX2 source build, generation | **~12 tok/s** |
| Director decision (grammar-constrained) | settles in ~80 tokens |
| Actor line | ~30–50 tokens |
| Full 10-beat hearing, pre-generated | **~90–120 s** behind a progress bar |

That first row is why the deployed Space is a **Docker** Space: not for the UI (plain
Gradio), but because the Dockerfile is the only place you can guarantee `llama.cpp`
gets compiled from source with `-DGGML_AVX2=ON -DGGML_FMA=ON` without hitting a timeout.
A 30–40× throughput swing from compile flags, on identical hardware and identical weights. 
The build is two-staged (compile in a throwaway image, copy one static binary into a slim Python
image), the weights are pulled from the Hub at startup rather than committed, and a
GitHub Action mirrors `main` to the Space so an ordinary `git push` deploys.

<!-- 📷 IMAGE SLOT: docs/assets/img/loading_bar.png — the "Preparing your hearing"
     card mid-generation ("Staging the hearing — beat 6 of 10…"). -->
<p align="center">
  <img src="https://huggingface.co/spaces/build-small-hackathon/BuzzwordsMisdemeanors/resolve/main/docs/assets/img/loading_bar.png" alt="The whole hearing pre-generates behind a beat-by-beat progress bar" width="540">
</p>

One design decision here was reversed by playtesting, and it is worth recording because
the "smart" version lost. We first shipped *pipelined* generation: show beat 1 the
moment it exists (~30 s), generate beat N+1 in a background thread while the player
reads beat N. Elegant, and the per-beat generation (7–8 s) could have genuinely outrun reading
speed but for fast readers it didn't. Some players progressed too quickly and had to wait in 
between turns, therefore in the end they preferred the progress bar: pre-generate the whole hearing up front
(~2 minutes, with a beat-by-beat counter), then make every click instant. When the
total wait is short enough, predictability beats cleverness. The background worker
survives in the code; the UI now simply waits for it to finish before raising the
curtain.

## Part III — The rebuild: reading our own transcript and tearing it down

There was a version of this game before the one you can play, and it passed its
benchmarks. Then we played it.

One real hearing — aviation jargon, hidden truth "tattoo artist" — produced, in seven
beats: the prosecutor speaking three times in a row; the prosecutor delivering the
*defense's* plea ("My client is deeply remorseful…"); the judge arguing as an advocate;
the phrase "your silence" three times; not a single concrete clue surfaced; and a
revealed fault that was itself aviation-flavored ("ignored a clear flight rule")
despite the profession being a tattoo artist — the smokescreen bleeding into the truth
it was supposed to conceal. Five distinct failure modes in one transcript, from a
system whose director benchmark reported 80% agreement with its teacher and perfectly
balanced speakers.

The benchmark was not lying. It was answering a different question. It asked: *given a
context written by the teacher, does the student choose what the teacher chose?* Yes,
80% of the time. But at runtime the director conditions on **its own actors' lines** —
weaker, repetitive, and drifting further off-distribution with every slightly-wrong
decision. This compounding is *exposure bias*, the textbook gap between teacher-forced
evaluation and autoregressive deployment, and our gate had a second sin on top: the
one test that ran the real stack played exactly **one** game, in one style, with
stubbed actors — a single trajectory, which is the one thing our own field notes
already said never to trust.

So we commissioned a full review of our own codebase (yes, with the AI pair programmer
as the reviewer, and yes, reading a 1,000-line critical review of your own three-day-old
code is humbling). The review found, beyond the post-mortem failures:

- **Contract duplication.** The grammars and prompts were copy-pasted across four
  files with "MUST stay in sync" comments. Our own notes called train/runtime mismatch
  "a silent quality killer," and the layout made that mismatch one careless edit away.
- **The shipped actor adapters were trained on stale data** — an old example shape
  (hidden brief in the system prompt) that the runtime never produces. The style-lift
  metric had masked it: a mismatch-trained adapter still produces dense jargon; what
  silently degrades is its instruction-following.
- **The train/validation split leaked by construction.** Examples exploded from the
  same transcript landed on both sides of a row-level random split, so the reported
  perplexity (~2.0) was optimistic by an unknown amount.
- **Loss was computed over entire sequences, prompts included** — a rank-32 adapter
  spending gradient on memorizing boilerplate it would never need to generate.
- **The content pools were tiny**: 12 professions × 8 generic faults. A third game
  had a high chance of repeating an answer, and anyone reading the open-source repo
  held the entire answer key.

The rebuild kept the architecture — it had survived the review — and rebuilt
everything around it in one day: the contracts module as single source of truth; the
66×3 domain-matched pools; the guard layer; the `fact_index` clue channel in the
decision schema (the director now *names which clue* each beat surfaces, and a forcing
rule guarantees full coverage); the llama-server engine; group-wise splits;
completion-only loss; manifests with `SHAPE_VERSION` enforcement (training now
*refuses* unmanifested data — the stale-adapter incident, made unrepresentable); and a
gate suite designed around self-rollouts. Thirteen unit tests reproduce the degenerate
transcript's every failure mode against a scripted worst-case director and prove the
deterministic layer makes each one impossible.

## Part IV — Data generation: a 31B teacher under code-enforced constraints

All training data is written offline by a teacher — **Gemma 4 31B-it in FP8** on an
L40S via vLLM on Modal — and never touches a player's machine. The teacher is brilliant
and unreliable in exactly the ways you would expect, so the pipeline treats it like a
talented contractor with strict acceptance criteria.

<!-- diagram source: docs/assets/diagrams/datagen.mmd — rendered to PNG (HF blog does not render mermaid) -->
<p align="center">
  <img src="https://huggingface.co/spaces/build-small-hackathon/BuzzwordsMisdemeanors/resolve/main/docs/assets/img/diagram_datagen.png" alt="A 31B teacher under code-enforced acceptance criteria: code samples the truth, the teacher writes and translates, validators sharing the runtime contracts accept or send a corrective retry, and only manifested data reaches training" width="820">
</p>

**Actors** (~14,700 examples). For each of eight styles plus a generic courtroom
register: the teacher writes a complete hearing in plain English — given a code-sampled
hidden truth, a tone, a disposition, and a seeded sample from a curated ~80-term jargon
bank — then *translates each line into the target register*. Each translation becomes one
training example *in the exact shape the runtime produces*: a role+style translator
system prompt, a user turn carrying only the plain line to rewrite, and the in-style line
as the answer. Every example carries a `group_id` tying it to its source transcript so
the train/validation split can separate at the transcript level.

**Director** (4,178 examples). Three tasks mixed into one dataset: writing oblique
facts (898), per-beat decisions on real game transcripts (2,980, with the `fact_index`
labels), and plea scoring across seven guess-quality buckets (300). The teacher's
games are validated turn by turn against the *same* contracts the runtime enforces:
speaker/beat compatibility, no three-in-a-row, a written line of at least four
words, no profession token anywhere on a player-visible surface (the leak check is the
identical function the runtime uses), fact indices in range. The forcing rule is
applied to the training *targets* too, so the model learns to obey the nudge it will
receive at inference.

Every dataset ships with a manifest: seed, counts requested versus kept, a
reject-reason histogram, the teacher model ID, and the `SHAPE_VERSION` of the
contracts that generated it. Yield on the final runs was effectively perfect — 200/200
transcripts kept per style, 298/300 director games (a corrective retry that re-prompts
with the rejection reason recovered 19 of 21 first-pass rejects) — but the road to
those numbers is the educational part:

- **The validators caught the teacher; then the teacher caught our spec.** When we
  added the beat/speaker compatibility check, the first batch rejected **0 of 8 — then
  8 of 8**: every game opened with `prosecutor/opening`, which our map forbade
  (opening = judge only). The teacher would not budge, because the teacher was right —
  a prosecution opening statement is correct courtroom procedure; our contract was
  stricter than reality. We widened the contract instead of fighting the model's
  prior, re-ran, and got 8/8 first-attempt compliance. When a validator and a strong
  teacher disagree systematically, check the spec before blaming the model.
- **Eyeballing one sample fixed a quality bug no metric measured.** An early smoke run
  printed a director `"line"` of *"vetoing the rhetorics"* — three words of nothing for
  the actor to translate and the player to read. The prompt now demands a full
  sentence and the validator rejects terse lines. The very next run
  produced *"Close the distance to apply maximum pressure on the bottom line."* Same
  pipeline, same teacher; the difference between unusable and excellent training data
  was one prompt sentence and one validator line — found by reading, not by metrics.
- **A teacher needs the enum sheet, not an example.** Our first attempt at
  teacher-labeled decisions (for the self-agreement baseline) silently produced **0
  valid labels out of 390** — the teacher invented beat names ("deflection",
  "evidence_introduction"), capitalized speakers, used intensity 6. The label prompt
  had shown a JSON *shape* but never the allowed *values*. Spelling out the enums took
  the yield to 390/390 in one run. The failure was invisible until we made rejects
  print themselves; silent validation is no validation.

## Part V — Training: bf16 LoRA, honest splits, and a one-prefix mismatch

All nine adapters train on Modal A100s with Unsloth, and the configuration is
deliberately boring: rank 32, alpha 32, dropout 0, all seven projection matrices,
2e-4 cosine, two epochs, fixed seed. The interesting decisions are around the
configuration:

**bf16 LoRA, not QLoRA.** Exact quantization matching between training and serving is
impossible anyway (bitsandbytes 4-bit and GGUF `Q4_K_M` are different formats), and a
1B trains comfortably in bf16 — faster than QLoRA, with zero training-side
quantization noise. The serving-side quantization is validated where it can actually
be validated: in the end-to-end gate that runs the real GGUF stack.

**Curriculum by forking, not resuming.** The actors train in two stages: one
`legal_base` adapter learns the courtroom register on generic data, then each style
adapter is initialized *from its weights* with a fresh optimizer and schedule —
a fork, never `resume_from_checkpoint`. The base model itself is never touched; the
app ships the official base GGUF plus deltas.

**Group-wise splits and completion-only loss.** Validation splits on transcript
`group_id`s, never on rows, so sibling examples from one hearing cannot straddle the
split. Loss is masked to the assistant's answer only. The resulting numbers are
honest and therefore *worse-looking* than the leaky ones they replaced: the director
lands at perplexity **2.31** (structured JSON is low-entropy), the actors at **9–14**
(free-form theatrical dialogue is not — and these are real held-out numbers, unlike
the old ~2.0).

**The empty think block.** The subtlest bug we caught never got to happen. Probing
MiniCPM5's chat template revealed that with `enable_thinking=False`, the *generation*
prompt ends with an empty `<think>\n\n</think>` block — the template closes the
reasoning channel by pre-filling it — while rendering a *complete* conversation omits
that block entirely. Train on naively-rendered conversations and your model learns a
prompt that is one prefix away from the one it will see at inference, on every single
example. Our training renderer now constructs each example as **generation prompt +
answer + terminator** — token-for-token what the server produces at runtime, empty
think block included — and the completion mask is derived by probing the template with
sentinels rather than hardcoding any string. The audit that prints one fully rendered
example per run, originally added to check for thought-channel leakage, is what made
this discoverable at all. 

## Part VI — Evaluation: the gauntlet, and the night the game was unwinnable

We had been fooled by our own evaluations three times in the project's earlier life —
by a single greedy trajectory, by out-of-distribution prompts, and by gating on JSON
validity that the runtime grammar guarantees anyway. The rebuilt evaluation suite is
designed by those scars, in three layers, each answering a different question.

<!-- diagram source: docs/assets/diagrams/evaluation.mmd — rendered to PNG (HF blog does not render mermaid) -->
<p align="center">
  <img src="https://huggingface.co/spaces/build-small-hackathon/BuzzwordsMisdemeanors/resolve/main/docs/assets/img/diagram_evaluation.png" alt="The three-layer gate: two learned-behaviour benches feed an end-to-end gate on the production stack, eight machinery checks, and a solvability metric that can veto a ship even at eight-of-nine" width="820">
</p>

**Layer 1 — Did the student learn the teacher?** A held-out benchmark on a
disjoint-seed slice of teacher games (390 contexts), base versus LoRA, *without* the
grammar (we measure learned behavior; the runtime grammar guarantees structure
regardless). The rebuilt director: speaker shares .22/.45/.39 against the teacher's
.20/.37/.43; prosecutor→defense response rate 0.83; valid clue indices 100%; zero
leaks; scoring separation 85 (spot-on guesses 100, unrelated 10–15). And one number we
are unreasonably fond of: top-1 agreement with the teacher is **0.87**, but when we
resampled the *teacher on its own contexts* — the baseline we had never measured —
the teacher only agrees with itself **0.64** of the time. Several speakers are often
legitimately valid, so the ceiling is low; the student sits *above* its teacher's
self-consistency. Distillation, saturated. Without that baseline, "0.87 agreement"
would have been an uninterpretable number that merely sounded good.

**Layer 2 — Did each actor learn its voice?** Style lift measured against curated
jargon banks: vanilla base versus adapter on identical runtime-shaped prompts, with a
fluency guard (lexical diversity must not collapse) and own-bank specificity (the
corporate adapter should peak on the corporate bank, not just any bank). Seven of
eight adapters pass cleanly, with bank-term density going from ~0 to 0.3–1.1 per
line. The eighth taught us about the limits of the yardstick: the sci-fi adapter
scored **zero** on its own bank, while producing lines like *"a fraudulent
phase-shift of the chronos-leech protocol"* — unmistakably in-style, just not on the
list. The training data itself contained bank terms in only 4% of lines; the teacher
writes open-vocabulary technobabble no closed list can capture. We documented the
limitation in the eval and accepted the adapter on inspection. A metric that fails a
model for inventing *better* vocabulary than your checklist is a metric, not a judge.

**Layer 3 — Is the game real?** The headline gate plays **N complete games on the
production stack** — real llama-server, real grammars, real style adapters, guards
live — and aggregates everything: speaker shares and transition matrices, maximum
same-speaker runs, how often the guard seatbelt fired, cross-line 4-gram repetition,
leaks on every player-visible surface, wrap-up rates, clue coverage, scorer
calibration, and the one metric that is the game itself: **solvability** — the 31B
teacher, given only what the player sees, must guess the profession and charge, and
its deterministically-scored result must land in a band (30–85): winnable, not given
away.

The first full run of that gate is the best story in this log. **Eight of nine checks
passed flawlessly** — zero guard triggers in 117 beats, zero repetition, zero leaks,
every game wrapped, every clue delivered, scoring cleanly calibrated. Solvability:
**0 out of 12.** The teacher guessed "accountant / embezzlement" for a tattoo
artist's case. We had built a mechanically perfect, completely unwinnable game — the
exact failure the metric had been designed for, before we knew we would need it.

The diagnosis took one controlled experiment. We re-ran the solver on the raw clue
facts alone, no jargon layer: 21.7 mean, 4/12 professions recovered. So the facts
carried *some* signal and the transcript carried *none* — because the actors were
doing precisely what they had been trained to do. Read the chain for the tattoo
artist: the fact "no signature on the waiver was found" became "the manifest lacks
all necessary flight-signature verification"; "the skin remained permanently covered"
became "the fuselage was sealed off from the wind." The actors translate every
concrete noun *into the smokescreen's own vocabulary*. That is the intended aesthetic
— and it destroys every anchor a solver could invert. Oblique facts, re-encoded
obliquely: obliqueness squared. Even a 31B cannot decode metaphor with no fixed
points, and no amount of additional training toward the *same target* could fix it,
because the target itself was unsolvable. The models had faithfully learned a
specification that did not contain a winnable game.

<!-- diagram source: docs/assets/diagrams/obliqueness.mmd — rendered to PNG (HF blog does not render mermaid) -->
<p align="center">
  <img src="https://huggingface.co/spaces/build-small-hackathon/BuzzwordsMisdemeanors/resolve/main/docs/assets/img/diagram_obliqueness.png" alt="Obliqueness squared: an oblique fact re-encoded into the smokescreen leaves a transcript-only solver with no anchor (solvability 0/12); the fix routes the same clues to the player verbatim on the evidence board (solvability 31.75, in band)" width="800">
</p>

The fix was therefore three-part, and only one part touched a model:

1. **The evidence board.** Released clues now surface to the player directly, as
   exhibits pinned under the stage, as each beat plays. The jargon theater is the
   entertainment layer; the oblique facts are the puzzle layer. This was always the
   design's "clue economy" intent — the transcript-only version had silently assumed
   the actors would transmit the clues faithfully, and measurement killed that
   assumption. Pure UI; no retraining.
2. **Sharper facts at the source.** 21.7 was below the band even reading the facts
   directly, so the facts prompt now requires that at least one fact evoke the
   distinctive tools or workplace of the job, and that the set *collectively* let a
   sharp guesser name it. The next smoke test produced, for an arcade owner rigging
   claw machines: "the tension settings on the gripping arms were manually adjusted,"
   "a technician modified the payout frequency," "patrons reported the prize dropped
   before reaching the chute." Gripping arms, payout, prize chute: nameable.
3. **Gate on the player's actual view.** The solver now reads exhibits + hearing,
   because that is what the player reads.

One director regeneration and retrain later (the full chain — data, held-out slice,
training, labels, benchmark, GGUF conversion, gate — runs unattended in about three
hours), the gate came back **9/9**: solvability 31.75, 58% exact-profession hits,
guards firing on 1.7% of beats, repetition 0.9%, zero leaks. The single "leak" the
run did flag was our own detector matching the substring "city" inside the word
*opacity* for a city council clerk — fixed with word-boundary matching and a
regression test, and a fitting final note: by the end, the only component still
producing false accusations in our courtroom was the prosecution's own software.

<!-- 📷 IMAGE SLOT: docs/assets/img/verdict_reveal.png — the verdict screen: the big
     score %, the verdict word, and the two-column "What you heard / The truth" reveal. -->
<p align="center">
  <img src="https://huggingface.co/spaces/build-small-hackathon/BuzzwordsMisdemeanors/resolve/main/docs/assets/img/verdict_reveal.png" alt="The verdict: what you heard versus what you actually did" width="380">
</p>

## Part VII — Ops notes from the trenches

A research log owes its readers the unglamorous parts, because they cost as many
hours as the clever parts.

**A sleeping laptop kills detached-looking work.** Our first full data-generation
night died silently: `modal run` keeps the remote app alive via a client heartbeat,
and when the laptop slept at 00:40, Modal tore the GPU jobs down server-side. Both
local processes woke at 07:07, failed heartbeats for 45 seconds, and exited —
**with code 0**. The volume timestamps, not the exit codes, told the truth (one
finished dataset; everything else still the stale files from days before — which is
also why training now hard-refuses any dataset without a fresh manifest).
Everything thereafter ran with `--detach` and single submitted remote calls that
survive client loss.

**Your log filter can kill your job.** We lost one complete gate run — an hour of
CPU games — because the results only lived in the local client's memory and the
shell pipeline truncating its logs (`| head -60`) closed the pipe, which killed the
client mid-solver. The gate now persists every game to the volume and has a
`--solve-only` mode that resumes from there. Artifacts belong on durable storage at
the moment they exist, not at the end of the run.

**Honest empty-input handling has to be deterministic.** In playtesting, submitting
an *empty* plea scored 60%. Of course it did: we had asked a 1B to grade a blank
string, and it hallucinated partial credit. Silence is now scored 0 by code, before
any model is consulted — with a regression test asserting the model is *never
called* — and the rationale the player sees is the courtroom's own: "the court
cannot grade silence."

## Part VIII — What we would tell ourselves at the start

1. **Code for invariants, models for quality.** The single most load-bearing idea in
   the project. The smokescreen, the clue economy, the turn order, the empty-plea
   score, the leak filter — every promise the game makes is kept by code; the models
   make the kept promise *entertaining*. Each time we relaxed this rule we got
   burned, and each fix was re-applying it.
2. **The training distribution must be the inference distribution — token for
   token.** Shape mismatches do not crash; they quietly degrade. The fixes that
   mattered: one contracts module imported by runtime and training alike, manifests
   with shape versions that training refuses to violate, exact-inference rendering
   down to a template's empty think block, and a printed, audited example per run.
3. **Evaluate the policy, not the snapshot.** Teacher-forced agreement said 0.80
   while real games were degenerate. Only self-rollouts — the model conditioning on
   its own outputs, real actors, real grammars — measure the thing you ship. And
   report every agreement number next to a self-agreement ceiling, or it is
   uninterpretable.
4. **Have one metric that *is* the product, and let it veto everything.** Eight
   machinery checks passed while the game was unwinnable. Solvability — a strong
   model playing the player's exact view, scored deterministically, required to land
   in a band — was the only number that knew. Find your equivalent before your
   benchmarks start agreeing with each other.
5. **Read the outputs. Then read the data.** The terse director lines, the
   ungrammatical briefs, the prosecutor/opening "violation" that was actually our
   spec being wrong, the sci-fi adapter failing a checklist while writing better
   technobabble than the checklist — every one was invisible in aggregate metrics
   and obvious in thirty seconds of reading. The single highest-value habit in the
   whole project.
6. **Constraints are a forcing function.** Two vCPUs killed the 4B director and
   forced the distillation that became the project's core contribution. The free
   tier was not the obstacle; it was the brief.

## Part IX — Truly open: everything on the table

Open source for an ML project has to mean more than a code dump, so here is the
complete inventory:

- **The game**, live on a free Space:
  [`build-small-hackathon/BuzzwordsMisdemeanors`](https://huggingface.co/spaces/build-small-hackathon/BuzzwordsMisdemeanors)
- **The code** — runtime, training, evaluation, deployment:
  [`BastienHot/BuildSmallHackathon`](https://github.com/BastienHot/BuildSmallHackathon).
  The contracts module, the guard layer, and the 14 unit tests reproducing every
  post-mortem failure mode are all there; `training/README.md` is the exact run
  order (every Modal command, smoke tests first) to regenerate everything from
  scratch.
- **The adapters**, with the gate results they shipped under:
  [`BastienHot/buzzwords-director-lora`](https://huggingface.co/BastienHot/buzzwords-director-lora)
  and [`BastienHot/buzzwords-style-loras`](https://huggingface.co/BastienHot/buzzwords-style-loras).
  The base is the unmodified
  [`openbmb/MiniCPM5-1B`](https://huggingface.co/openbmb/MiniCPM5-1B-GGUF) — we ship
  deltas, never a mutated base.
- **64 complete agent traces**:
  [`BastienHot/buzzwords-agent-trace`](https://huggingface.co/datasets/BastienHot/buzzwords-agent-trace).
  Each trace records, per beat, the director's raw structured decision, whether a
  deterministic guard remapped it, the clue channel, the actor's line, the hidden
  truth, and the scorer calibration probes — the full anatomy of an orchestrated
  game, not just its transcript.
- **The data pipeline with provenance.** Every dataset is regenerable from a seed,
  and every generated file carries a manifest: seed, counts requested versus kept,
  reject-reason histogram, teacher model ID, and the `SHAPE_VERSION` of the
  contracts that produced it. Reproducibility is a property of the pipeline, not a
  promise in a README.

Known, accepted limitations, in the open as well: one teacher (Gemma 4 31B), one
prompt family per task — the student's variety is bounded by one model's voice, and
agreement metrics partially measure "sounds like Gemma"; the jargon-bank yardstick
under-measures open-vocabulary styles; the solvability band is calibrated against a
31B solver, not against humans (early human playtests suggest the band is roughly
right, and humans bring world knowledge the solver lacks); and the "purist mode" —
actors trained to keep clue nouns recognizable so the hearing is solvable *without*
the evidence board — is designed but unbuilt, and would be the first thing we
attempt with another week.

## Try it

Pick a jargon. Sit through your hearing. Watch the exhibits pile up. Then tell the
court, in your own words, what you think you actually did.

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/6.0.1/gradio.js"></script>
<gradio-app src="https://build-small-hackathon-buzzwordsmisdemeanors.hf.space"></gradio-app>

## Appendix — the full diagrams

The two whole-system diagrams, collected here so they don't interrupt the read.

**One game, end to end** — from jargon pick to verdict. The two right-hand lanes
(director and actor) are the same 1B base wearing different hats.

<!-- diagram source: docs/assets/diagrams/interactions.mmd — rendered to PNG (HF blog does not render mermaid) -->
<p align="center">
  <img src="https://huggingface.co/spaces/build-small-hackathon/BuzzwordsMisdemeanors/resolve/main/docs/assets/img/diagram_interactions.png" alt="Sequence diagram: one game, from jargon pick to verdict" width="860">
</p>

**One base, many hats: the architecture** — adapters are trained and gated offline on
Modal, published to the Hub, then pulled into a free 2-vCPU Space where one resident 1B
base wears every hat.

<!-- diagram source: docs/assets/diagrams/architecture.mmd — rendered to PNG (HF blog does not render mermaid) -->
<p align="center">
  <img src="https://huggingface.co/spaces/build-small-hackathon/BuzzwordsMisdemeanors/resolve/main/docs/assets/img/diagram_architecture.png" alt="One base, many hats: adapters are trained and gated offline on Modal, published to the Hub, then pulled into a free 2-vCPU Space where one resident 1B base wears every hat" width="860">
</p>

*Built by Bastien Hottelet (design, ML, engineering), with a classmate learning the
ropes alongside (and making me explain myself, which made everything better), art
by our third teammate, and Claude Code as the pair programmer. The full revision
history of this project — including the review that tore it down and the gates it had
to pass to come back — is in the repository.*

