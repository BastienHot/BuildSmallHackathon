# Buzzwords & Misdemeanors — Full Codebase Review & Rebuild Blueprint

*Reviewed 2026-06-10, at commit `668da3c`. Scope: game design, hackathon constraints,
runtime architecture, synthetic data generation, training, evaluation and gating,
app integration, and CPU inference. This document is the agreed base for the rebuild
of the training pipeline and the app integration.*

---

## 1. Executive summary

The project is in unusually good shape for a hackathon: the core design (a
grammar-constrained 1B director plus per-style actor adapters on one shared base) is
sound, the distillation pipeline produced a validated director adapter, and the
evaluation story — after three documented false starts — landed on a methodology that
is genuinely correct (held-out, in-distribution contexts with the teacher as the
reference). The CPU deployment insight (AVX2 source build, no-think via grammar,
prompt caching) is the difference between a demo and a slideshow, and it is measured,
not asserted.

The most important findings, in order of how much they should shape the rebuild:

1. **Contract duplication is the number-one structural risk.** The grammars, system
   prompts, and the Game Master prompt template are copy-pasted, with "MUST stay in
   sync" comments, across four files (`buzzwords/text_engine.py`,
   `training/director_datagen.py`, `training/director_evaluate.py`,
   `training/director_gate.py`), plus the actor prompts duplicated between
   `buzzwords/text_engine.py` and `training/teacher_datagen.py`. The project's own
   field notes identify train/runtime mismatch as "a silent quality killer" — and the
   current layout makes that mismatch a single careless edit away. The rebuild should
   make one shared contract module the single source of truth (Section 10.1).

2. **The committed actor data is the stale, pre-fix shape.** `data/style_*.jsonl` in
   the repo still carries the old example shape (hidden brief in the system prompt,
   previous line as the user turn) — exactly the train/runtime mismatch the field
   notes say was fixed. The shipped style LoRAs were therefore likely trained on
   mismatched data and need regeneration and retraining (Section 5.1).

3. **Train/validation split leaks by construction.** Both `finetune.py` and
   `director_finetune.py` split at the *example* level, but examples are exploded
   from shared transcripts/games — so sibling rows from the same case land on both
   sides of the split. The reported held-out loss and perplexity (~2.0) are
   optimistic by an unknown amount. Split by transcript/game identifier instead
   (Section 6.1).

4. **The game's content diversity is bounded at ~12 professions × 8 faults.** The
   director was trained only on case files drawn from these fixed pools, so at
   runtime it will mostly reproduce them. A returning player sees repeats within a
   handful of games. This is the single biggest *player-facing* limitation
   (Section 5.2).

5. **The runtime has no deterministic anti-spoiler latch and no explicit no-think
   guard on the actor path.** Both are cheap to add and both protect the core game
   promise (the player must be able to guess, and must not be told). The actor call
   has neither a grammar nor an explicit thinking-off toggle — it currently depends
   on the GGUF's embedded chat template behaving (Section 4.4, 4.5).

6. **The final gate contradicts the project's own evaluation lessons.** After
   correctly concluding that "a single greedy trajectory lies",
   `director_gate.py` gates on… a single game, in a single style, with stubbed actor
   lines. It should play N games across styles and aggregate (Section 7.1).

7. **The app diverged from its own architecture document on serving.** The
   architecture calls for `llama-server` with one resident base and per-request LoRA
   scales; the app instead instantiates separate `llama_cpp.Llama` objects per
   adapter (two copies of the same base in RAM, a buggy Python-side prefix-reuse
   path that already forced a "safe mode" workaround, and no clean per-request
   adapter switching). The rebuild should either adopt `llama-server` as designed or
   deliberately re-commit to the Python binding with the known costs written down
   (Section 8.2).

None of these invalidate the design. The architecture (one 1B base + a director
adapter + style adapters, all behind GBNF) survives this review intact and should be
kept. The rebuild is about consolidating contracts, regenerating the actor data,
hardening the splits and gates, and closing the runtime gaps.

---

## 2. The project, the hackathon, and the constraints

### 2.1 The game

The player wakes up in a courtroom. A judge, a prosecutor, and a defense counsel
argue the player's case in dense jargon of a *style the player chose* (corporate,
aviation, medical, …). The jargon is a **smokescreen**: it is deliberately unrelated
to the hidden truth. A hidden Case File — the player's real profession, the fault in
plain English, and a few oblique facts — was written by the Game Master at the start
and is never shown. After a short, turn-budgeted hearing, the player guesses their
real profession and charge in plain English; a model scores the guess 0–100 and the
truth is revealed.

The game mechanic that makes this a *game* rather than a text toy is the
profession-⊥-jargon orthogonality: the player cannot pattern-match the vocabulary
and must instead decode the oblique facts woven through the smokescreen.

### 2.2 The hackathon constraints (self-imposed and external)

For the Hugging Face **Build Small** hackathon, the project committed to:

- **Small models only** — everything at play time well under 32B (in the end, a
  single 1B); the 31B teacher runs offline only and never reaches a player.
- **Fully local inference** — all play-time text generation goes through the
  `llama.cpp` runtime, no cloud inference in the loop.
- **Free-tier hardware** — a Hugging Face Docker Space on `cpu-basic`: **2 vCPUs, no
  GPU**. This is the constraint that drove the architecture's biggest decision
  (killing the 4B Game Master) and most of the inference engineering.

The stated goal is the right one and worth restating because it disciplines the
rebuild: *not* a frontier model, but a fun, playable app whose engineering and
evaluation are scientifically clean. Every recommendation below is filtered through
that lens — nothing is proposed that only makes sense for a production ML system.

### 2.3 How the architecture got here (the recorded history)

The evolution is well documented across `docs/ARCHITECTURE.md` (v1 → v2 notes),
`docs/FIELD_NOTES.md`, and the project memory, and it matters for the rebuild
because each pivot was evidence-driven:

1. **v0 design:** 4B Game Master (vanilla, grammar-constrained) + 1B actors with
   per-style LoRA adapters. Rationale: directing is the reasoning-heavy job.
2. **CPU benchmark (2026-06-08):** on the free 2-vCPU tier, the 4B decodes at ~1.2
   tokens/second — about four minutes per decision, over an hour per trial. The 1B,
   *with an AVX2 build*, does 60–80 tokens/second prompt evaluation and ~12
   tokens/second generation. Verdict: the 4B is non-viable on the target hardware.
3. **Gate test of the base 1B as Game Master:** incoherent — broken smokescreen,
   degenerate decisions, prompt echoes. Conclusion: the 1B *must be fine-tuned* to
   direct.
4. **v2 (shipped):** the director's whole job (case file + beat decisions + scoring)
   distilled from a Gemma 4 31B teacher into a single multitask LoRA on the same
   MiniCPM5-1B base the actors use. The entire game is one ~1B base + two ~90 MB
   adapters, pure CPU.

This is a textbook example of letting the constraint (2 vCPUs) drive the
architecture rather than the other way around, and the rebuild should preserve the
collapsed all-1B design.

---

## 3. What is working well — keep these in the rebuild

These are genuine strengths, several of them above the bar for a hackathon, and the
rebuild should treat them as fixed points rather than re-litigating them.

**3.1 Grammar-constrained orchestration ("a small model classes well but invents
poorly").** Forcing every Game Master output through a GBNF grammar — case file,
beat decision, score — turns "direct a scene" into "pick from a finite deck". This
is what keeps a 1B coherent over a dozen turns, and it has a free structural bonus:
`root ::= "{"` makes a thinking preamble impossible for the director, which on this
hybrid-reasoning model is the difference between an answer and an empty, budget-
exhausted generation. The grammar strings themselves are carefully done (the
single-line rule constraint of llama.cpp is respected; the string rule excludes
control characters and newlines specifically so the model cannot pad into
truncation).

**3.2 Smokescreen by construction, not by hope.** The training pipeline draws the
hidden profession from a fixed pool with the jargon's own domain excluded
(`STYLE_EXCLUDE`), so every training target is *provably* profession-⊥-jargon; the
teacher only writes the oblique phrasing around a truth the pipeline already
controls. The same philosophy — inject the invariant, don't pray for it — fixed the
50% leak-reject rate via sampled few-shot exemplars, an explicit banned-word list,
and a corrective retry. This is the right way to get guarantees out of a fallible
teacher and should be the template for every future invariant.

**3.3 Train-shape equals runtime-shape, enforced deliberately.** Every training
example is emitted in the exact message shape the app sends at inference (the actor
gets role+style system and a "Stage direction: …" user turn with *no* hidden brief;
the director gets the three runtime call shapes verbatim). The project learned this
the hard way (the original actor data had the brief in the system prompt) and now
treats it as a hard rule. Correct rule — the rebuild's job is to make it
*structurally* enforced rather than enforced by comments (Section 10.1).

**3.4 The evaluation methodology that finally stuck.** The director benchmark
(`director_evaluate.py`) evaluates base-versus-LoRA on a disjoint-seed **held-out**
slice of real teacher games, using the teacher's own choices as reference labels,
over ~180 contexts, without the grammar (so it measures learned behavior, with JSON
validity demoted to an informational floor since the runtime grammar guarantees
structure anyway). The three recorded lessons — gate on content not on JSON
validity, never judge an orchestrator on a single greedy trajectory, evaluate
in-distribution — are correct and hard-won. The published numbers (speaker balance
.31/.31/.38 vs teacher .33/.32/.34, prosecutor→defense 0.70 vs teacher 0.65, 80%
top-1 agreement, zero leaks, score separation 86) are a credible validation of the
director on its trained distribution.

**3.5 The checkpoint-fork curriculum (no merge).** Stage 1 trains one `legal_base`
adapter on generic courtroom register; stage 2 loads it as *initialization* (fresh
optimizer and schedule — explicitly not `resume_from_checkpoint`) and forks per
style. The base stays vanilla, so the app ships the official base GGUF plus N small
adapters. The distinction between forking and resuming is correctly understood and
documented in three places.

**3.6 The CPU inference findings.** Three measured facts carry the deployment: the
prebuilt `llama-cpp-python` CPU wheel has no SIMD (1.8 tokens/second prompt
evaluation — slower than decode, which was the tell); an AVX2 source build gives
60–80 prompt / ~12 decode tokens per second on the same 2-vCPU box; and prompt
caching makes the growing-transcript loop ~80× cheaper than cold re-evaluation. The
Dockerfile exists solely to bake in the AVX2 build, with the compile flags tuned to
fit the Space's build-time budget. All of this is correct and verified.

**3.7 Operational hygiene.** Seeded, reproducible data generation with printed
samples and reject-reason tallies ("always smoke-test small and eyeball the output");
the app launches without weights and tells the user exactly what is missing
(`preflight`); failures log full tracebacks to the container logs; the UI closes a
hearing early rather than aborting it if generation dies mid-trial. The field notes
are honest about what went wrong, which makes this codebase auditable in a way most
hackathon code is not.

---

## 4. Runtime architecture findings (`buzzwords/`)

Severity scale used below: **High** = should block the rebuild's "done" definition;
**Medium** = fix during the rebuild; **Low** = note and decide.

### 4.1 Two resident copies of the same base model — Medium

`config.GM_MODEL` and `config.JARGON_BASE_MODEL` point at the same
`MiniCPM5-1B-Q4_K_M.gguf`, but because the Game Master and the actor are created as
separate `llama_cpp.Llama` instances (different cache keys: `gm-director` versus
`actor-<style>`), the ~0.8 GB base weights are loaded **twice**, each with its own
KV cache (the GM's sized at 8192 context). On `cpu-basic` (16 GB RAM) this fits,
so it is not a crash risk — but it doubles model load time on cold start, doubles
the memory page working set (which matters for CPU cache behavior on a 2-vCPU
box), and forecloses the cleaner design the architecture document itself specifies:
one resident base served by `llama-server`, with the director and style adapters
loaded via `--lora-init-without-apply` and switched per request by scale. See
Section 8.2 for the serving recommendation.

### 4.2 The prompt-cache "safe mode" is a workaround for a self-inflicted hazard — Medium

`TextEngine` keeps llama-cpp-python's Python-side prefix-reuse on the happy path
and, after the first runtime fault, permanently degrades the session to full
re-prefills (`enable_safe_mode`), because the reuse path "can raise 'index N out of
bounds…' on some models". Two observations:

1. The hazard lives in the Python binding's longest-prefix-reuse logic, not in
   llama.cpp proper. `llama-server`'s slot-based caching does not have this
   failure mode. Choosing the binding *and* its buggy reuse path is what created
   the need for safe mode in the first place.
2. Even when reuse works, the Game Master prompt largely defeats it. The prompt is
   `system + hidden brief + transcript[-6:] + turn counter`. Because the transcript
   window *slides*, the shared prefix across consecutive turns ends where the
   window begins — everything from the first windowed line onward changes every
   turn. Only the system prompt and brief (~150–250 tokens) are reusable; the
   ~500+ tokens of windowed transcript re-prefill every beat. At 60–80 tokens/
   second prefill this costs ~6–9 seconds per beat, which is most of the per-beat
   latency.

The rebuild should choose the prompt structure *for* the cache: keep a stable
prefix (system + brief) and append-only transcript (full transcript, not a sliding
window) so each turn only prefills the new lines. With a 12-beat budget and 1–2
sentence lines, the full transcript stays under ~1,500 tokens — well within an
8192 context, and strictly cheaper than re-prefilling a 6-line window every turn.

### 4.3 Actors generate with no dialogue context — Medium (design decision to revisit)

`TextEngine.act()` sends the actor only `role + style` (system) and
`stage_direction + intensity` (user). The actor never sees a single previous line.
All continuity therefore rides on the Game Master's one-sentence stage direction.
Consequences observable in play: actors cannot reference or rebut what was just
said, exchanges read as a sequence of non-sequiturs glued together by the judge,
and nothing prevents two near-identical lines in a row.

This was a deliberate simplification (it keeps the actor prompt tiny and the
training data shape trivial), and it does have a real benefit the rebuild must
preserve: **the actor physically cannot leak the truth because it never sees it.**
But the fix is cheap and compatible with that guarantee: include the last 1–2
transcript lines in the actor's user turn (they are public, already-spoken text —
no secret enters the actor), and regenerate the actor training data in that same
shape. This buys local coherence ("Objection! Counsel's 'synergy audit' is pure
theater…") at a cost of ~60–100 extra prefill tokens per beat.

### 4.4 No deterministic anti-spoiler latch on the player-visible surface — High

v1 explicitly deferred the regex latch and trusts the Game Master. But the
runtime *does* hand the plain-English truth to the GM every turn ("the defendant is
a {profession} who {fault_plain}"), and the GM's `stage_direction` is then passed
verbatim into the actor's prompt — and the actor's line goes straight to the
player. The training-side leak filters were applied to the *teacher's* outputs, not
to what the runtime student generates; the held-out benchmark measured a 0% leak
rate for the director *in distribution*, but the actor line itself is never
checked at all, and a single leak ends the game's premise.

A latch is ~10 lines: after `act()`, substring-check the line (and the stage
direction) against the profession tokens (the same `_banned`-style token list used
in training) and regenerate once on a hit, falling back to a generic line. Cost:
near zero on the happy path. The original v1 worry — that a latch makes the system
rigid — does not apply to a pure post-hoc spoiler check. **Recommendation: reinstate
it in the rebuild.**

### 4.5 Nothing explicitly disables thinking on the actor path — High (verify)

The field notes correctly state that thinking mode left on returns empty answers,
and that the *director* is structurally protected because its grammars force `{` as
the first token. The actor call, however, has **no grammar and no explicit
no-think toggle** — `create_chat_completion` is called with plain messages, so the
behavior depends entirely on what the GGUF's embedded chat template does by
default for this hybrid-reasoning model. If the template defaults to thinking-on
(or a llama.cpp update changes template handling), every actor line silently
burns its 160-token budget on a thought channel and comes back empty or truncated.
The project's own memory notes name "No-Think" as a required inference setting.

Rebuild action: make the suppression explicit and tested — either pass the
template kwarg / inject the no-think tag the model documents, or give actor lines
a minimal grammar (e.g. `root ::= [^<] .*` is not expressible in GBNF, but a rule
that forbids the thought-open token's first character, or simply a
`chat_format`/template override) — and add a startup self-test that generates one
actor line and asserts it is non-empty and free of thought markers.

### 4.6 One global engine, unknown concurrency safety — Medium

`pipeline._engine` is a module-level singleton shared by every Gradio session, and
`llama_cpp.Llama` is not thread-safe for concurrent generation on one context. Per-
event default concurrency in Gradio limits each *listener* to one run at a time,
but `start_case`, `submit_plea`, and a second user's events are different listeners
and can interleave. Two users on the Space at once → two concurrent
`create_chat_completion` calls on the same `Llama` object → undefined behavior
(the observed "index out of bounds" faults may even be this, not the prefix-reuse
bug). Rebuild action: a single global `threading.Lock` around engine calls (the
honest option on 2 vCPUs — true parallel inference is not possible anyway), or
`demo.queue(default_concurrency_limit=1)` plus documentation that the Space is
single-player-at-a-time.

### 4.7 Smaller runtime observations — Low

- **ZeroGPU scaffolding is dead weight on the shipped CPU path.** `import spaces`
  first-thing in `app.py`, the `@spaces.GPU` decorator, and the `libcudart`
  preload block in `_gpu_call` all exist for a GPU path the deployment no longer
  uses. They are documented as no-ops, but they are ~40 lines of misdirection in
  the most safety-critical file. The rebuild should delete them (keep TTS behind
  its own optional module if VoxCPM2 ever lands).
- **`casefile()` can fall through to `return None`** if the first attempt raises
  `JSONDecodeError` and the retry path is reached — actually the retry re-raises
  on the second failure, so the only real gap is stylistic (no explicit final
  raise), but `new_case` would crash opaquely on a `None`. Add the explicit raise.
- **Generation budgets are looser than the outputs.** GM decision `max_tokens=320`
  for a JSON that the field notes say settles in ~80 tokens; case file 600 for
  ~200–300; score 256 for ~60. The grammar usually terminates early so the cost is
  only on the truncation/failure path, but tighter caps (128/350/96) bound the
  worst case at 12 tokens/second decode.
- **`trace_corporate.json` is stray at the repo root** (an artifact of
  `share_trace.py`); move under `data/` or delete.
- **`turn_budget` is read from config, and difficulty is hard-pinned** to
  `DEFAULT_DIFFICULTY` in `ui.py` (`start_case` never exposes the difficulty
  picker). Either expose it or delete the dead branch.
- **TTS is a stub** (`tts_engine.voice_line` always returns `None`); the "Voices"
  radio option is therefore a silent no-op presented to the player. Hide it until
  it works.

---

## 5. Synthetic data generation findings (`training/teacher_datagen.py`, `training/director_datagen.py`)

### 5.1 The committed actor data is the stale, pre-fix shape — High

`data/style_*.jsonl` (the snapshot committed to the repo) still has the **old**
example shape:

```
system: "You are the JUDGE in a courtroom. Speak in dense corporate jargon… Hidden brief: a high-school chemistry teacher who…"
user:   "Open the hearing."
```

That is precisely the train/runtime mismatch the field notes describe fixing —
the runtime actor receives *no* hidden brief and a *stage direction* user turn.
`teacher_datagen.py` has been fixed to emit the new shape, but the committed data
(and, per the project memory, the trained style LoRAs currently served from
`BastienHot/buzzwords-style-loras`) predate the fix. **The actors in the live game
are very likely running on adapters trained against a distribution the runtime
never produces.** The actor evaluation (`evaluate.py`) would partially mask this:
it measures jargon density, which a mismatch-trained adapter can still deliver,
while the instruction-following half (obeying the stage direction) silently
degrades.

Rebuild action: regenerate all eight style datasets with the fixed generator,
retrain the eight adapters, re-gate, re-publish — and delete or clearly version
the stale `data/` snapshot so it cannot be reused by accident (Section 5.5).

### 5.2 Content diversity is capped at 12 professions × 8 fault archetypes — High (game design)

Every hidden truth in every dataset — actor and director alike — is drawn from
`PROFESSIONS` (12 entries) × `FAULT_ARCHETYPES` (8 entries). Tones, dispositions,
severity, and turn counts add surface variety, but the *answer space the player
plays against* is 96 combinations, and the director LoRA has only ever been
trained to emit case files for those 12 professions. At runtime the case file is
open-ended generation, but a rank-32 adapter trained on 12 jobs over ~hundreds of
examples will overwhelmingly reproduce the pool (occasionally with the "jargon-
bleed" mutations the memory notes record). A player's third game has a high chance
of repeating a profession; a player who reads the open-source repo knows the
entire answer key.

Rebuild actions, in increasing ambition:
1. **Widen the pools** to ~80–150 professions and ~25–40 fault archetypes (cheap:
   the teacher generates these too — one offline prompt with dedup and the same
   exclusion logic; human-skim the list once).
2. **Per-style exclusion needs to scale with the pool.** `STYLE_EXCLUDE` is
   hand-maintained for 2 of 8 styles today; with a bigger pool, derive exclusions
   from a domain tag on each profession (e.g. `{"airline pilot": "aviation"}`)
   instead of hand-listing.
3. Optionally, let the teacher *invent* the profession under the banned-domain
   constraint and validate it against the style's jargon bank (reject if any bank
   term appears in the profession). This restores open-endedness while keeping the
   smokescreen provable.

### 5.3 Leak filtering is exact-substring only — Medium

`_reject_reason` / `_leaks` check `profession.lower() in blob`. This catches the
profession string and its substrings but misses: synonyms ("baker" for *pastry
chef*, "doc" for a physician profession), single-token aliases, and decompositions
("the man who flies the plane"). The training-side mitigation (banned-word list in
the corrective retry) only bans tokens of the profession string itself. The
residual risk is real but bounded — the held-out leak metric came back 0 — and
semantic near-leaks are partially the *point* of the game (facts must hint).
Rebuild action: extend the banned list with a tiny hand-written alias map per
profession (one line each when the pool is generated), and run the same check at
runtime in the latch (Section 4.4). Do not attempt embedding-based leak detection;
it is over-engineering for this risk level.

### 5.4 One teacher, one prompt family — Low

All data comes from a single teacher (Gemma 4 31B FP8) under one prompt template
per task, temperature ~1.0. Stylistic monoculture is the known cost: the student
can only ever be as varied as the teacher's one voice. For a hackathon this is
acceptable (and the seeded term-bank injection fights phrase recycling well), but
note that the 80% top-1 agreement metric partially measures "sounds like Gemma",
not "directs well" — there is no second opinion anywhere in the pipeline. A cheap
widening, if time permits: two prompt phrasings per task, or two temperatures,
tagged in the manifest so their effect is measurable.

### 5.5 No dataset lineage or manifests — Medium

Datasets land in a Modal volume (and a partial snapshot in `data/`) as bare JSONL.
Nothing records: the base seed, n requested versus kept, reject-reason histogram,
teacher model+revision, generator git commit, or generation date. The
train/runtime-shape bug of Section 5.1 went unnoticed partly because a JSONL file
carries no provenance — you cannot tell a pre-fix file from a post-fix file
without reading examples. Rebuild action: every generation run writes
`<name>.manifest.json` next to the data (seed, counts, rejects, teacher id, git
SHA, prompt-shape version string), and training refuses datasets whose shape
version doesn't match the contract module's version (Section 10.1).

### 5.6 Director data composition is sensible but its ratios are folklore — Low

The mixed `director.jsonl` is games (1 casefile + ~12 decide examples each) +
scores + optionally upsampled standalone casefiles (`--casefile-extra 600` on a
300-game run). So decide examples outnumber casefile examples ~4:1 even after
upsampling, and the score task rides along at ~n examples. The multitask balance
was chosen by feel; the per-task gate results justify it after the fact (all three
tasks pass), so this is not a defect — but the rebuild should record the ratio as
a tunable in the manifest, because it is the first knob to turn if a sub-task
fails the gate after retraining on new pools.

---

## 6. Training pipeline findings (`training/finetune.py`, `training/director_finetune.py`)

### 6.1 The train/validation split leaks across sibling examples — High (metrics validity)

Both trainers do `ds.train_test_split(test_size=0.05, seed=42)` on the *exploded*
example rows. But the rows are not independent: an actor transcript explodes into
~8 examples sharing one hidden case, one tone, one set of injected jargon terms;
a director game explodes into 1 casefile + ~12 decide examples sharing the same
case file text (which appears verbatim inside every decide prompt). Random
row-level splitting puts siblings on both sides, so the "held-out" loss is partly
measured on near-duplicates of training rows. The reported perplexity (~2.0) is
therefore optimistic by an unknown amount, and `eval_loss`-based comparisons
between runs are noisy in a correlated way.

Fix (small): carry a `group_id` (transcript/game index) through `_to_examples` /
`_game_to_examples`, and split on unique group ids before exploding. This is a
~10-line change in each datagen script plus using `datasets`' filter instead of
`train_test_split`. Note the *benchmark* evals are unaffected (they use a
disjoint-seed file), which is why the shipped director is still trustworthy — the
leak only corrupts the in-training metrics.

### 6.2 Loss is computed over the whole sequence, prompts included — Medium

`SFTTrainer` is fed `apply_chat_template(..., tokenize=False)` full conversations
with no completion-only masking (`train_on_responses_only` in Unsloth, or
TRL's completion-loss collator). The assistant targets are short (~25 words for
actors, compact JSON for the director) while system+user prompts are long and
highly repetitive across examples — so a large fraction of gradient signal goes
into memorizing the fixed prompt boilerplate. Three concrete costs: (a) capacity
of a rank-32 adapter spent on text the model never needs to generate, (b)
perplexity numbers dominated by easy prompt tokens (further inflating the
optimism of Section 6.1), and (c) a subtle inference-time risk that the model
becomes *too* anchored to byte-exact prompts (brittleness if a prompt is ever
reworded — which cuts against the contract-module future where prompts evolve
deliberately). Rebuild action: enable response-only loss; it is one argument in
Unsloth and makes both training and its metrics mean what they claim.

### 6.3 The chat template's thinking behavior during training is unverified — Medium

Evaluation code passes `enable_thinking=False` to `apply_chat_template`; the
*training* `fmt()` does not pass it at all. If MiniCPM5's template defaults to
inserting a thinking scaffold (or an empty think block) when the flag is absent,
the trained-on token sequence differs from both the eval sequence and the runtime
llama.cpp template rendering. This may be entirely benign — but nobody has looked
at one rendered training example end-to-end. Rebuild action: print one fully
rendered, tokenized example per run into the manifest, and assert the absence of
thought-channel tokens. (This is the same class of bug as Section 5.1, one level
down the stack: shape mismatch, silent.)

### 6.4 Quantization mismatch between training and serving is unvalidated — Low

Adapters are trained with QLoRA against the bnb-4bit quantized base, then applied
at inference onto a *different* 4-bit quantization (GGUF `Q4_K_M`). This is
common practice and usually fine, and the end-to-end gate de-risks it implicitly
(the gate runs the real GGUF+LoRA stack). `docs/ARCHITECTURE.md` lists "LoRA
quality on Q4 vs Q8 base" as an open point that was never formally closed; the
honest statement for the rebuild is: *validated implicitly by the llama.cpp gate,
never measured in isolation* — and that is acceptable. Only revisit if the gate
ever shows the transformers-eval passing while the llama.cpp gate fails.

### 6.5 Hyperparameters and reproducibility — Low

Rank 32 / alpha 32 / dropout 0 across all seven projection matrices, LR 2e-4
cosine, 2 epochs, effective batch 16 — all reasonable defaults for a 1B LoRA and
"identical config C across stage 1 and 2" is correctly enforced for the fork.
Gaps worth closing cheaply in the rebuild: no training seed is set (two runs of
the same command produce different adapters — awkward for the before/after
comparisons this project loves); `train_metrics.json` records only final
eval_loss/perplexity (add the LoRA/TRAIN dicts, dataset manifest reference, and
git SHA); and the stage-1-helps-stage-2 claim of the curriculum has never been
ablated (run one style with and without `init_adapter=legal_base` — a single
afternoon — and either keep the curriculum with evidence or simplify to
single-stage; on ~1,600-example style datasets it is genuinely plausible that
stage 1 adds nothing).

### 6.6 The director's optional `legal_base` warm-start is untested — Low

`director_finetune.py` supports forking from `legal_base` but the shipped
director was trained from scratch (per the README run order). Fine — just delete
or de-document one of the two paths in the rebuild rather than carrying both
untested.

---

## 7. Evaluation and gating findings (`training/evaluate.py`, `director_evaluate.py`, `director_gate.py`)

### 7.1 The end-to-end gate is a single trajectory — the project's own forbidden pattern — High

`director_gate.py` is the only test that runs the *real* stack (llama.cpp + GBNF +
the GGUF LoRA), which makes it the most load-bearing check in the repo — and it
plays exactly **one** game, in **one** style/difficulty (`aviation/normal`,
hard-coded), with stubbed actor lines, and binarizes on that sample ("spot-on
guess scores ≥ 80", "all 3 speakers used"). The field notes' first lesson is "a
single greedy trajectory lies"; this gate is a single (near-greedy at temperature
0.7-per-call) trajectory. A directionally fine director will fail it ~sometimes
and a subtly broken one will pass it ~sometimes. Rebuild action: loop the existing
game function over N ≥ 10 games across all styles and both difficulty extremes,
aggregate (speaker entropy, leak count, wrap-within-budget rate, score means per
bucket), and threshold the aggregates. The code is already structured so this is
a for-loop plus a counters dict.

### 7.2 Nothing measures whether the game is *solvable* — High (the missing metric)

Every existing metric checks the machinery (valid decisions, balanced speakers,
no leaks, calibrated scoring). None checks the *product*: can a player who reads
the final transcript actually infer the profession and fault? The game can pass
every gate while being unwinnable (facts too oblique, actors ignoring stage
directions) or trivially easy (semi-leaks everywhere). There is a free, automatic
proxy: feed the finished transcript (player's view only — no case file) to the
teacher, ask it to guess profession + fault, and score that guess with the
*deterministic* bucket logic (not the learned scorer). Track "teacher-solver
score" per difficulty: it should land in a band (say 40–80 at normal — guessable
but not given away) and should *order* correctly across easy/normal/hard. This
single metric closes the loop between "the pipeline works" and "the game is fun",
and it doubles as the difficulty-calibration tool the architecture document
lists as an open point. Strongly recommended as a first-class gate in the
rebuild.

### 7.3 The learned scorer is validated only on synthetic, in-pool guesses — Medium

Score calibration was measured on guesses *constructed from the same
profession/fault pools* with template phrasings (`_make_guess` buckets). Real
players type free text: misspellings, partial answers ("something with planes?
bribery?"), answers at different abstraction levels ("a teacher" vs "a high-school
chemistry teacher"). The 100/14/86 separation says nothing about that
distribution. Cheap rebuild actions: (a) add noisy variants to the eval buckets
(typos, hedges, fragments — templated, no humans needed); (b) write 20 guesses by
hand once and keep them as a frozen regression set; (c) consider blending the
learned score with a deterministic component (e.g. token overlap on the
profession) so a scorer hallucination can't award 95 to a wrong answer — the
deterministic part also gives the reveal screen something explainable.

### 7.4 Teacher-agreement has no self-agreement baseline — Low

The headline "80% top-1 agreement with the teacher" lacks the denominator that
makes it interpretable: how often does the *teacher agree with itself* when
resampled on the same context? If teacher self-consistency is ~85%, the student
is near-ceiling; if it is ~95%, there is headroom. One extra eval column
(resample the teacher per held-out context, before sleeping the GPU) turns the
number into a real distillation-efficiency measure. Same note for
prosecutor→defense rates: the teacher's 0.65 *is* reported — good — keep that
pattern for every reference metric.

### 7.5 Actor evaluation is gameable by term-stuffing — Low

`evaluate.py`'s style lift (bank-term coverage/density) plus a 1.5× density
threshold and own-bank specificity is a clever label-free yardstick, and the
distinct-word-ratio fluency check exists — but it has no teeth (it is computed
and printed, never gated). A LoRA that collapses into jargon salad passes. Two
cheap hardenings: gate on `distinct_ratio` not regressing more than ~10% versus
base, and add a stage-direction-following check (the stage directions in the eval
set name concrete things — "altered records", "timeline" — substring-check that
the response engages with the cue's content words at some minimum rate). Note
also the current eval uses only 16 prompts × 2 repeats; bump to ≥ 50 samples per
style now that generation is the cheap part.

### 7.6 Gate thresholds are magic numbers — accepted

`GATE`'s thresholds (leak ≤ 0.1, defense share ≥ 0.15, separation ≥ 45, …) are
judgment calls. That is fine — gates exist to catch regressions, not to be
publishable — but the rebuild should move them into one constants block with a
one-line rationale each, because today they are duplicated reasoning living
inside a lambda dict.

---

## 8. Inference on 2 vCPUs — current state and optimization plan

### 8.1 Where the time actually goes (budget per game)

With the AVX2 build (~60–80 tokens/second prefill, ~12 tokens/second decode) and
the current code, one `normal` game costs roughly:

| Step | Prefill | Decode | Est. time |
|---|---|---|---|
| Case file (once) | ~250 tok | ~150–250 tok | 15–25 s |
| GM decision (×12) | ~600–800 tok *(window re-prefill, §4.2)* | ~80 tok | 14–18 s each |
| Actor line (×12) | ~120 tok | ~30–50 tok | 4–6 s each |
| Score (once) | ~150 tok | ~60 tok | ~6 s |

Total pre-generation: **4–5 minutes** behind a progress bar before the player sees
the first line. The single biggest line item is the GM decision's re-prefilled
transcript window, and the second is the serialization of all beats before play.

### 8.2 Serving: adopt `llama-server`, or knowingly keep the binding — High

The architecture document specifies `llama-server` with one resident base,
adapters loaded with `--lora-init-without-apply`, and per-request activation by
scale. The app instead uses the `llama-cpp-python` binding with one `Llama`
instance per adapter. Costs of the current choice, all already paid at least
once: double base residency (§4.1), the Python-side prefix-reuse bug and its
safe-mode workaround (§4.2), full actor reload on style switch, and no
per-request adapter switching. Benefits: no subprocess management, in-process
grammar objects, simpler error handling.

Recommendation for the rebuild: **run `llama-server` as a managed subprocess**
(started by `app.py`, health-checked by `preflight`) with the base + director +
style adapters all registered, and talk to it over HTTP from the pipeline. What
this buys, concretely: one copy of the base in RAM; slot-based KV caching that
works (and per-slot cache retention across requests — pin the GM to one slot, the
actor to another, so each keeps its own warm prefix); per-request
`lora` scale switching with no reload; and the option to set
`--cache-reuse` for prefix tolerance. The bench memory notes already validated
this exact setup on the bench Space (`llama-server` + No-Think + `--lora`). The
binding path can remain as a local-dev fallback, but the Space should ship the
server. If the rebuild instead keeps the binding, write down the accepted costs
in `ARCHITECTURE.md` so the divergence is a decision, not drift.

> **Clarification (author audit):** the concern that an unmerged LoRA "cannot run
> under AVX2 and falls back to pure CPU" is unfounded. AVX2 is a compile-time
> property of the whole ggml library; since llama.cpp's LoRA refactor (2024, PR
> #8332) an unmerged adapter is applied as extra low-rank matmuls per layer,
> executed by the same vectorized kernels as the base (adapter tensors in F16 —
> unquantized, but fully SIMD). Measured cost at rank 32 on a 1B is a few percent
> per token, not a tier change. No merge is required for performance.

### 8.3 Hide the latency instead of only shrinking it — High (UX)

The player clicks through pre-generated beats at reading speed (~5–15 seconds per
line) — which is *longer than it takes to generate the next beat*. Pre-generating
everything up front buys nothing except the 4-minute wall at the start.
Pipelined generation — generate beat 1, show it, and generate beat N+1 in a
background thread while the player reads beat N — reduces time-to-first-line to
~30–40 seconds (case file + one beat) and makes every subsequent "Continue" feel
instant, with zero model-side changes. The session model already separates
generation (`turn`) from playback (`playback_idx`), so the seam exists; the work
is a worker thread + a "next line ready" flag, with the existing pre-gen loop as
fallback. This is the highest player-perceived-value change in the whole
rebuild.

### 8.4 Prompt-structure and decoding economies — Medium

- **Stable-prefix GM prompt** (append-only transcript, §4.2): turns the 600–800
  token re-prefill per decision into ~50–100 new tokens — saves ~8–10 seconds per
  beat on its own, the largest pure-speed win available.
- **Pin threads explicitly**: `n_threads=2, n_threads_batch=2` (or `--threads 2`
  on the server). Inside a Space container, `os.cpu_count()` may report the
  host's cores; oversubscribing 2 vCPUs with 16 threads measurably *slows*
  llama.cpp. Verify once on the Space and pin.
- **Tighten `max_tokens`** (§4.7): bounds the worst case at 12 tokens/second.
- **KV cache type `q8_0`** for K and V halves the KV memory with negligible
  quality cost at these context lengths — only relevant if the 8192-context GM
  plus actor plus OS pressure ever matters; low priority on 16 GB.
- **Do not pursue speculative decoding**: the memory notes already concluded the
  ~130K vocabulary (embedding+head = 401M of the 1.08B parameters) caps the
  payoff; agreed, and a draft model would double residency.
- **Quantization**: `Q4_K_M` is the right default. If chasing decode speed later,
  benchmark the `Q4_0`-with-runtime-repacking path (llama.cpp repacks to
  AVX2-friendly layouts) — it sometimes beats K-quants on CPU prefill — but only
  after the structural wins above; expected gain is ~10–20%, not a tier change.

### 8.5 Deployment is in good shape — keep

The Docker Space setup is correct and well-commented: AVX2/FMA/F16C source build
with `--no-binary`, test/example/server sub-targets disabled and parallel level 2
to fit the build budget, non-root uid 1000, weights pulled from the Hub at
startup (not committed), GitHub Action mirroring `main`. Two small notes: if the
rebuild adopts `llama-server` (§8.2), drop `-DLLAMA_BUILD_SERVER=OFF` and ship
the server binary from the same build; and consider HF persistent storage or a
baked-in base GGUF layer if cold-start downloads ever become the bottleneck
(today ~0.9 GB from the Hub is fast; fine as-is).

---

## 9. App integration findings (`app.py`, `ui.py`, `scene.py`)

The Gradio layer is clean and appropriately small. `gr.Walkthrough` mapping the
four game phases to steps removes show/hide juggling; the per-session
`GameSession` in `gr.State` is the right state boundary; error paths bounce the
player back to Charges with a readable card instead of crashing; and the
"close the hearing early if ≥ 2 beats exist" recovery is a good player-first
call. Specific items for the rebuild:

- **The over-eager-wrap floor is a quiet design rule** —
  `floor = max(4, budget // 2)` in `pipeline.next_turn` means an `easy` (8-beat)
  game can legitimately end at beat 4. Combined with §7.2's solvability metric,
  check that 4 beats can ever contain enough clues to guess from; if not, raise
  the floor or feed it from difficulty.
- **Difficulty is dead UI-side** (§4.7): `start_case` hard-codes
  `DEFAULT_DIFFICULTY`; the `TURN_BUDGET_BY_DIFFICULTY` map and the director's
  difficulty-conditioned training are unreachable from the front end. Expose the
  picker — the training data already covers all three settings.
- **The "Voices" option is a no-op presented as a feature** (§4.7) — hide until
  VoxCPM2 is wired.
- **Engine concurrency** (§4.6) — serialize engine access before sharing the
  Space link widely.
- **The progress narrative is good UX** — keep the streamed loading-card pattern
  through the pipelined-generation refactor (§8.3); it becomes "beat ready"
  notifications instead of a monolithic bar.

---

## 10. Rebuild blueprint

Ordered so that each phase de-risks the next; the contract module comes first
because everything else depends on it.

### 10.1 Phase 0 — one contract module (the keystone)

Create `buzzwords/contracts.py` owning, as importable constants/functions:
- the three GBNF grammars;
- every system prompt (`_CASEFILE_SYS`, `_GM_SYS`, `_SCORE_SYS`, `ROLE_SYS`,
  `_ACTOR_RULES`) and the user-turn builders (`gm_prompt`, `actor_prompt`,
  `score_prompt`, `casefile_prompt`);
- the enums (speakers, beats, intensities), the difficulty→budget map,
  `WRAP_PRESSURE_AT`;
- a `SHAPE_VERSION` string, bumped on any change.

Training scripts import it the same way `teacher_datagen.py` already imports
`jargon_banks` (`add_local_python_source`). Datagen stamps `SHAPE_VERSION` into
manifests; training and evaluation assert it. This deletes four hand-synced
copies and makes the project's own "train shape = runtime shape" rule a property
of the code rather than of vigilance. Also move the profession/fault/tone pools
here (or a sibling `pools.py`) — runtime and training currently each carry their
own copies of constants like `WRAP_PRESSURE_AT = 2`.

### 10.2 Phase 1 — data regeneration on widened pools *(updated after author audit)*

1. Widen pools (§5.2): teacher-generate + human-skim ~100 professions, each with
   a domain tag **and 3–5 domain-matched fault archetypes** (a tattoo artist gets
   tattoo-shop faults, never aviation ones — see the §13.4 bleed failure). Derive
   `STYLE_EXCLUDE` from the domain tags.
2. **Casefile generation moves to "code samples the truth"** (decision, §12):
   the runtime samples profession + one of its matched faults from the pools
   (with the jargon's domain excluded) — smokescreen and profession/fault
   coherence become guarantees, not learned behaviors. The director model
   generates only the oblique `facts`, grammar-constrained and leak-validated,
   with a shipped pre-validated casefile bank as the retry-exhausted fallback.
   The datagen `casefile` task shrinks accordingly (facts-only targets).
3. **Difficulty is removed everywhere** (decision, §12): one turn budget
   (suggest 10–12), no difficulty axis in prompts, data, or UI. This simplifies
   the contract and removes a conditioning variable the player never sees.
4. Add `group_id` and manifests (§5.5, §6.1) to both datagen scripts.
5. **Actor examples gain dialogue context** (decision, §12): the user turn
   becomes `last 1–2 public transcript lines + stage direction + the selected
   fact text (§13.5) + intensity`. Decide this shape *before* generating — it is
   the contract.
6. Regenerate `legal_generic`, all eight styles, and the director set; regenerate
   a disjoint-seed held-out slice; **delete the stale `data/*.jsonl`** from the
   repo (replaced by the versioned, manifested release sets of Phase 5).
7. Keep: seeded specs, jargon-bank injection, few-shots + banned words +
   corrective retry, smokescreen by construction.

### 10.3 Phase 2 — retraining with honest metrics *(updated after author audit)*

1. Group-wise split (§6.1), response-only loss (§6.2), fixed training seed,
   `enable_thinking=False` passed in the training `fmt()` (§6.3), enriched
   `train_metrics.json` (§6.5), one rendered example printed and checked for
   thought tokens.
2. **Train LoRA on the bf16 base, not QLoRA** (decision, §12): exact
   quant-matching with the GGUF serving base is not possible (bitsandbytes 4-bit
   and GGUF `Q4_K_M` are different formats), but a 1B trains comfortably — and
   faster — in bf16 on the A100 already in use, removing training-side
   quantization noise entirely. The end-to-end llama.cpp gate continues to
   validate the serving-side quant.
3. Ablate stage 1 on one style (§6.5); keep or drop the curriculum on evidence.
   Delete the unused director `legal_base` warm-start path (§6.6).
4. Retrain all eight actors (mandatory — stale, §5.1) and the director.
5. **Second director round on self-rollouts if the on-policy gate fails**
   (§13.6): roll out games with the trained student actors, have the teacher
   label the correct decision at each *student-generated* context, and fine-tune
   the director on that mixture. This is the principled fix for the exposure
   bias demonstrated in §13; the deterministic guards of §13.3 are the cheap
   insurance shipped regardless.

### 10.4 Phase 3 — the gate suite *(updated after author audit)*

1. Actor bench: ≥ 50 samples per style, fluency gate with teeth,
   stage-direction- and fact-engagement checks (§7.5, §13.5).
2. Director bench: keep, add the teacher self-agreement baseline (§7.4). The
   noisy/handwritten scorer guess sets (§7.3) are **deferred** (decision, §12) —
   not on the critical path.
3. **End-to-end gate on full self-rollouts with the real actors — no stubs**
   (§7.1 + §13.6): N ≥ 10 llama.cpp games across styles, aggregating
   (a) speaker-transition-matrix distance to the teacher's (sequencing, not just
   marginals), (b) max consecutive same-speaker run, (c) cross-line n-gram
   repetition, (d) role-consistency violations, (e) leak count, (f) wrap-within-
   budget rate, (g) **the teacher-solver solvability score** (§7.2) — the
   headline metric.
4. Ship nothing that hasn't passed 3.

### 10.5 Phase 4 — runtime integration *(updated after author audit)*

1. Serving: `llama-server` subprocess, ONE resident base + director/style
   adapters switched per request by scale (no merge needed — see the §8.2
   clarification), pinned slots for GM/actor prefix caches, threads pinned to 2.
2. Stable-prefix GM prompt with append-only transcript (§4.2).
3. **No pre-generated games** (decision, §12): pipelined beat generation —
   show beat 1 as soon as it exists, generate beat N+1 while the player reads
   beat N (§8.3).
4. Deterministic sequencing guards (§13.3): beat→allowed-speaker compatibility
   map, no speaker three times consecutively, defense guaranteed by mid-game.
   Cheap, in code, regardless of how well retraining goes.
5. Runtime-sampled casefile (§10.2 item 2) + the `fact` channel into actor
   prompts (§13.5); fix the `casefile()` silent-`None` path with an explicit
   raise (§4.7).
6. Explicit no-think on the actor path + startup self-test (§4.5); add
   `repeat_penalty` and a cross-line repetition check with one retry (§13.4).
7. Engine lock / queue serialization (§4.6); **delete the ZeroGPU scaffolding
   and the TTS stub + "Voices" option** (decision, §12); tighten token budgets;
   **remove the difficulty setting** (decision, §12).
8. Anti-spoiler latch (§4.4): **deferred to post-MVP** (decision, §12) — easy to
   add later; worthless if the core loop doesn't work first.

### 10.6 Phase 5 — release deliverables *(new, per author audit)*

The hackathon asks for shared agent traces; share them at meaningful scale:
- **Agent traces:** a pool of 50–100 full game traces (extend `share_trace.py`
  to batch; one JSON per game, covering all styles), not the single
  `trace_corporate.json` currently sitting at the repo root (delete/move it).
- **Datasets:** the regenerated training sets *with their manifests* (§5.5) —
  seed, counts, reject histograms, teacher id, prompt-shape version. The
  manifest-per-dataset convention is the open-source story.
- **Adapters:** the retrained director + eight style LoRAs, with gate results in
  the model cards.
- **Final report note** (decision, §12): one paragraph acknowledging the
  single-teacher / single-prompt-family limitation (§5.4) as accepted hackathon
  scope.

### 10.7 Explicitly out of scope (agreed non-goals)

Embedding-based leak detection; speculative decoding; multiple teachers;
reinforcement learning or preference tuning for the director; merging adapters
into the base; GPU paths of any kind; TTS (removed with its stub). Deferred to
post-MVP rather than rejected: the anti-spoiler latch (§4.4), scorer
noise-robustness sets (§7.3), profession alias maps for leak checking (§5.3).

---

## 11. Open questions — all resolved in the author audit (§12)

The four questions originally posed here were answered on 2026-06-10; the
decision log in §12 records the rulings, and §10 has been updated in place.
Summary: (1) actors get dialogue context — yes; (2) the casefile truth is
sampled in code, the model writes only the facts; (3) difficulty is removed
entirely; (4) `llama-server` is the serving target.

---

## 12. Author audit — decision log (2026-06-10)

Rulings from the author's review of Sections 4–9. ✅ accepted as written,
🔶 accepted with modification, ⏸ deferred (post-MVP), ❌ dropped.

| Finding | Ruling | Notes |
|---|---|---|
| §4.1 single resident base | ✅ | Author's LoRA/AVX2 concern resolved — see §8.2 clarification: unmerged adapters run vectorized; no merge needed. |
| §4.2 cache / stable-prefix prompt | ✅ | |
| §4.3 actor dialogue context | 🔶 | Original intent was "the director's stage direction *is* the context"; accepted that public transcript lines add coherence the 1B director can't compress into one cue. Shape decided before Phase 1 datagen. |
| §4.4 anti-spoiler latch | ⏸ | Correctly cheap, but not on the critical path: "if the rest doesn't work it becomes useless". Post-MVP. |
| §4.5 explicit no-think for actors | ✅ | Investigate the most robust mechanism (template kwarg vs. tag injection) during Phase 4. |
| §4.6 engine serialization | ✅ | Queue/lock to protect the Space. |
| §4.7 ZeroGPU + TTS removal | ✅ | Both removed, including the "Voices" UI option. |
| §4.7 `casefile()` None path | ✅ | Explicit raise. Largely mooted by runtime-sampled casefiles (below). |
| §4.7 trace file | 🔶 | Traces are a hackathon deliverable — share a *pool* of 50–100, plus datasets and adapters (new Phase 5, §10.6). |
| §4.7 / §9 difficulty | ❌→removed | Difficulty setting deleted everywhere (UI, config, prompts, data). One turn budget. |
| §5.1 regenerate stale actor data | ✅ | Old `data/*.jsonl` disposed of. |
| §5.2 widen pools | 🔶 | Plus: faults become *domain-specific and matched to their profession* — directly motivated by the §13.4 bleed. |
| §5.3 alias-map leak hardening | ⏸ | Same logic as the latch: post-MVP. |
| §5.4 single teacher | ✅(accepted risk) | Note it in the final report (Phase 5). |
| §5.5 manifests per dataset | ✅ | "That is true open source." |
| §6.1 group-wise split | ✅ | |
| §6.2 response-only loss | ✅ | |
| §6.3 `enable_thinking=False` in training | ✅ | |
| §6.4 train/serve quant match | 🔶 | Exact match impossible (bnb-4bit ≠ GGUF Q4_K_M; can't train on GGUF). Adopted instead: **bf16 LoRA training** (1B is small; faster than QLoRA, zero training-side quant noise); serving quant stays validated by the e2e gate. |
| §6.5 seed + ablation | ✅ | Stage-1 ablation to be run quickly. |
| §6.6 unused warm-start path | ✅ | Delete and de-document. |
| §7.1 multi-game gate | ✅ | |
| §7.2 teacher-solver solvability | ✅ | "Interesting indeed" — promoted to headline gate metric. |
| §7.3 scorer noise sets | ⏸ | Not mandatory now. |
| §7.4 teacher self-agreement | ✅ | |
| §7.5 actor bench hardening | ✅ | More samples; metrics toward diversity and *case engagement*. |
| §8.3 hide latency / no pregen | ✅ | "Let's avoid pregenerated games." |
| §8.4 inference economies | ✅ | At reviewer's discretion: stable prefix, pinned threads, tight budgets adopted; KV-q8_0 only if memory pressure appears; no speculative decoding, quant stays Q4_K_M. |

One new directive from the audit, triggered by the live transcript below: data
generation and training must teach **sequencing and on-policy behavior**, not
just marginal distributions — see §13.6. This supersedes the earlier project-
memory conclusion that a deterministic speaker guard was unnecessary.

---

## 13. Live-game post-mortem: the degenerate transcript

A real game played on this version (aviation jargon; hidden truth revealed as
*"tattoo artist — ignored a clear flight rule to save time on the ground"*)
produced, in seven beats: judge, prosecutor ×3 consecutively, defense, judge,
prosecutor — with the prosecutor at one point delivering the defense's plea
("My client is deeply remorseful…"), the judge speaking as an advocate ("why my
client has been on the taxiway"), the phrase "your silence" appearing three
times, and a revealed fault that is itself aviation-flavored despite the
profession being a tattoo artist. This one transcript exhibits five distinct
failure modes, and none of them is random noise — each maps to a specific gap
already identified above. This section is the diagnosis; the fixes are wired
into the Phase plan (§10).

### 13.1 Why every benchmark passed while this happened

The held-out director benchmark asked: *given a teacher-written context, does
the student pick what the teacher would?* It answered yes (80% agreement,
balanced speakers) — and that answer was true. But at runtime the director
conditions on **its own actors' lines**, which are weaker, repetitive, and (with
the stale adapters of §5.1) off-distribution. Each slightly-off decision
produces a slightly-worse line, which pushes the next decision context further
from anything seen in training. This compounding is *exposure bias*, the classic
gap between teacher-forced training and autoregressive deployment. The eval was
not wrong; it measured the wrong regime. The earlier conclusion in the project
memory — "the speaker-rotation guard is NOT needed" — was valid only in the
teacher-forced regime and is hereby reversed for the on-policy one.

### 13.2 Failure: role/persona mismatch (prosecutor pleads, judge advocates)

The decision grammar permits any `next_speaker` × `beat_type` pair. The teacher
data was coherent, so compatible pairs dominate the learned distribution — but
at temperature 0.7 over drifted contexts, the student samples incompatible ones
(e.g. `prosecutor` + `plea`), and the actor then follows the *content* of the
stage direction regardless of its system-prompt role, because the stage
direction is the only content it has (and the stale actors were trained with a
different persona-anchoring shape anyway). **Fix (deterministic, in code):** a
beat→allowed-speakers compatibility map applied after the GM decision —
`plea`/`objection` → defense, `charge`/`evidence`/`cross_examine` → prosecutor,
`opening`/`closing` → judge, the rest free. If the sampled pair is
incompatible, remap the speaker (keep the beat). This costs nothing and removes
the most jarring failure outright; retraining reduces how often it fires.

### 13.3 Failure: degenerate sequencing (prosecutor ×3, defense ×1 in 7)

Direct exposure-bias symptom (§13.1). Layered fixes, cheapest first:
1. **Deterministic guards in `pipeline.next_turn`** (ship regardless): never the
   same speaker three times consecutively; the defense must have spoken at least
   once by the budget midpoint; both enforced by remapping the speaker while
   preserving the rest of the decision. The Game Master keeps owning beat,
   intensity, stage direction, and wrap — the guard is a seatbelt, not a
   co-driver (this is exactly the v1 "deterministic latch" philosophy that was
   deferred, now reinstated with evidence).
2. **Gate on self-rollouts** (§10.4): the transition-matrix distance and
   max-run-length metrics over N full games with the real actors would have
   flagged this transcript class immediately.
3. **On-policy distillation round if the gate fails** (§13.6).

### 13.4 Failure: smokescreen broken at the source (aviation fault for a tattoo artist)

"Ignored a clear flight rule" is the *jargon's* domain bleeding into
`fault_plain` — the intermittent "jargon-bleed" watch-item from the project
memory, now observed in production where it ruins the reveal (the player is
told an incoherent truth). Root cause: the 1B was asked to *invent* the truth
under a constraint it only ever saw satisfied, never enforced. **Fix
(structural, decided in §12):** the truth is no longer generated at all —
runtime code samples profession + a domain-matched fault from the curated pools
(jargon's domain excluded), and the director only writes the oblique facts,
which are validated (leak substring check) with retry and a shipped fallback
bank. The profession/fault pairing being matched *by construction* (§10.2)
makes this failure unrepresentable rather than unlikely.

### 13.5 Failure: repetition and zero case content (unsolvable game)

"Your silence" ×3 and "positive rate of climb" ×2 happen because each actor
line is generated blind — no actor ever sees a previous line (§4.3), so nothing
can avoid repeating it. Worse, *no concrete fact ever surfaces*: the only
channel from the case file to the player runs through a one-sentence stage
direction written by a 1B, and in this game that channel carried nothing. The
transcript is therefore unsolvable — the teacher-solver metric (§7.2) would
score ~0 — and the final reveal feels arbitrary, which is the worst possible
player experience.

Fixes: (a) actor prompts include the last 1–2 public lines (§12 decision) and a
runtime `repeat_penalty` plus a cross-line n-gram check with one retry;
(b) **add a `fact` channel to the decision schema**: the GM decision gains a
`fact_index` field (an enum over the case file's facts, or null), and the
runtime passes the *verbatim oblique fact text* into the actor prompt alongside
the stage direction. The facts are already oblique and leak-checked, so this
reintroduces the v0 "clue economy" in its lightest possible form — the code can
even force the unused facts out before the budget ends, guaranteeing
solvability material reaches the player. The actor and director datasets must
be generated in this shape (the teacher's game prompt already asks it to weave
facts; it now labels *which* fact per turn).

### 13.6 The general lesson for datagen and training (author directive)

The pipeline validated *marginal* statistics (speaker shares, transition rates
as aggregates) on teacher-forced contexts. The rebuild must treat the director
as a **policy evaluated on its own rollouts**:

1. **Data:** keep teacher games as the base SFT set, but add a second slice
   where the *contexts* come from student self-rollouts (real student actors,
   real student decisions) and the *targets* come from the teacher labeling the
   correct decision at each such context — a single lightweight DAgger-style
   iteration. This is the canonical correction for exposure bias and directly
   implements "teach the model how to act during inference". Run it only if the
   on-policy gate fails after the other fixes; the guards may make it
   unnecessary.
2. **Evaluation:** every gate that matters runs on self-rollouts (§10.4) and
   measures *sequences* — transition matrices, run lengths, beat
   progressions (opening early, closing late, evidence in between), repetition
   across lines — not just per-decision agreement.
3. **Code:** deterministic guards make the worst sequences unrepresentable
   regardless of what the model learned (§13.2–13.3). Learned behavior is for
   quality; code is for invariants. That division — already used for the
   smokescreen — is the project's most reliable design pattern, and §13.4
   extends it to the casefile truth itself.

---

*Sections 12–13 added after the author audit of 2026-06-10; §10 was revised in
place the same day. This document is the rebuild baseline.*

*Solvability iteration (2026-06-11, SHAPE 2.1): the first e2e gate run passed all
eight machinery checks (guard triggers 0/117 beats, repetition 0, leaks 0, wrap 1.0,
fact coverage 1.0, scorer separation 87) but scored **solvability 0/12** — §7.2's
predicted failure, caught by the metric built for it. Layer diagnosis: facts-only
solver baseline = 21.7 (4/12 job hits) vs transcript-only = 0 → the actors' metaphor
re-encoding (skin→"fuselage", waiver→"manifest") is the intended aesthetic but
destroys all anchors; oblique facts × oblique acting = unwinnable. Fixes: (1) the
EVIDENCE DOCKET — released facts surface to the player in the UI as exhibits (this
was always the clue-economy intent; pure UI, no retraining); (2) FACTS_SYS and the
teacher prompts now require facts that collectively let a sharp guesser NAME the job
(smoke-verified: "gripping arms / payout frequency / prize chute" → arcade owner);
(3) the gate's solver now reads the true player view (docket + hearing). Director
regen + retrain under SHAPE 2.1 in flight. Other findings the gates produced today:
teacher self-agreement on decide = 0.64, so the student's 0.86 agreement is ABOVE the
teacher's own consistency ceiling — distillation saturated; the closed jargon-bank
yardstick under-measures open-vocabulary styles (scifi data itself contains bank
terms in only 4% of lines) — documented in evaluate.py, REVIEW-by-inspection.*

*Implementation status (2026-06-10): Phases 0–5 are implemented in code —
`buzzwords/contracts.py` + `pools.py` (66 professions × 3 matched faults),
`engine.py` (llama-server), guarded `pipeline.py` with background generation,
rewritten datagen/training/gate scripts, batch `share_trace.py`, and
`tests/test_game_logic.py` (13 tests, passing). Data regeneration, retraining,
and the gate runs on Modal are the remaining (paid, user-run) steps — see
`training/README.md` for the run order.*

---

*Outcome (2026-06-12): the SHAPE 2.1 chain passed the full gate — **9/9** (guards
1.7% of beats, repetition 0.9%, leaks 0 after fixing a word-boundary false positive
in the detector itself, wrap 1.0, fact coverage 1.0, scorer separation 85,
solvability 31.75 in-band with 58% exact-job hits). Adapters and 64 agent traces
published; the Space is live and play-verified. Post-ship playtest fixes: empty-plea
deterministic zero, sentence-cased reveal, the evidence-board UI, full
pre-generation behind the progress bar. This document is now a historical record;
the narrative version is `docs/FIELD_NOTES.md`.*

*End of review. Companion documents: `docs/ARCHITECTURE.md` (design history),
`docs/FIELD_NOTES.md` (the field log / blog post), `training/README.md` (run order).*
