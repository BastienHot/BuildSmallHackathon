"""Benchmark a style LoRA: does the actor speak the style AND engage the case? (Modal GPU)

Compares vanilla MiniCPM5-1B against base+actor_<style> on the EXACT runtime prompt shape
from buzzwords.contracts (role+style system; context lines + fact + stage direction user),
with NO human labels — the jargon banks are the yardstick. Hardened per REBUILD_REVIEW.md
§7.5: >= 48 samples per style, a fluency gate WITH TEETH, and an ENGAGEMENT metric (does
the line actually pick up the fact / stage-direction content, or is it jargon salad?).

  1. Style lift   -- target-bank coverage & density, base vs LoRA (>= 1.5x density).
  2. Specificity  -- LoRA output peaks on its OWN bank across all banks.
  3. Fluency      -- distinct-word ratio must not regress > 10% vs base (collapse guard).
  4. Engagement   -- informational only (oblique-by-design defeats literal overlap).

KNOWN YARDSTICK LIMIT: open-vocabulary styles (scifi technobabble, politics spin) score
low on a closed bank even when the register is clearly right — measured 2026-06-11: the
scifi TEACHER DATA itself contains bank terms in only 4% of lines, yet sample output is
unmistakably in-style ("a fraudulent phase-shift of the chronos-leech protocol"). For
those styles REVIEW + human inspection of the printed samples is the verdict; the e2e
gate's teacher-solver covers decodability.

Run: modal run training/evaluate.py --style corporate     (or no flag = all 8)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root -> buzzwords pkg

from buzzwords import contracts, pools
from jargon_banks import JARGON

BASE_MODEL = "openbmb/MiniCPM5-1B"
N_SAMPLES = 48   # per style (8 cues x 3 roles x 2 seeds)

STAGE_DIRECTIONS = [
    "Open the hearing and set a stern tone.",
    "Press the defendant on the altered records; cite the discrepancy.",
    "Object to the prosecution's characterization; reframe it as routine.",
    "Escalate -- accuse them of a deliberate cover-up.",
    "Plead for leniency; express genuine remorse.",
    "Introduce the key piece of evidence, obliquely.",
    "Cross-examine on the inconsistency in the timeline.",
    "Deliver the closing and demand a guilty verdict.",
]
# Facts with distinctive content words so engagement is measurable.
EVAL_FACTS = [
    "the cold-storage log had a six-hour gap",
    "the sign-off sheet shows one name in two hands",
    "a backup drive was reformatted the next morning",
    "the manifest lists varieties that were out of season",
    None, None,   # some beats carry no fact, as at runtime
]
EVAL_CONTEXT = [("judge", "Order. Counsel will keep to the matter at hand."),
                ("prosecutor", "The record before this court speaks for itself.")]

app = modal.App("buzzwords-evaluate")
image = (modal.Image.debian_slim()
         .pip_install("unsloth", "peft", "transformers", "datasets")
         .add_local_python_source("jargon_banks")
         .add_local_python_source("buzzwords"))
adapters_vol = modal.Volume.from_name("buzzwords-adapters", create_if_missing=True)
OUT = "/adapters"

_STOP = {"the", "a", "an", "to", "of", "and", "on", "in", "that", "was", "had", "with"}


def _content(text: str) -> set[str]:
    return {w.strip(".,;:!?\"'").lower() for w in (text or "").split()} - _STOP - {""}


def _hits(text: str, terms: list[str]) -> int:
    t = text.lower()
    return sum(1 for term in terms if term.lower() in t)


def _score(lines: list[str], terms: list[str]) -> dict:
    n = max(len(lines), 1)
    hits = [_hits(x, terms) for x in lines]
    return {"coverage": sum(h > 0 for h in hits) / n, "density": sum(hits) / n}


def _distinct_ratio(lines: list[str]) -> float:
    words = [w for x in lines for w in x.lower().split()]
    return len(set(words)) / max(len(words), 1)


def _engagement(lines: list[str], prompts: list[dict]) -> float:
    """Fraction of lines reusing >= 1 content word from the fact (or, factless, the cue)."""
    hit = 0
    for line, p in zip(lines, prompts):
        target = _content(p["fact"] or p["sd"])
        hit += bool(_content(line) & target)
    return hit / max(len(lines), 1)


def _prompts() -> list[dict]:
    """N_SAMPLES (role, sd, fact, context) combos in the exact runtime shape."""
    rng = random.Random(7)
    out = []
    roles = list(contracts.ROLE_SYS)
    for rep in range(2):
        for i, sd in enumerate(STAGE_DIRECTIONS):
            for j, role in enumerate(roles):
                fact = EVAL_FACTS[(i + j + rep) % len(EVAL_FACTS)]
                ctx = EVAL_CONTEXT[-(1 + (i + rep) % 2):]
                out.append({"role": role, "sd": sd, "fact": fact, "ctx": list(ctx)})
    rng.shuffle(out)
    return out[:N_SAMPLES]


@app.function(image=image, gpu="A10G", timeout=2 * 60 * 60,
              volumes={OUT: adapters_vol}, secrets=[modal.Secret.from_name("huggingface")])
def evaluate(styles: list[str]):
    import gc
    import torch
    from unsloth import FastLanguageModel
    from peft import PeftModel

    prompts = _prompts()
    summary = []
    for style in styles:
        try:
            model, tok = FastLanguageModel.from_pretrained(
                BASE_MODEL, max_seq_length=2048, load_in_4bit=False, dtype=torch.bfloat16)
            model = PeftModel.from_pretrained(model, f"{OUT}/actor_{style}")

            def gen():
                outs = []
                for p in prompts:
                    msgs = [{"role": "system",
                             "content": contracts.actor_system(p["role"], style)},
                            {"role": "user",
                             "content": contracts.actor_user(p["sd"], 3, p["fact"], p["ctx"])}]
                    text = tok.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True,
                                                   enable_thinking=False)
                    ids = tok(text, return_tensors="pt").to(model.device)
                    out = model.generate(**ids, max_new_tokens=80, do_sample=True,
                                         temperature=0.9, top_p=0.95)
                    outs.append(tok.decode(out[0][ids.input_ids.shape[1]:],
                                           skip_special_tokens=True).strip())
                return outs

            torch.manual_seed(0)
            lora_out = gen()
            with model.disable_adapter():
                torch.manual_seed(0)
                base_out = gen()

            base_s, lora_s = _score(base_out, JARGON[style]), _score(lora_out, JARGON[style])
            spec = sorted(((st, _score(lora_out, JARGON[st])["density"]) for st in JARGON),
                          key=lambda kv: -kv[1])
            own_top = spec[0][0] == style
            fluency_ok = _distinct_ratio(lora_out) >= 0.9 * _distinct_ratio(base_out)
            # engage is INFORMATIONAL only: the game demands oblique allusion, so literal
            # content-word overlap under-measures by design (measured 0.06-0.29 across all
            # styles on adapters whose style lift was clearly real). Decodability is gated
            # end-to-end by the e2e_gate teacher-solver instead (REBUILD_REVIEW.md §7.2).
            engage = _engagement(lora_out, prompts)
            checks = {"lift": lora_s["density"] > base_s["density"] * 1.5,
                      "own_top": own_top, "fluency": fluency_ok}
            verdict = "PASS" if all(checks.values()) else "REVIEW"

            print(f"\n===== actor_{style} =====  cov {base_s['coverage']:.2f}->"
                  f"{lora_s['coverage']:.2f}  density {base_s['density']:.2f}->"
                  f"{lora_s['density']:.2f}  engage={engage:.2f}  "
                  f"checks={ {k: ('Y' if v else 'n') for k, v in checks.items()} }  {verdict}")
            print("  top banks: " + ", ".join(f"{st}:{d:.2f}" for st, d in spec[:3]))
            for p, l in list(zip(prompts, lora_out))[:2]:
                print(f"  [{p['role']}|fact={bool(p['fact'])}] {l[:140]}")
            summary.append((style, base_s, lora_s, engage, verdict))
            del model, tok
            gc.collect(); torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"\n===== actor_{style} ===== ERROR: {e}")
            summary.append((style, {"coverage": 0, "density": 0},
                            {"coverage": 0, "density": 0}, 0.0, "ERROR"))

    print("\n================ SUMMARY ================")
    print(f"{'style':<11}{'covBase':>8}{'covLoRA':>8}{'denBase':>8}{'denLoRA':>8}"
          f"{'engage':>8}  verdict")
    for style, b, l, eng, verdict in summary:
        print(f"{style:<11}{b['coverage']:>8.2f}{l['coverage']:>8.2f}{b['density']:>8.2f}"
              f"{l['density']:>8.2f}{eng:>8.2f}  {verdict}")
    print(f"PASS: {sum(v == 'PASS' for *_, v in summary)}/{len(summary)}")
    return summary


@app.local_entrypoint()
def main(style: str = ""):
    styles = [s.strip() for s in style.split(",")] if style else list(JARGON)
    evaluate.remote(styles)
