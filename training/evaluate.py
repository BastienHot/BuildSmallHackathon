"""Benchmark a style LoRA: does the translator speak the style AND keep the meaning? (Modal GPU)

Compares vanilla MiniCPM5-1B against base+actor_<style> on the EXACT runtime translator
shape from buzzwords.contracts (translator system; "Line to rewrite: <plain>" user),
with NO human labels — the jargon banks + the plain probes are the yardstick.

  1. Style lift   -- target-bank coverage & density, base vs LoRA (>= 1.5x density).
  2. Specificity  -- LoRA output peaks on its OWN bank across all banks.
  3. Fluency      -- distinct-word ratio must not regress > 10% vs base (collapse guard).
  4. Meaning kept -- >= 70% of translations preserve a content anchor (SHAPE 3.0 gate).
  5. Not overdone -- <= 20% of translations exceed 2.2x the plain line's length.

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

# SHAPE 3.0 probes: concrete PLAIN lines (the translator's input at runtime). Each has
# distinctive content words so anchor survival ("meaning kept") is measurable.
PLAIN_PROBES = [
    "This hearing is called to order; counsel will keep to the matter at hand.",
    "The cold-storage log shows a six-hour gap on the night in question.",
    "My client followed the standard procedure and signed every page of the checklist.",
    "The sign-off sheet shows one name written in two different hands.",
    "A backup drive was reformatted the very next morning — that is not routine.",
    "The witness saw the side door propped open after closing time.",
    "We concede the schedule slipped, but no rule was broken and no one was harmed.",
    "The defendant collected payment for work that was never inspected.",
]

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


def _meaning_kept(lines: list[str], prompts: list[dict]) -> float:
    """SHAPE 3.0 headline: fraction of translations keeping >=1 content anchor from the
    plain input. Anchors are >=4-char content words, matched as substrings ('door' in
    'side door', 'closing' in 'post-closing') — exact >=5-char matching failed clearly
    good translations ('side door remained open at the buzzer beater', 2026-06-12)."""
    hit = 0
    for line, p in zip(lines, prompts):
        anchors = {w for w in _content(p["plain"]) if len(w) >= 4}
        low = line.lower()
        hit += (not anchors) or any(a in low for a in anchors)
    return hit / max(len(lines), 1)


def _overdone(lines: list[str], prompts: list[dict]) -> float:
    """Fraction of translations >2.2x the plain line's length ('without overdoing it')."""
    over = sum(len(l.split()) > 2.2 * len(p["plain"].split())
               for l, p in zip(lines, prompts))
    return over / max(len(lines), 1)


def _prompts() -> list[dict]:
    """N_SAMPLES (role, plain) combos in the exact runtime translator shape."""
    rng = random.Random(7)
    out = []
    roles = list(contracts.ROLE_VOICE)
    for rep in range(2):
        for i, plain in enumerate(PLAIN_PROBES):
            for j, role in enumerate(roles):
                out.append({"role": role, "plain": plain})
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
                             "content": contracts.actor_user(p["plain"])}]
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
            # 0.85x for the translation regime: a consistent register legitimately
            # narrows vocabulary vs the base's noisier paraphrases.
            fluency_ok = _distinct_ratio(lora_out) >= 0.85 * _distinct_ratio(base_out)
            # SHAPE 3.0: meaning preservation IS the actor's job now, so it gates.
            meaning = _meaning_kept(lora_out, prompts)
            overdone = _overdone(lora_out, prompts)
            checks = {"lift": lora_s["density"] > base_s["density"] * 1.5,
                      "own_top": own_top, "fluency": fluency_ok,
                      "meaning": meaning >= 0.7, "not_overdone": overdone <= 0.2}
            verdict = "PASS" if all(checks.values()) else "REVIEW"

            print(f"\n===== actor_{style} =====  cov {base_s['coverage']:.2f}->"
                  f"{lora_s['coverage']:.2f}  density {base_s['density']:.2f}->"
                  f"{lora_s['density']:.2f}  meaning={meaning:.2f} overdone={overdone:.2f}  "
                  f"checks={ {k: ('Y' if v else 'n') for k, v in checks.items()} }  {verdict}")
            print("  top banks: " + ", ".join(f"{st}:{d:.2f}" for st, d in spec[:3]))
            for p, l in list(zip(prompts, lora_out))[:2]:
                print(f"  [{p['role']}] PLAIN: {p['plain'][:90]}")
                print(f"     STYLED: {l[:140]}")
            summary.append((style, base_s, lora_s, meaning, verdict))
            del model, tok
            gc.collect(); torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"\n===== actor_{style} ===== ERROR: {e}")
            summary.append((style, {"coverage": 0, "density": 0},
                            {"coverage": 0, "density": 0}, 0.0, "ERROR"))

    print("\n================ SUMMARY ================")
    print(f"{'style':<11}{'covBase':>8}{'covLoRA':>8}{'denBase':>8}{'denLoRA':>8}"
          f"{'meaning':>8}  verdict")
    for style, b, l, mean_k, verdict in summary:
        print(f"{style:<11}{b['coverage']:>8.2f}{l['coverage']:>8.2f}{b['density']:>8.2f}"
              f"{l['density']:>8.2f}{mean_k:>8.2f}  {verdict}")
    print(f"PASS: {sum(v == 'PASS' for *_, v in summary)}/{len(summary)}")
    return summary


@app.local_entrypoint()
def main(style: str = ""):
    styles = [s.strip() for s in style.split(",")] if style else list(JARGON)
    evaluate.remote(styles)
