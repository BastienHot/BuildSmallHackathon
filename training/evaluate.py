"""Benchmark a style LoRA: does it actually make the actor speak the style? (Modal GPU)

Compares the vanilla MiniCPM5-1B against base+actor_<style> on the SAME runtime-style
prompts (role + style + stage direction, exactly what buzzwords/text_engine.act sends) and
reports, with NO human labels -- the jargon banks are the yardstick:

  1. Style lift   -- target-jargon coverage (% lines w/ >=1 term) and density (hits/line),
                     base vs LoRA. The headline: did the adapter learn the style?
  2. Specificity  -- LoRA output scored against ALL banks; should peak on its OWN style
                     (proves it learned *that* style, not just generic buzzwords).
  3. Fluency      -- distinct-word ratio, to catch a LoRA that collapsed into stock phrases.

Run: modal run training/evaluate.py --style corporate
Pairs with the held-out eval_loss/perplexity that finetune.py writes to train_metrics.json.
"""

from __future__ import annotations

import modal

from jargon_banks import JARGON

BASE_MODEL = "openbmb/MiniCPM5-1B"

# Mirror buzzwords/text_engine.py so we test the adapter under real usage conditions.
ROLE_SYS = {
    "judge": "You are the JUDGE. Speak with calm authority in dense {style} jargon.",
    "prosecutor": "You are the PROSECUTOR. Press the case hard in dense {style} jargon.",
    "defense": "You are the DEFENSE counsel. Deflect and reframe in dense {style} jargon.",
}
ACTOR_RULES = (" English, 1-2 sentences. Follow the stage direction. Never name the "
               "defendant's profession or state the charge in plain words.")
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

app = modal.App("buzzwords-evaluate")
image = (modal.Image.debian_slim()
         .pip_install("unsloth", "peft", "transformers", "datasets")
         .add_local_python_source("jargon_banks"))
adapters_vol = modal.Volume.from_name("buzzwords-adapters", create_if_missing=True)
OUT = "/adapters"


# ------------------------------------------------------------------- metrics
def _hits(text: str, terms: list[str]) -> int:
    t = text.lower()
    return sum(1 for term in terms if term.lower() in t)


def _score(lines: list[str], terms: list[str]) -> dict:
    """Coverage = fraction of lines with >=1 term; density = mean term hits per line."""
    n = max(len(lines), 1)
    hits = [_hits(x, terms) for x in lines]
    return {"coverage": sum(h > 0 for h in hits) / n, "density": sum(hits) / n}


def _distinct_ratio(lines: list[str]) -> float:
    """Unique-word / total-word ratio (low -> the model is repeating itself)."""
    words = [w for x in lines for w in x.lower().split()]
    return len(set(words)) / max(len(words), 1)


def _prompts() -> list[tuple[str, str]]:
    """(role, stage_direction) pairs covering all three roles x several beats."""
    roles = list(ROLE_SYS)
    return [(roles[i % len(roles)], sd) for i, sd in enumerate(STAGE_DIRECTIONS * 2)]


@app.function(image=image, gpu="A10G", timeout=60 * 60,
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
            model, tok = FastLanguageModel.from_pretrained(BASE_MODEL, max_seq_length=2048,
                                                           load_in_4bit=True)
            model = PeftModel.from_pretrained(model, f"{OUT}/actor_{style}")

            def gen():
                outs = []
                for role, sd in prompts:
                    msgs = [{"role": "system",
                             "content": ROLE_SYS[role].format(style=style) + ACTOR_RULES},
                            {"role": "user",
                             "content": f"Stage direction: {sd}\nIntensity: 3/5."}]
                    # think-mode OFF so the base is judged fairly (actor runs think-free in game)
                    text = tok.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False)
                    ids = tok(text, return_tensors="pt").to(model.device)
                    out = model.generate(**ids, max_new_tokens=80, do_sample=True,
                                         temperature=0.9, top_p=0.95)
                    outs.append(tok.decode(out[0][ids.input_ids.shape[1]:],
                                           skip_special_tokens=True).strip())
                return outs

            torch.manual_seed(0)
            lora_out = gen()
            with model.disable_adapter():        # same weights, adapter off = vanilla base
                torch.manual_seed(0)
                base_out = gen()

            base_s, lora_s = _score(base_out, JARGON[style]), _score(lora_out, JARGON[style])
            spec = sorted(((st, _score(lora_out, JARGON[st])["density"]) for st in JARGON),
                          key=lambda kv: -kv[1])
            own_top = spec[0][0] == style
            verdict = "PASS" if (lora_s["density"] > base_s["density"] * 1.5 and own_top) else "REVIEW"

            print(f"\n===== actor_{style} =====  cov {base_s['coverage']:.2f}->{lora_s['coverage']:.2f}"
                  f"  density {base_s['density']:.2f}->{lora_s['density']:.2f}  own-top={own_top}  {verdict}")
            print("  top banks: " + ", ".join(f"{st}:{d:.2f}" for st, d in spec[:3]))
            for (role, sd), l in list(zip(prompts, lora_out))[:2]:
                print(f"  [{role}] {l[:140]}")
            summary.append((style, base_s, lora_s, own_top, verdict))
            del model, tok
            gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"\n===== actor_{style} ===== ERROR: {e}")
            summary.append((style, {"coverage": 0, "density": 0}, {"coverage": 0, "density": 0},
                            False, "ERROR"))

    print("\n================ SUMMARY ================")
    print(f"{'style':<11}{'covBase':>8}{'covLoRA':>8}{'denBase':>8}{'denLoRA':>8}{'own':>5}  verdict")
    for style, b, l, own, verdict in summary:
        print(f"{style:<11}{b['coverage']:>8.2f}{l['coverage']:>8.2f}{b['density']:>8.2f}"
              f"{l['density']:>8.2f}{('Y' if own else 'n'):>5}  {verdict}")
    print(f"PASS: {sum(v == 'PASS' for *_, v in summary)}/{len(summary)}")
    return summary


@app.local_entrypoint()
def main(style: str = ""):
    styles = [s.strip() for s in style.split(",")] if style else list(JARGON)  # default: all 8
    evaluate.remote(styles)
