"""Grammar-faithful END-TO-END gate for the director LoRA (Modal, real llama.cpp engine).

The transformers eval (director_evaluate.py) can't apply the GBNF grammar, but the game ALWAYS
runs llama.cpp + grammar. So this gate runs the real thing — base MiniCPM5-1B + director.lora.gguf,
the exact grammars/prompts from buzzwords/text_engine.py — and plays a full game. Grammar
guarantees valid JSON, so we judge what ships: a coherent smokescreen case file, varied/oblique
direction that converges, and calibrated scoring. Actor lines are STUBBED (this gate is about the
DIRECTOR; actor style is gated by training/evaluate.py).

  modal run training/director_gate.py
"""

from __future__ import annotations

import json

import modal

GM_REPO, GM_FILE = "openbmb/MiniCPM5-1B-GGUF", "MiniCPM5-1B-Q4_K_M.gguf"
DIRECTOR_GGUF = "director.lora.gguf"
STYLE, DIFF, BUDGET = "aviation", "normal", 12
WRAP_PRESSURE_AT = 2
SPEAKERS = ["judge", "prosecutor", "defense"]

# Grammars + prompts: copies of buzzwords/text_engine.py — MUST stay in sync.
_STR = (r'string ::= "\"" ([^"\\\x00-\x1F\x7F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""'
        + "\n" + r'ws ::= [ \t]*')
CASEFILE_GRAMMAR = (r'''
root ::= "{" ws "\"profession\":" ws string ws "," ws "\"fault_plain\":" ws string ws "," ws "\"facts\":" ws facts ws "}"
facts ::= "[" ws string (ws "," ws string){2,4} ws "]"
''' + _STR)
GM_DECISION_GRAMMAR = (r'''
root ::= "{" ws "\"next_speaker\":" ws speaker ws "," ws "\"beat_type\":" ws beat ws "," ws "\"intensity\":" ws intensity ws "," ws "\"stage_direction\":" ws string ws "," ws "\"wrap_up\":" ws bool ws "}"
speaker ::= "\"judge\"" | "\"prosecutor\"" | "\"defense\""
beat ::= "\"opening\"" | "\"charge\"" | "\"evidence\"" | "\"objection\"" | "\"escalate\"" | "\"plea\"" | "\"cross_examine\"" | "\"closing\"" | "\"exchange\""
intensity ::= "1" | "2" | "3" | "4" | "5"
bool ::= "true" | "false"
''' + _STR)
SCORE_GRAMMAR = (r'''
root   ::= "{" ws "\"score\":" ws number ws "," ws "\"rationale\":" ws string ws "}"
number ::= [0-9] | [1-9][0-9] | "100"
''' + _STR)

_CASEFILE_SYS = ("Invent a hidden courtroom case. profession = the defendant's real job "
                 "(2-4 words), UNRELATED to the given jargon style. fault_plain = a VERB "
                 "PHRASE completing \"The defendant ___\" (a specific wrongful act, e.g. "
                 "\"falsified an inspection log to hide a missed check\"); do NOT write a full "
                 "sentence and do NOT repeat \"the defendant\". facts = 3-5 short oblique clues "
                 "that never name the profession in plain words.")
_GM_SYS = ("You are the GAME MASTER directing a short courtroom debate. Output ONLY the "
           "requested JSON. Never reveal the profession or charge in plain words.")
_SCORE_SYS = ("Grade how well the player's guess matches the true profession and charge. "
              "score 0-100, rationale one sentence.")

# Adversarial stub transcript: a realistic courtroom has prosecutorial accusations, which is
# what gives the GM a reason to call the defense. (Actor style is gated separately.)
_STUB = {"judge": "[the judge calls for order and asks both sides to proceed]",
         "prosecutor": "[the prosecutor presses a sharp accusation and demands the defendant explain]",
         "defense": "[the defense pushes back and reframes the accusation]"}

app = modal.App("buzzwords-director-gate")
image = (modal.Image.debian_slim()
         .apt_install("build-essential", "cmake", "git")
         .env({"CMAKE_ARGS": "-DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_NATIVE=OFF"})
         .pip_install("llama-cpp-python", "huggingface_hub"))
gguf_vol = modal.Volume.from_name("buzzwords-gguf", create_if_missing=True)
GGUF_DIR = "/gguf"


def _gm_prompt(cf, transcript, turn, budget):
    if turn >= budget - 1:
        pressure = "This MUST be the closing beat: set wrap_up=true."
    elif turn >= budget - WRAP_PRESSURE_AT:
        pressure = "Begin converging; set wrap_up=true once a verdict is natural."
    else:
        pressure = ""
    recent = "\n".join(f"{a}: {t}" for a, t in transcript[-6:]) or "(no lines yet)"
    return (f"Hidden brief (keep oblique): the defendant is a {cf['profession']} who "
            f"{cf['fault_plain']}. Facts: {'; '.join(cf['facts'])}\n"
            f"Transcript so far:\n{recent}\n"
            f"This is turn {turn + 1} of {budget}. {pressure}")


@app.function(image=image, timeout=60 * 60, volumes={GGUF_DIR: gguf_vol},
              secrets=[modal.Secret.from_name("huggingface")])
def gate():
    import os
    from collections import Counter
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama, LlamaGrammar

    base = hf_hub_download(GM_REPO, GM_FILE)
    lora = f"{GGUF_DIR}/{DIRECTOR_GGUF}"
    if not os.path.exists(lora):
        print(f"ERROR: {lora} not found (run convert_gguf first)"); return
    gm = Llama(model_path=base, lora_path=lora, lora_scale=1.0, n_ctx=8192,
               n_gpu_layers=0, verbose=False)

    def jcall(system, user, grammar, max_tokens=512, temp=0.7):
        g = LlamaGrammar.from_string(grammar)
        out = gm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            grammar=g, max_tokens=max_tokens, temperature=temp)
        return json.loads(out["choices"][0]["message"]["content"])

    print(f"===== DIRECTOR GATE (grammar + lora) — {STYLE}/{DIFF}, budget {BUDGET} =====")
    cf = jcall(_CASEFILE_SYS, f"Jargon style (smokescreen, unrelated): {STYLE}. Difficulty: {DIFF}.",
               CASEFILE_GRAMMAR, max_tokens=512, temp=0.9)
    print("CASE FILE:")
    print(f"  profession : {cf['profession']}")
    print(f"  fault_plain: {cf['fault_plain']}")
    for x in cf["facts"]:
        print(f"  fact       : {x}")

    prof = cf["profession"].lower()
    blob = (cf["fault_plain"] + " " + " ".join(cf["facts"])).lower()
    casefile_leak = prof in blob
    verb_phrase = not cf["fault_plain"].strip().lower().startswith(("the defendant", "he ", "she ", "they "))

    transcript, turn, wrapped = [], 0, False
    speakers, beats, sd_leak = Counter(), [], 0
    while not (wrapped or turn >= BUDGET):
        d = jcall(_GM_SYS, _gm_prompt(cf, transcript, turn, BUDGET), GM_DECISION_GRAMMAR, max_tokens=320)
        role = d["next_speaker"]
        speakers[role] += 1
        beats.append(d["beat_type"])
        if prof in d["stage_direction"].lower():
            sd_leak += 1
        transcript.append((role, _STUB[role]))   # realistic (adversarial) stub so the GM has
        turn += 1                                 # a reason to rotate to the defense
        wrapped = turn >= BUDGET or (d["wrap_up"] and turn >= max(4, BUDGET // 2))
        print(f"T{turn:02d} {role}/{d['beat_type']} i{d['intensity']} wrap={d['wrap_up']} | {d['stage_direction'][:80]}")

    spot = f"a {cf['profession']} who {cf['fault_plain']}"
    s_spot = jcall(_SCORE_SYS, f"True profession: {cf['profession']}\nTrue charge: {cf['fault_plain']}\n"
                   f"Player's guess: {spot}", SCORE_GRAMMAR, max_tokens=256, temp=0.3)
    s_wrong = jcall(_SCORE_SYS, f"True profession: {cf['profession']}\nTrue charge: {cf['fault_plain']}\n"
                    f"Player's guess: a beekeeper who mislabeled honey jars", SCORE_GRAMMAR, max_tokens=256, temp=0.3)

    print("\n===== GATE CHECKS =====")
    checks = {
        "casefile valid + no self-leak": not casefile_leak,
        "fault_plain is verb phrase": verb_phrase,
        "all 3 speakers used": len(speakers) == 3,
        "no profession leak in directions": sd_leak == 0,
        "converged (wrapped) within budget": wrapped,
        "spot-on guess scores >= 80": s_spot["score"] >= 80,
        "wrong guess scores <= 25": s_wrong["score"] <= 25,
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  speakers={dict(speakers)}  beats={beats}")
    print(f"  score spot-on={s_spot['score']}  wrong={s_wrong['score']}")
    overall = all(checks.values())
    print(f"\nEND-TO-END GATE: {'PASS — one 1B can orchestrate; clear for Track A' if overall else 'FAIL — iterate'}")
    return overall


@app.local_entrypoint()
def main():
    gate.remote()
