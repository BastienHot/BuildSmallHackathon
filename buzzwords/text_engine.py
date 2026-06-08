"""Text inference through llama.cpp. Real models only.

Two roles:
  * Game Master (Qwen3.5-4B, pure transformer) -> GBNF-constrained Case File + beat
    decisions + scoring. GBNF guarantees parseable JSON; thinking mode is suppressed via
    /no_think system prefix (GBNF's root ::= "{" also enforces this structurally).
  * Actors (MiniCPM5-1B + one per-style LoRA) -> in-character jargon lines.

GPU execution path (priority order):
  1. Local CUDA GPU   — N_GPU_LAYERS=-1 set at import; @spaces.GPU is no-op.
  2. HF T4 persistent — same as above; GPU always present, no decorator needed.
  3. HF ZeroGPU       — @spaces.GPU borrows an A10G for each _gpu_call invocation.
                        Model loads inside the GPU context on first call; small models
                        (~0.6 GB actor, ~2.7 GB GM) reload fast on A10G.

The offline/demo path never reaches here (it plays hand-authored cases in pipeline.py),
so this module assumes weights are present and fails loudly otherwise.
"""

from __future__ import annotations

import gc
import json
import os

import spaces

from . import config
from .models import CaseFile, GMDecision, Line

# llama_cpp is imported lazily inside _gpu_call (which runs inside @spaces.GPU) because
# the CUDA wheel links against libcudart.so.12 which is only available inside the GPU
# context on ZeroGPU — importing at module level crashes the Space at startup.

# Valid-JSON string (no unescaped control chars) so json.loads never chokes on
# grammar-valid output. Matches llama.cpp's official json.gbnf string rule.
_STR = (r'string ::= "\"" ([^"\\\x00-\x1F\x7F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""'
        + "\n" + r'ws ::= [ \t]*')   # no newlines: stops the model padding output into truncation

# Judge scoring -> clean JSON.
SCORE_GRAMMAR = (r"""
root   ::= "{" ws "\"score\":" ws number ws "," ws "\"rationale\":" ws string ws "}"
number ::= [0-9] | [1-9][0-9] | "100"
""" + _STR)

# Hidden Case File. profession must be unrelated to the jargon style (smokescreen).
# NOTE: each GBNF rule MUST be on a single line (llama.cpp ends a rule at a newline).
CASEFILE_GRAMMAR = (r"""
root ::= "{" ws "\"profession\":" ws string ws "," ws "\"fault_plain\":" ws string ws "," ws "\"facts\":" ws facts ws "}"
facts ::= "[" ws string (ws "," ws string){2,4} ws "]"
""" + _STR)

# One beat from a finite deck (a small model classes well but invents poorly).
GM_DECISION_GRAMMAR = (r"""
root ::= "{" ws "\"next_speaker\":" ws speaker ws "," ws "\"beat_type\":" ws beat ws "," ws "\"intensity\":" ws intensity ws "," ws "\"stage_direction\":" ws string ws "," ws "\"wrap_up\":" ws bool ws "}"
speaker ::= "\"judge\"" | "\"prosecutor\"" | "\"defense\""
beat ::= "\"opening\"" | "\"charge\"" | "\"evidence\"" | "\"objection\"" | "\"escalate\"" | "\"plea\"" | "\"cross_examine\"" | "\"closing\"" | "\"exchange\""
intensity ::= "1" | "2" | "3" | "4" | "5"
bool ::= "true" | "false"
""" + _STR)

ROLE_SYS = {
    "judge": "You are the JUDGE. Speak with calm authority in dense {style} jargon.",
    "prosecutor": "You are the PROSECUTOR. Press the case hard in dense {style} jargon.",
    "defense": "You are the DEFENSE counsel. Deflect and reframe in dense {style} jargon.",
}
_ACTOR_RULES = (" English, 1-2 sentences. Follow the stage direction. Never name the "
                "defendant's profession or state the charge in plain words.")
# /no_think is a Qwen3 (not Qwen3.5) convention; llama-cpp-python has no stable
# chat_template_kwargs API for thinking control yet. GBNF is the real guarantee:
# root ::= "{" means the first generated token MUST be `{`, so thinking tokens
# (<think>…</think>) are structurally impossible for all three GM call sites.
_GM_SYS = ("You are the GAME MASTER directing a short courtroom debate. Output ONLY the "
           "requested JSON. Never reveal the profession or charge in plain words.")
_CASEFILE_SYS = ("Invent a hidden courtroom case. profession = the defendant's real job "
                 "(2-4 words), UNRELATED to the given jargon style. fault_plain = ONE "
                 "complete sentence stating exactly what they did wrong (a specific act, not "
                 "a single word). facts = 3-5 short oblique clues that never name the "
                 "profession in plain words.")
_SCORE_SYS = ("Grade how well the player's guess matches the true profession and charge. "
              "score 0-100, rationale one sentence.")


# ---------------------------------------------------------------------------
# GPU-bound inference kernel
# ---------------------------------------------------------------------------
# @spaces.GPU allocates an A10G for the duration of this call on ZeroGPU.
# On T4 / local CUDA / CPU it is a transparent no-op (the spaces package ships
# a stub that just calls the function directly when not running on ZeroGPU).
#
# Model loading happens lazily inside this function so it falls within the GPU
# context on ZeroGPU (CUDA is unavailable outside it). The model_cache dict is
# owned by the TextEngine instance and passed by reference, so loaded models
# persist across calls for the lifetime of the engine object.
@spaces.GPU(duration=120)
def _gpu_call(
    spec: dict,
    model_cache: dict,
    system: str,
    prompt: str,
    grammar_str: str | None,
    max_tokens: int,
    temperature: float,
) -> str:
    """Load model (if not cached) and run one chat-completion. GPU-context-safe."""
    from llama_cpp import Llama, LlamaGrammar  # noqa: PLC0415 — must be inside GPU context
    key = spec["key"]
    if key not in model_cache:
        # Keep at most ONE actor model resident. The 3 roles share the same base +
        # adapter; switching style reloads this single actor slot. GM stays alongside.
        if key.startswith("actor-"):
            for stale in [k for k in model_cache if k.startswith("actor-")]:
                del model_cache[stale]
            gc.collect()
        model_cache[key] = Llama(
            model_path=spec["path"],
            n_ctx=spec.get("n_ctx", 4096),
            n_gpu_layers=spec.get("n_gpu_layers", 0),
            lora_path=spec.get("lora_path"),
            verbose=False,
        )
    grammar_obj = LlamaGrammar.from_string(grammar_str) if grammar_str else None
    out = model_cache[key].create_chat_completion(
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        **({'grammar': grammar_obj} if grammar_obj else {}),
    )
    return out["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Public engine
# ---------------------------------------------------------------------------

class TextEngine:
    """Owns the model cache and exposes typed inference methods."""

    def __init__(self):
        self._models: dict = {}

    def complete(self, spec: dict, system: str, prompt: str, *,
                 grammar: str | None = None, max_tokens: int = 256,
                 temperature: float = 0.8) -> str:
        return _gpu_call(spec, self._models, system, prompt, grammar, max_tokens, temperature)

    # ---------------------------------------------------------- Game Master
    def casefile(self, style: str, difficulty: str) -> dict:
        prompt = (f"Jargon style (smokescreen, unrelated): {style}. Difficulty: {difficulty}.")
        for attempt in range(2):  # grammar guarantees valid JSON; retry once if a gen truncates
            raw = self.complete(config.GM_MODEL, _CASEFILE_SYS, prompt,
                                grammar=CASEFILE_GRAMMAR, max_tokens=600, temperature=0.9)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                if attempt:
                    raise

    def gm_decide(self, case_file: CaseFile, transcript: list[Line], turn: int,
                  budget: int) -> GMDecision:
        prompt = _gm_prompt(case_file, transcript, turn, budget)
        for attempt in range(2):
            try:
                d = json.loads(self.complete(config.GM_MODEL, _GM_SYS, prompt,
                                             grammar=GM_DECISION_GRAMMAR, max_tokens=320,
                                             temperature=0.7))
                break
            except json.JSONDecodeError:
                if attempt:
                    raise
        return GMDecision(next_speaker=d["next_speaker"], beat_type=d["beat_type"],
                          stage_direction=d["stage_direction"], intensity=d["intensity"],
                          wrap_up=d["wrap_up"])

    def act(self, role: str, stage_direction: str, style: str, intensity: int = 3) -> str:
        # Use the style adapter once it's trained; until then run the vanilla base.
        lora = config.STYLE_LORAS.get(style)
        lora = lora if (lora and os.path.exists(lora)) else None
        spec = {**config.JARGON_BASE_MODEL, "lora_path": lora,
                "key": f"actor-{style}" if lora else "actor-base"}
        system = ROLE_SYS[role].format(style=style) + _ACTOR_RULES
        prompt = f"Stage direction: {stage_direction}\nIntensity: {intensity}/5."
        return self.complete(spec, system, prompt, max_tokens=160, temperature=0.9).strip()

    def score(self, profession: str, fault_plain: str, guess: str) -> tuple[int, str]:
        prompt = (f"True profession: {profession}\nTrue charge: {fault_plain}\n"
                  f"Player's guess: {guess}")
        for attempt in range(2):
            try:
                res = json.loads(self.complete(config.GM_MODEL, _SCORE_SYS, prompt,
                                               grammar=SCORE_GRAMMAR, max_tokens=256,
                                               temperature=0.3))
                break
            except json.JSONDecodeError:
                if attempt:
                    raise
        return max(0, min(100, int(res["score"]))), res["rationale"]


def _gm_prompt(case_file: CaseFile, transcript: list[Line], turn: int, budget: int) -> str:
    if turn >= budget - 1:
        pressure = "This MUST be the closing beat: set wrap_up=true."
    elif turn >= budget - config.WRAP_PRESSURE_AT:
        pressure = "Begin converging; set wrap_up=true once a verdict is natural."
    else:
        pressure = ""
    recent = "\n".join(f"{l.actor}: {l.text}" for l in transcript[-6:]) or "(no lines yet)"
    return (f"Hidden brief (keep oblique): the defendant is a {case_file.profession} who "
            f"{case_file.fault_plain}. Facts: {'; '.join(case_file.facts)}\n"
            f"Transcript so far:\n{recent}\n"
            f"This is turn {turn + 1} of {budget}. {pressure}")
