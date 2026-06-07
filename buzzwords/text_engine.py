"""Text inference through llama.cpp. Real models only.

Two roles:
  * Game Master (Nemotron 4B, vanilla) -> GBNF-constrained Case File + beat decisions
    + scoring. GBNF guarantees parseable JSON, so we never guess at malformed output.
  * Actors (MiniCPM5-1B + one per-style LoRA) -> in-character jargon lines.

The offline/demo path never reaches here (it plays hand-authored cases in pipeline.py),
so this module assumes weights are present and fails loudly otherwise.
"""

from __future__ import annotations

import gc
import json
import os

from llama_cpp import Llama, LlamaGrammar

from . import config
from .models import CaseFile, GMDecision, Line

_STR = r'string ::= "\"" ([^"\\] | "\\" .)* "\""' + "\n" + r'ws ::= [ \t\n]*'

# Judge scoring -> clean JSON.
SCORE_GRAMMAR = (r"""
root   ::= "{" ws "\"score\":" ws number ws "," ws "\"rationale\":" ws string ws "}"
number ::= [0-9] | [1-9][0-9] | "100"
""" + _STR)

# Hidden Case File. profession must be unrelated to the jargon style (smokescreen).
# NOTE: each GBNF rule MUST be on a single line (llama.cpp ends a rule at a newline).
CASEFILE_GRAMMAR = (r"""
root ::= "{" ws "\"profession\":" ws string ws "," ws "\"fault_plain\":" ws string ws "," ws "\"facts\":" ws facts ws "}"
facts ::= "[" ws string (ws "," ws string)* ws "]"
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
_GM_SYS = ("You are the GAME MASTER directing a short courtroom debate. Output ONLY the "
           "requested JSON. Never reveal the profession or charge in plain words.")
_CASEFILE_SYS = ("Invent a hidden courtroom case. profession = the defendant's real job "
                 "(2-4 words), UNRELATED to the given jargon style. fault_plain = ONE "
                 "complete sentence stating exactly what they did wrong (a specific act, not "
                 "a single word). facts = 3-5 short oblique clues that never name the "
                 "profession in plain words.")
_SCORE_SYS = ("Grade how well the player's guess matches the true profession and charge. "
              "score 0-100, rationale one sentence.")


class TextEngine:
    """Loads (and keeps) the GM and one actor model. Only ever two of them."""

    def __init__(self):
        self._models: dict[str, Llama] = {}

    def _get(self, spec: dict) -> Llama:
        key = spec["key"]
        if key not in self._models:
            # Keep at most ONE actor model resident. The 3 roles are the same base+adapter
            # (only the system prompt differs); switching style reloads this single actor
            # slot. The GM model stays loaded alongside -> memory = GM + 1 actor, never 3.
            if key.startswith("actor-"):
                for stale in [k for k in self._models if k.startswith("actor-")]:
                    del self._models[stale]
                gc.collect()
            self._models[key] = Llama(
                model_path=spec["path"], n_ctx=spec.get("n_ctx", 4096),
                n_gpu_layers=spec.get("n_gpu_layers", 0),
                lora_path=spec.get("lora_path"), verbose=False,
            )
        return self._models[key]

    def complete(self, spec: dict, system: str, prompt: str, *, grammar: str | None = None,
                 max_tokens: int = 256, temperature: float = 0.8) -> str:
        kwargs = {"grammar": LlamaGrammar.from_string(grammar)} if grammar else {}
        out = self._get(spec).create_chat_completion(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temperature, **kwargs,
        )
        return out["choices"][0]["message"]["content"]

    # ---------------------------------------------------------- Game Master
    def casefile(self, style: str, difficulty: str) -> dict:
        raw = self.complete(config.GM_MODEL, _CASEFILE_SYS,
                            f"Jargon style (smokescreen, unrelated): {style}. "
                            f"Difficulty: {difficulty}.",
                            grammar=CASEFILE_GRAMMAR, max_tokens=400, temperature=0.9)
        return json.loads(raw)

    def gm_decide(self, case_file: CaseFile, transcript: list[Line], turn: int,
                  budget: int) -> GMDecision:
        d = json.loads(self.complete(config.GM_MODEL, _GM_SYS,
                                     _gm_prompt(case_file, transcript, turn, budget),
                                     grammar=GM_DECISION_GRAMMAR, max_tokens=200,
                                     temperature=0.7))
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
        res = json.loads(self.complete(config.GM_MODEL, _SCORE_SYS,
                                       f"True profession: {profession}\n"
                                       f"True charge: {fault_plain}\nPlayer's guess: {guess}",
                                       grammar=SCORE_GRAMMAR, max_tokens=200, temperature=0.3))
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
