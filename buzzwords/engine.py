"""Inference through a managed llama-server subprocess (the only runtime).

ONE resident MiniCPM5-1B base; the director LoRA and every present style LoRA are
registered at startup with --lora-init-without-apply (all scales 0) and activated
PER REQUEST via the `lora` field — no merge, no reload, no second copy of the base
(REBUILD_REVIEW.md §8.2). Unmerged adapters run through the same AVX2 kernels as the
base; the rank-32 overhead is a few percent per token.

Slots: the GM is pinned to slot 0 and the actor to slot 1 (`id_slot`), so each keeps
its own warm prompt cache (`cache_prompt: true`). The GM prompt is stable-prefix /
append-only (contracts.gm_user), so per-beat prefill is ~one new line, not the
whole transcript (§4.2).

Thinking is suppressed twice: structurally for the GM (every grammar starts with
root ::= "{") and explicitly for everyone via chat_template_kwargs
{"enable_thinking": false} (§4.5). A startup self-test asserts the actor path
actually returns clean text.

All calls are serialized behind one lock: 2 vCPUs can't run two generations anyway,
and it makes the engine safe under Gradio's concurrent event listeners (§4.6).
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import threading
import time

import requests

from . import config, contracts

log = logging.getLogger(__name__)

_THINK_MARKERS = ("<think", "<|thought_begin|>", "</think")


class EngineError(RuntimeError):
    pass


class TextEngine:
    """Owns the llama-server subprocess and exposes the four typed game calls."""

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._url = f"http://127.0.0.1:{config.LLAMA_SERVER_PORT}"
        self._lora_ids: dict[str, int] = {}   # adapter name -> server id (order of --lora flags)

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._proc and self._proc.poll() is None:
            return
        adapters = [("director", config.DIRECTOR_LORA)]
        adapters += [(f"style-{s}", p) for s, p in config.STYLE_LORAS.items()
                     if os.path.exists(p)]
        cmd = [config.LLAMA_SERVER_BIN,
               "-m", config.BASE_GGUF,
               "--host", "127.0.0.1", "--port", str(config.LLAMA_SERVER_PORT),
               "-c", str(config.N_CTX),
               "--parallel", str(config.N_PARALLEL),
               "--threads", str(config.N_THREADS),
               "--threads-batch", str(config.N_THREADS),
               "--lora-init-without-apply"]
        for name, path in adapters:
            cmd += ["--lora", path]
        self._lora_ids = {name: i for i, (name, _) in enumerate(adapters)}
        log.info("Starting llama-server: %s", " ".join(cmd))
        # stdout/stderr to our stdout -> HF Container logs capture server-side errors.
        self._proc = subprocess.Popen(cmd)
        atexit.register(self.stop)
        self._wait_healthy()
        self._self_test()

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def _wait_healthy(self, timeout: float = 180.0) -> None:
        """Poll /health until the model is loaded (CPU load takes tens of seconds)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise EngineError(f"llama-server exited with code {self._proc.returncode} "
                                  "during startup — see container logs above")
            try:
                if requests.get(f"{self._url}/health", timeout=2).status_code == 200:
                    log.info("llama-server healthy; adapters: %s", self._lora_ids)
                    return
            except requests.RequestException:
                pass
            time.sleep(1.0)
        raise EngineError(f"llama-server not healthy after {timeout:.0f}s")

    def _self_test(self) -> None:
        """Assert the actor path returns clean, non-empty, think-free text (§4.5)."""
        out = self._chat(config.ACTOR_SLOT, None,
                         contracts.actor_system("judge", "corporate"),
                         contracts.actor_user("Call the hearing to order.", 2, None, []),
                         grammar=None, max_tokens=60, temperature=0.7)
        if not out.strip():
            raise EngineError("Self-test: actor returned an empty line — thinking mode "
                              "is likely eating the token budget; check the chat template")
        low = out.lower()
        if any(m in low for m in _THINK_MARKERS):
            raise EngineError(f"Self-test: thought-channel markers in actor output: {out[:120]!r}")
        log.info("Actor self-test OK: %r", out[:80])

    # ------------------------------------------------------------ transport
    def _chat(self, slot: int, lora_name: str | None, system: str, user: str, *,
              grammar: str | None, max_tokens: int, temperature: float,
              repeat_penalty: float | None = None) -> str:
        body = {
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            # llama.cpp extensions (accepted alongside OAI fields):
            "cache_prompt": True,
            "id_slot": slot,
            "lora": ([{"id": self._lora_ids[lora_name], "scale": 1.0}]
                     if lora_name else []),                 # [] -> all adapters at scale 0
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if grammar:
            body["grammar"] = grammar
        if repeat_penalty:
            body["repeat_penalty"] = repeat_penalty
        with self._lock:
            r = requests.post(f"{self._url}/v1/chat/completions", json=body, timeout=600)
        if r.status_code != 200:
            raise EngineError(f"llama-server {r.status_code}: {r.text[:300]}")
        return r.json()["choices"][0]["message"]["content"]

    def _json_call(self, slot: int, lora: str | None, system: str, user: str,
                   grammar: str, max_tokens: int, temperature: float) -> dict:
        """Grammar guarantees valid JSON; retry once if a generation truncates mid-string."""
        for attempt in range(2):
            raw = self._chat(slot, lora, system, user, grammar=grammar,
                             max_tokens=max_tokens, temperature=temperature)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                log.warning("Grammar output failed json.loads (attempt %d): %r",
                            attempt + 1, raw[:160])
        raise EngineError("GM produced unparseable output twice (truncation?)")

    # ------------------------------------------------------------ game calls
    def facts(self, style: str, profession: str, fault: str) -> list[str]:
        d = self._json_call(config.GM_SLOT, "director", contracts.FACTS_SYS,
                            contracts.facts_user(style, profession, fault),
                            contracts.FACTS_GRAMMAR,
                            config.MAX_TOKENS["facts"], temperature=0.9)
        return [str(f).strip() for f in d["facts"]]

    def decide(self, profession: str, fault: str, facts: list[str],
               transcript: list[tuple[str, str]], turn: int, budget: int,
               forced_fact: int | None) -> dict:
        d = self._json_call(config.GM_SLOT, "director", contracts.GM_SYS,
                            contracts.gm_user(profession, fault, facts, transcript,
                                              turn, budget, forced_fact),
                            contracts.DECISION_GRAMMAR,
                            config.MAX_TOKENS["decide"], temperature=0.7)
        if not contracts.valid_decision(d):
            raise EngineError(f"Invalid decision shape: {d!r}")
        return d

    def act(self, role: str, style: str, stage_direction: str, intensity: int,
            fact: str | None, context: list[tuple[str, str]]) -> str:
        lora = f"style-{style}" if f"style-{style}" in self._lora_ids else None
        return self._chat(config.ACTOR_SLOT, lora,
                          contracts.actor_system(role, style),
                          contracts.actor_user(stage_direction, intensity, fact, context),
                          grammar=None, max_tokens=config.MAX_TOKENS["act"],
                          temperature=0.9, repeat_penalty=config.REPEAT_PENALTY).strip()

    def score(self, profession: str, fault_plain: str, guess: str) -> tuple[int, str]:
        d = self._json_call(config.GM_SLOT, "director", contracts.SCORE_SYS,
                            contracts.score_user(profession, fault_plain, guess),
                            contracts.SCORE_GRAMMAR,
                            config.MAX_TOKENS["score"], temperature=0.3)
        return max(0, min(100, int(d["score"]))), str(d["rationale"])
