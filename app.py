"""Buzzwords & Misdemeanors - HF Space entrypoint.

Run locally:  python app.py
The UI launches even without weights; it then tells you which GGUFs to add to models/.
"""

import spaces  # must be first — spaces requires CUDA not yet initialized at its import time

from buzzwords import config
from buzzwords import text_engine as _te  # noqa: F401 — registers @spaces.GPU at import time
from buzzwords.theme import get_css
from buzzwords.ui import build_ui

# On HF Spaces / fresh machines, set BW_FETCH_WEIGHTS=1 to pull the GGUFs from the Hub
# at startup (base models + trained style LoRAs) instead of committing them to the repo.
if config.FETCH_WEIGHTS:
    from buzzwords.weights import ensure_weights
    ensure_weights()

demo = build_ui()

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",   # Docker Space: bind the public port (HF expects 7860)
        server_port=7860,
        css=get_css(),
        allowed_paths=[str(config.MAPS_DIR)],   # serve the courtroom backdrops
    )
