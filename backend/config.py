"""Runtime configuration, read from environment variables.

Provider selection lives here so swapping Gemini <-> Claude is a one-line env
change (PREDICTOR_PROVIDER) with no code edits.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

try:  # load .env once at import; optional so the pure core has no hard dep
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


@dataclass(frozen=True)
class Settings:
    # --- prediction LLM ---
    predictor_provider: str          # "gemini" | "claude" | "mock"
    gemini_model: str
    gemini_api_key: str | None
    claude_model: str
    claude_api_key: str | None
    deepseek_model: str
    deepseek_api_key: str | None
    deepseek_base_url: str
    # --- demo-day network fallback (PREDICTOR_PROVIDER=demo_fallback) ---
    demo_fallback_inner: str         # inner live provider: "gemini" | "claude"
    demo_fallback_timeout_s: float   # seconds before falling back to the local lookup
    # --- speech-to-text ---
    stt_provider: str                # "mock" | "deepgram"
    deepgram_api_key: str | None
    # --- behaviour tuning ---
    max_candidates: int
    pause_ms: int                    # stall pause threshold
    context_turns: int               # verbatim-layer cap: recent turns sent as-is
    # --- acoustic channel ---
    acoustic_model: str              # FillerNet checkpoint path ("" disables)
    acoustic_conf: float             # filler confidence threshold
    # --- speculative prefetch ---
    prefetch: bool
    prefetch_every: int              # shadow-predict every N new content words
    # --- salient-entity memory ---
    entity_memory: bool              # inject out-of-window entity hints into the prompt
    # --- long-horizon context ---
    # Defaulted and placed last on purpose: Settings is constructed positionally
    # in places, so a new REQUIRED field in the middle breaks every call site.
    context_budget_tokens: int = 2048   # approximate token budget for assembled context
    # --- speaker gate (proximity/energy; see backend/acoustic/speaker_gate.py) ---
    wearer_gate: bool = False        # OFF by default -- see the note in get_settings()
    wearer_conf_min: float = 0.35    # below this, a word/event is treated as not the wearer
    # --- local LLM (llama.cpp serving Qwen3.8-27B Q4) ---
    local_llm_url: str = "http://127.0.0.1:8080"
    local_llm_model: str = "Qwen3.8-27B-Q4_K_M"
    local_llm_ctx: int = 4096
    # --- verbatim ASR (server-side CrisperWhisper on the /ws/audio PCM) ---
    # "browser": the page's Web Speech transcript arrives on /ws, as before.
    # "crisper": the server transcribes the SAME PCM the acoustic channel
    #            already receives, so the transcript keeps the fillers,
    #            repetitions and cut-off words Chrome deletes (measured: 0.000
    #            preserved on 5,044 clips) and both channels share one clock.
    asr_provider: str = "browser"
    asr_model: str = "turbo"
    asr_mode: str = "verbatim"       # "verbatim" keeps dysfluency; "intended" strips it
    asr_device: str = "auto"
    asr_compute_type: str = "float16"
    asr_step_ms: int = 700
    asr_turn_end_ms: int = 2000
    asr_word_timestamps: bool = False
    # --- stutter channel (StutterNet supersedes FillerNet when set) ---
    stutter_model: str = "models/stutternet.pt"
    # "cnn" is the 583k log-mel model; "ssl" is WavLM Base+ (Block AP 0.325 vs
    # 0.256, ANY 0.882 vs 0.786). CNN remains the default because SSL costs
    # 156 ms per 3 s window on an idle CPU and 438 ms per hop under the live
    # server, against a 125 ms hop -- it needs a GPU (19 ms) to run at all.
    # Set STUTTER_BACKEND=ssl AND ACOUSTIC_DEVICE=cuda together; STUTTER_MODEL
    # then defaults to the v2 SSL checkpoint. STUTTER_BACKEND=ssl with a CPU
    # device is REFUSED at startup rather than shipped as a footgun (override
    # with STUTTER_ALLOW_CPU=on, for offline correctness checks only).
    stutter_backend: str = "cnn"
    stutter_scale: float = 1.0       # scales the checkpoint's fitted thresholds
    # --- live-path detection gates (see backend/acoustic/stream.py) ---
    # These are env-configurable because they were the root cause of "obvious
    # stutter -> nothing" in real use (eval/diagnose_live_stutter.py). The
    # AcousticStream CONSTRUCTOR defaults are unchanged (min_voiced 800, lag 0,
    # refractory 1200) so every eval harness that builds a stream directly keeps
    # its published numbers; these config defaults drive only the live server.
    #
    # context_lag_ms is THE fix: the live decision used to read the trailing edge
    # of the 3 s window, where a frame has no right context and a bidirectional
    # encoder scores an obvious dysfluency far below its full-context peak -- the
    # peak the threshold is calibrated on. Reading `context_lag_ms` behind the
    # edge gives the frame that much right context. 400 ms of added latency, well
    # inside the ~1.5 s stall budget. See eval/results/recall_v2_eval.json.
    acoustic_context_lag_ms: int = 400
    # The min_voiced gate ate obvious events at utterance onset and, because a
    # block IS silence, reset the voiced counter across the block itself. 400 ms
    # keeps the "a lone um before any sentence is not a stall" guard while no
    # longer blinding the detector for 800 ms after every pause; measured, it
    # fires the obvious set as well as 200 ms while roughly halving fluent
    # false-fires (eval/results/recall_v2_eval.json).
    acoustic_min_voiced_ms: int = 400
    acoustic_refractory_ms: int = 1200
    # Minimum time between two served suggestions, across every trigger.
    # Measured on real aphasic speech the stack fired 17-25 times/minute --
    # aphasic speech is dense with dysfluency and firing on each one is the
    # "it nags" failure. See eval/results/aphasia_eval.json for the sweep.
    stall_min_gap_ms: int = 4000
    # Torch device the acoustic models run on ("cpu" | "cuda" | "cuda:N" |
    # "mps"; ACOUSTIC_DEVICE=auto resolves to cuda when one is visible). It
    # exists because the SSL backend is unusable without it: the config told
    # the operator to "use a machine that has a GPU" while there was no knob to
    # put the model ON that GPU, so AcousticStream took its device="cpu"
    # default and 438 ms of torch ran on the event loop every 125 ms hop.
    acoustic_device: str = "cpu"
    # True when the operator NAMED the stutter checkpoint or backend
    # (STUTTER_MODEL / STUTTER_BACKEND present in the environment). A named
    # checkpoint that is not on disk is an error; the built-in default that is
    # not on disk is a loud downgrade, because models/*.pt is untracked and a
    # clean clone must still start. See AcousticStream.__init__.
    stutter_required: bool = False
    # --- Gemini transport (GEMINI_STREAM) ---
    # Off by default: the shipped path is one non-streaming generate_content
    # call. On, the same request goes through generate_content_stream and the
    # chunks are concatenated before parsing -- the candidate list the pipeline
    # sees is identical, only the transport changes. Measured with
    # eval/bench_predictor_latency.py (eval/results/predictor_latency_bench.json).
    gemini_stream: bool = False
    # --- provider failover (PREDICTOR_FALLBACKS) ---
    # Comma list of providers to race behind PREDICTOR_PROVIDER, in order
    # (e.g. "gemini,mock"). Empty = the bare provider, exactly as before. Added
    # after a rehearsal where DeepSeek answered 503/timeout for 30+ stalls in
    # a row and every one became an empty card -- see predictor/failover.py.
    predictor_fallbacks: tuple[str, ...] = ()


def _stutter_backend() -> str:
    """Which acoustic backend, validated rather than assumed.

    A typo here would silently fall back to the CNN while the operator
    believes WavLM is running, and the two disagree by 0.07 AP on Block.
    """
    v = os.getenv("STUTTER_BACKEND", "cnn").strip().lower()
    if v not in ("cnn", "ssl", "temporal"):
        raise ValueError("STUTTER_BACKEND must be 'cnn', 'ssl', or 'temporal', got %r" % v)
    return v


_DEVICE_RE = re.compile(r"cpu|cuda(:\d+)?|mps")


def _acoustic_device() -> str:
    """Which torch device the acoustic models load onto, validated the same way
    the backend string is.

    A typo must not silently downgrade to CPU: on CPU the SSL backend takes
    438 ms per 125 ms hop, so "cude" would look like a working GPU config and
    behave like a stalled server. "auto" is the one value that resolves at
    read time rather than being taken literally.
    """
    v = os.getenv("ACOUSTIC_DEVICE", "cpu").strip().lower()
    if v == "auto":
        try:  # torch is a hard dep of the acoustic path, but not of config
            import torch

            v = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # pragma: no cover - torch absent
            v = "cpu"
    if not _DEVICE_RE.fullmatch(v):
        raise ValueError(
            "ACOUSTIC_DEVICE must be 'auto', 'cpu', 'cuda', 'cuda:N' or 'mps', "
            "got %r" % v)
    return v


def _flag(name: str, default: str = "off") -> bool:
    return os.getenv(name, default).strip().lower() in ("on", "1", "true", "yes")


def get_settings() -> Settings:
    """Build Settings from the current environment (re-read on each call)."""
    backend = _stutter_backend()
    device = _acoustic_device()
    # SSL on CPU is refused rather than shipped. Measured with the released
    # checkpoint: 438 ms per _run_stutter against a 125 ms hop, run
    # synchronously from the /ws/audio handler -- it does not merely miss the
    # budget, it makes the whole session late. Failing at startup with an
    # instruction is strictly kinder than a server that appears to run.
    if backend in ("ssl", "temporal") and device == "cpu" and not _flag("STUTTER_ALLOW_CPU"):
        raise ValueError(
            "STUTTER_BACKEND=ssl needs a GPU: measured 438 ms per 125 ms hop "
            "on CPU. Set ACOUSTIC_DEVICE=cuda (or auto on a CUDA machine). To "
            "run it on CPU anyway -- offline correctness checks only, NOT a "
            "live session -- set STUTTER_ALLOW_CPU=on.")
    return Settings(
        predictor_provider=os.getenv("PREDICTOR_PROVIDER", "gemini").lower(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        claude_model=os.getenv("CLAUDE_MODEL", "claude-haiku-4-5"),
        claude_api_key=os.getenv("ANTHROPIC_API_KEY"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        demo_fallback_inner=os.getenv("DEMO_FALLBACK_INNER", "gemini").lower(),
        demo_fallback_timeout_s=float(os.getenv("DEMO_FALLBACK_TIMEOUT_S", "1.5")),
        stt_provider=os.getenv("STT_PROVIDER", "mock").lower(),
        deepgram_api_key=os.getenv("DEEPGRAM_API_KEY"),
        max_candidates=int(os.getenv("MAX_CANDIDATES", "3")),
        pause_ms=int(os.getenv("STALL_PAUSE_MS", "1300")),
        context_turns=int(os.getenv("CONTEXT_TURNS", "6")),
        # Approximate (len/4), never exact -- no tokenizer dependency. Bounds
        # the assembled context so a long conversation cannot grow the prompt
        # without limit, and so a small local model stays inside its window.
        context_budget_tokens=int(os.getenv("CONTEXT_BUDGET_TOKENS", "2048")),
        # WEARER_GATE defaults OFF, deliberately. Four eval harnesses
        # (run_prolongation_eval, run_stall_eval, run_dual_channel_ablation,
        # run_latency_bench) construct AcousticStream directly and their numbers
        # are published in docs/EVAL.md -- switching the gate on by default would
        # silently confound every one of them. It is opt-in at the app layer
        # (backend/session.py), where the published benches never reach.
        # It is also validated on SYNTHETIC MIXES ONLY: no real two-speaker
        # recording exists, so it must not be described as room-tested.
        wearer_gate=os.getenv("WEARER_GATE", "off").lower() in ("on", "1", "true", "yes"),
        wearer_conf_min=float(os.getenv("WEARER_CONF_MIN", "0.35")),
        local_llm_url=os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:8080"),
        local_llm_model=os.getenv("LOCAL_LLM_MODEL", "Qwen3.8-27B-Q4_K_M"),
        local_llm_ctx=int(os.getenv("LOCAL_LLM_CTX", "4096")),
        asr_provider=os.getenv("ASR_PROVIDER", "browser").lower(),
        asr_model=os.getenv("ASR_MODEL", "turbo"),
        asr_mode=os.getenv("ASR_MODE", "verbatim"),
        asr_device=os.getenv("ASR_DEVICE", "auto"),
        asr_compute_type=os.getenv("ASR_COMPUTE_TYPE", "float16"),
        asr_step_ms=int(os.getenv("ASR_STEP_MS", "700")),
        # Longer than STALL_PAUSE_MS on purpose: a word-search pause must not
        # be mistaken for the end of the turn, or the fragment the predictor
        # needs is filed away as a completed utterance at the exact moment the
        # speaker is still trying to finish it.
        asr_turn_end_ms=int(os.getenv("ASR_TURN_END_MS", "2000")),
        asr_word_timestamps=os.getenv("ASR_WORD_TIMESTAMPS", "off").lower()
        in ("on", "1", "true", "yes"),
        stutter_backend=backend,
        stutter_model=os.getenv(
            "STUTTER_MODEL",
            "models/stutternet_temporal_v1.pt" if backend == "temporal"
            else "models/stutternet_ssl_v2.pt" if backend == "ssl"
            else "models/stutternet.pt"),
        stutter_scale=float(os.getenv("STUTTER_SCALE", "1.0")),
        acoustic_context_lag_ms=int(os.getenv("ACOUSTIC_CONTEXT_LAG_MS", "400")),
        acoustic_min_voiced_ms=int(os.getenv("ACOUSTIC_MIN_VOICED_MS", "400")),
        acoustic_refractory_ms=int(os.getenv("ACOUSTIC_REFRACTORY_MS", "1200")),
        acoustic_device=device,
        stutter_required=bool(os.getenv("STUTTER_MODEL")
                              or os.getenv("STUTTER_BACKEND")),
        stall_min_gap_ms=int(os.getenv("STALL_MIN_GAP_MS", "4000")),
        acoustic_model=os.getenv("ACOUSTIC_MODEL", "models/fillernet.pt"),
        acoustic_conf=float(os.getenv("ACOUSTIC_CONF", "0.75")),
        prefetch=os.getenv("PREFETCH", "on").lower() in ("on", "1", "true", "yes"),
        prefetch_every=int(os.getenv("PREFETCH_EVERY", "3")),
        # default ON: measured on the frozen longctx set (docs/EVAL.md) --
        # out-of-window proper-noun recovery 0/20 without vs 19/20 with.
        entity_memory=os.getenv("ENTITY_MEMORY", "on").lower() in ("on", "1", "true", "yes"),
        gemini_stream=_flag("GEMINI_STREAM"),
        predictor_fallbacks=tuple(
            p.strip().lower() for p in os.getenv("PREDICTOR_FALLBACKS", "").split(",")
            if p.strip()),
    )
