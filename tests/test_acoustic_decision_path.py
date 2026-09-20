"""The StutterNet decision path, which production always takes and the suite
never executed.

Every other test builds `AcousticStream(model_path=None)` with no
`stutter_model`, so `self.stutter` is always None -- while `backend/config.py`
ships `stutter_model="models/stutternet.pt"`. Two mutations proved the hole:

  * `if stutter_backend == "ssl"` -> `if False`   (SSL silently loads the CNN)
  * `frame_thresholds` -> `thresholds`            (the documented over-fire
    regression: 25 events/min on real aphasic speech, false alarm 0.704 ->
    0.889)

Both survived the whole suite. Both are caught here.

Checkpoints are untracked (models/*.pt is gitignored apart from fillernet.pt),
so every test that needs one skips cleanly when it is absent.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.acoustic.features import SR                      # noqa: E402
from backend.acoustic.stream import HOP_MS, AcousticStream    # noqa: E402
from backend.schemas import SilenceTick, Word                 # noqa: E402

CNN_CKPT = ROOT / "models" / "stutternet.pt"
SSL_CKPT = ROOT / "models" / "stutternet_ssl_v2.pt"
FILLERNET = ROOT / "models" / "fillernet.pt"

needs_cnn = pytest.mark.skipif(not CNN_CKPT.exists(),
                               reason="models/stutternet.pt absent (untracked)")
needs_ssl = pytest.mark.skipif(not SSL_CKPT.exists(),
                               reason="models/stutternet_ssl_v2.pt absent (untracked)")


def silence(ms: int) -> bytes:
    return b"\x00\x00" * int(SR * ms / 1000)


# ==========================================================================
# Backend selection: the checkpoint that loads must be the one asked for
# ==========================================================================
@needs_cnn
def test_cnn_backend_loads_the_log_mel_cnn_at_40_ms_frames():
    from backend.acoustic.stutter import StutterNet

    st = AcousticStream(model_path=None, stutter_model=CNN_CKPT, stutter_backend="cnn")
    assert isinstance(st.stutter, StutterNet)
    assert st.stutter_frame_ms == 40
    assert st.stutter_types == ["Block", "Prolongation", "SoundRep", "WordRep",
                                "Interjection"]


@needs_ssl
def test_ssl_backend_loads_wavlm_not_the_cnn():
    """MUTATION GUARD. `if stutter_backend == "ssl"` -> `if False` made the
    stream load the 583k log-mel CNN while the operator, /healthz and every
    published SSL number said WavLM. It survived the entire suite because no
    test ever passed a real checkpoint."""
    from backend.acoustic.stutter import StutterNet
    from backend.acoustic.stutter_ssl import StutterSSL

    st = AcousticStream(model_path=None, stutter_model=SSL_CKPT, stutter_backend="ssl")
    assert isinstance(st.stutter, StutterSSL)
    assert not isinstance(st.stutter, StutterNet)
    assert st.stutter_backend == "ssl"


@needs_ssl
def test_ssl_frame_rate_comes_from_the_ssl_module():
    """stream.py reads FRAME_MS from whichever module it imported. WavLM emits
    a frame every 20 ms and the CNN every 40; taking the wrong constant
    misattributes an event by up to 1.5 s, which is the entire stall-to-word
    budget."""
    from backend.acoustic.stutter import FRAME_MS as CNN_FRAME_MS
    from backend.acoustic.stutter_ssl import FRAME_MS as SSL_FRAME_MS

    assert (SSL_FRAME_MS, CNN_FRAME_MS) == (20, 40), "the two must stay distinct"
    st = AcousticStream(model_path=None, stutter_model=SSL_CKPT, stutter_backend="ssl")
    assert st.stutter_frame_ms == SSL_FRAME_MS


def _fire_at(stream: AcousticStream, hot_index: int) -> list:
    """Run the decision on a synthetic frame grid with exactly one hot frame,
    `hot_index` counted from the END. Bypasses the model so the test measures
    the SLICING (how much trailing audio the hop reads), not the weights."""
    n_frames = int(3000 / stream.stutter_frame_ms)
    logits = torch.full((1, len(stream.stutter_types), n_frames), -20.0)
    logits[0, :, n_frames - hot_index] = 20.0
    stream.stutter = lambda _x: logits
    stream._buf = torch.zeros(stream._model_window)
    stream._voiced_ms = 5000            # past the utterance-accumulation gate
    stream._consumed = SR * 5
    return stream._run_stutter()


@needs_cnn
def test_the_hop_reads_only_the_audio_since_the_last_hop_cnn():
    """125 ms hop / 40 ms frames = 3 trailing frames. A frame 4 back is older
    than this hop and must not fire now -- if it does, FRAME_MS is too small
    and every event is being attributed to the wrong moment."""
    st = AcousticStream(model_path=None, stutter_model=CNN_CKPT, stutter_backend="cnn")
    assert _fire_at(st, hot_index=3) != [], "the last hop must be read"
    st2 = AcousticStream(model_path=None, stutter_model=CNN_CKPT, stutter_backend="cnn")
    assert _fire_at(st2, hot_index=5) == [], "audio older than the hop must not fire"


@needs_ssl
def test_the_hop_reads_only_the_audio_since_the_last_hop_ssl():
    """125 ms hop / 20 ms frames = 6 trailing frames. Frame 5 back IS inside
    this hop for WavLM; reading the CNN's 40 ms constant would drop it."""
    st = AcousticStream(model_path=None, stutter_model=SSL_CKPT, stutter_backend="ssl")
    assert _fire_at(st, hot_index=5) != [], "5 frames = 100 ms is inside a 125 ms hop"
    st2 = AcousticStream(model_path=None, stutter_model=SSL_CKPT, stutter_backend="ssl")
    assert _fire_at(st2, hot_index=9) == [], "9 frames = 180 ms is older than the hop"


# ==========================================================================
# Thresholds: per-frame, never the pooled clip numbers
# ==========================================================================
@needs_cnn
def test_frame_thresholds_are_used_not_the_clip_thresholds():
    """MUTATION GUARD. `frame_thresholds` -> `thresholds` is the documented
    over-fire regression: the clip thresholds are fitted on a linear-softmax
    pool over ~75 frames, so they are systematically LOWER than the per-frame
    numbers this code compares against, and using them fired 25 times a minute
    on real aphasic speech (false alarm 0.704 -> 0.889)."""
    from backend.acoustic.stutter import checkpoint_meta

    meta = checkpoint_meta(CNN_CKPT)
    assert meta.get("frame_thresholds"), "this checkpoint is calibrated; fix the fixture"
    st = AcousticStream(model_path=None, stutter_model=CNN_CKPT, stutter_backend="cnn")

    assert st.stutter_thresholds == meta["frame_thresholds"]
    # The clip table carries an "ANY" key the frame table does not: its
    # presence alone proves the wrong dict was read.
    assert "ANY" not in st.stutter_thresholds
    for t in st.stutter_types:
        assert st.stutter_thresholds[t] > meta["thresholds"][t], (
            "%s: the frame threshold must be the HIGHER number; pooling averages "
            "a short event down" % t)


@needs_ssl
def test_frame_thresholds_are_used_for_the_ssl_backend_too():
    from backend.acoustic.stutter_ssl import checkpoint_meta

    meta = checkpoint_meta(SSL_CKPT)
    st = AcousticStream(model_path=None, stutter_model=SSL_CKPT, stutter_backend="ssl")
    assert st.stutter_thresholds == meta["frame_thresholds"]
    assert "ANY" not in st.stutter_thresholds


@needs_cnn
def test_an_uncalibrated_checkpoint_falls_back_to_clip_thresholds_loudly(monkeypatch,
                                                                        caplog):
    """The fallback is deliberate (an uncalibrated checkpoint should still
    run) but must never be quiet -- it is the exact configuration that
    over-fires."""
    import backend.acoustic.stutter as stutter_mod

    real = stutter_mod.checkpoint_meta

    def no_frames(path):
        meta = dict(real(path))
        meta["frame_thresholds"] = {}
        return meta

    monkeypatch.setattr(stutter_mod, "checkpoint_meta", no_frames)
    with caplog.at_level(logging.WARNING, logger="echo.acoustic"):
        st = AcousticStream(model_path=None, stutter_model=CNN_CKPT,
                            stutter_backend="cnn")
    assert st.stutter_thresholds == real(CNN_CKPT)["thresholds"]
    assert any("frame_thresholds" in r.getMessage() for r in caplog.records), \
        "an uncalibrated checkpoint must say so"


# ==========================================================================
# DEFECT 2 -- a missing checkpoint must not silently become FillerNet
# ==========================================================================
@pytest.mark.skipif(not FILLERNET.exists(), reason="models/fillernet.pt absent")
def test_a_missing_checkpoint_is_reported_not_silently_downgraded(caplog):
    """The operator asked for the five-type model and got the four-class
    FillerNet, with 'FillerNet ready from models/fillernet.pt' as the only log
    line. config._stutter_backend() validates its string precisely to stop this
    class of silent downgrade; the checkpoint on disk gets the same treatment."""
    missing = str(ROOT / "models" / "definitely_not_a_checkpoint.pt")
    with caplog.at_level(logging.ERROR, logger="echo.acoustic"):
        st = AcousticStream(model_path=str(FILLERNET), stutter_model=missing)
    assert st.stutter is None
    assert st.stutter_missing == missing, "the downgrade must survive as state"
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "a missing checkpoint must log at ERROR, not be inferred from silence"
    text = " ".join(r.getMessage() for r in errors)
    assert "definitely_not_a_checkpoint.pt" in text
    assert "FillerNet" in text, "the log must name what is actually running"


def test_a_checkpoint_the_operator_named_must_exist():
    """STUTTER_MODEL/STUTTER_BACKEND present in the environment is an explicit
    request. Degrading it is the bug; the default path may still degrade,
    because models/*.pt is untracked and a clean clone has to start."""
    missing = str(ROOT / "models" / "definitely_not_a_checkpoint.pt")
    with pytest.raises(FileNotFoundError) as exc:
        AcousticStream(model_path=None, stutter_model=missing, stutter_required=True)
    assert "definitely_not_a_checkpoint.pt" in str(exc.value)


def test_a_present_checkpoint_never_sets_the_missing_flag():
    st = AcousticStream(model_path=None, stutter_model=None)
    assert st.stutter_missing is None


# ==========================================================================
# DEFECT 3 -- one classifier pass PER HOP, not per feed()
# ==========================================================================
def test_every_hop_inside_a_large_frame_gets_its_own_pass():
    """A single `-=` inside an `if` cannot catch up when the frame is longer
    than the hop. Measured before the fix: 10 s fed as 1 s buffers ran 10
    passes instead of ~80, with _since_hop accumulating an 8.75 s backlog that
    grew without bound. The live path (the browser worklet's 100 ms) is under
    the 125 ms hop, so only eval replay and larger-buffer clients saw the 8x
    undersampled acoustic channel -- silently."""
    st = AcousticStream(model_path=None)      # no checkpoint: fast, path is the same
    seen: list[int] = []

    def counting_filler(end_off: int = 0):
        seen.append(end_off)
        return None

    st._run_filler = counting_filler
    for _ in range(10):
        st.feed(silence(1000))                # 8 hops per frame

    hops_per_frame = SR // st.hop             # 8
    assert len(seen) >= 9 * hops_per_frame, (
        "expected ~%d passes over 10 s, got %d" % (10 * hops_per_frame, len(seen)))
    assert st._since_hop < st.hop, "the backlog must not accumulate"


def test_catch_up_passes_read_distinct_windows_not_the_same_one_n_times():
    """Restoring the pass count while re-reading the newest window would look
    fixed and still classify one moment eight times, stamping every event at
    the end of the frame."""
    st = AcousticStream(model_path=None)
    st.feed(silence(2000))                    # fill the rolling buffer
    seen: list[int] = []
    st._run_filler = lambda end_off=0: seen.append(end_off)
    st.feed(silence(1000))

    assert len(seen) == SR // st.hop
    assert seen == sorted(seen, reverse=True), "oldest pending hop first"
    assert len(set(seen)) == len(seen), "every pass must look at a different window"
    assert seen[-1] == 0 and seen[0] == (len(seen) - 1) * st.hop


def test_a_frame_shorter_than_the_hop_still_runs_exactly_one_pass():
    """The live path must be untouched: 100 ms frames against a 125 ms hop."""
    st = AcousticStream(model_path=None)
    seen: list[int] = []
    st._run_filler = lambda end_off=0: seen.append(end_off)
    for _ in range(10):
        st.feed(silence(100))
    assert len(seen) == 8, "10 x 100 ms = 1000 ms = 8 hops of 125 ms"
    assert set(seen) == {0}


# ==========================================================================
# DEFECT 4 -- two /ws/audio connections, one Timeline, one clock
# ==========================================================================
def _mock_settings(monkeypatch, **env):
    import backend.config as config_mod

    monkeypatch.setenv("PREDICTOR_PROVIDER", "mock")
    monkeypatch.setenv("ACOUSTIC_MODEL", "")
    monkeypatch.setenv("STUTTER_MODEL", "")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return config_mod.get_settings()


class _FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_a_late_connection_is_offset_onto_the_session_clock(monkeypatch):
    """Repro: one audio socket connects, a second (a reload, a second tab)
    joins 40 s later. Each AcousticStream/VerbatimASR starts _consumed=0 at
    connect, so the late socket stamped its first word at 100 ms and
    interleaved it 40 s into the past on
    the Timeline the session's StallDetector owns. The observed fragment was
    'remember hello the there' and the computed pause 40100 ms."""
    import backend.session as session_mod

    clock = _FakeClock()
    monkeypatch.setattr(session_mod.time, "monotonic", clock)
    sess = session_mod.EchoSession(_mock_settings(monkeypatch))

    first = sess.new_audio_channels()
    first.acoustic.feed(silence(1000))
    clock.t += 40.0                            # the second socket joins 40 s later
    second = sess.new_audio_channels()

    assert first.clock_offset_ms == 0
    assert second.clock_offset_ms == 40000
    assert second.acoustic.now_ms == 40000, "a late stream must not restart at 0"
    second.acoustic.feed(silence(100))
    assert second.acoustic.now_ms == 40100

    first.acoustic.feed(silence(39000))
    assert first.acoustic.now_ms == 40000
    assert second.acoustic.now_ms > first.acoustic.now_ms, (
        "the later connection's events must sort AFTER the earlier one's")


def test_asr_words_from_a_late_connection_are_lifted_onto_the_session_clock(monkeypatch):
    """The ASR is not ours to change, so its Words and SilenceTicks are lifted
    in handle_audio_events -- before anything downstream compares them."""
    import backend.session as session_mod

    class _FakeASR:
        def __init__(self):
            self.interim_text = ""

        def feed(self, pcm):
            return [Word(text="hello", start_ms=0, end_ms=100), SilenceTick(at_ms=150)]

        def close(self):
            pass

    class _FakeAcoustic:
        wearer_conf = None

        def feed(self, pcm):
            return []

    sess = session_mod.EchoSession(_mock_settings(monkeypatch))
    seen = []

    async def record(item):
        seen.append(item)

    sess.pipeline.handle = record
    channels = session_mod.AudioChannels(
        acoustic=_FakeAcoustic(), asr=_FakeASR(), clock_offset_ms=40000)
    asyncio.run(sess.handle_audio_events(channels, silence(100)))

    word = next(i for i in seen if isinstance(i, Word))
    tick = next(i for i in seen if isinstance(i, SilenceTick))
    assert (word.start_ms, word.end_ms) == (40000, 40100)
    assert tick.at_ms == 40150


def test_only_one_connection_owns_the_transcript(monkeypatch):
    """Two server-side ASRs on two sockets would put every word on the shared
    timeline twice -- the failure /api/config already guards against for the
    browser's SpeechRecognition. Ownership is released when the socket drops."""
    import backend.session as session_mod
    import backend.stt.verbatim as verbatim_mod

    class _StubASR:
        def __init__(self, **kw):
            self.interim_text = ""

        def close(self):
            pass

    monkeypatch.setattr(verbatim_mod, "VerbatimASR", _StubASR)
    sess = session_mod.EchoSession(_mock_settings(monkeypatch, ASR_PROVIDER="crisper"))

    first = sess.new_audio_channels()
    second = sess.new_audio_channels()
    assert isinstance(first.asr, _StubASR)
    assert second.asr is None, "the second audio socket must not transcribe too"

    first.close()
    third = sess.new_audio_channels()
    assert isinstance(third.asr, _StubASR), "ownership must be released on disconnect"


def test_a_directly_constructed_stream_keeps_a_zero_offset():
    """Every eval harness builds AcousticStream itself and publishes the
    numbers; the offset must be inert there."""
    st = AcousticStream(model_path=None)
    assert st.clock_offset_ms == 0
    st.feed(silence(1000))
    assert st.now_ms == 1000


# ==========================================================================
# DEFECT 1 -- the acoustic model must not run on the event loop
# ==========================================================================
def test_acoustic_feed_does_not_block_the_event_loop(monkeypatch):
    """Measured: 438 ms per _run_stutter on CPU against a 125 ms hop, called
    synchronously from an awaited handler -- so enabling the SSL backend
    stalled PCM ingest, UI broadcasts and every in-flight prediction for the
    whole process, once per hop."""
    import backend.session as session_mod

    class _SlowAcoustic:
        wearer_conf = None

        def feed(self, pcm):
            time.sleep(0.30)               # what 438 ms of torch looks like
            return []

    sess = session_mod.EchoSession(_mock_settings(monkeypatch))

    async def main():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        t = asyncio.create_task(ticker())
        await asyncio.sleep(0)
        await sess.handle_audio_events(_SlowAcoustic(), silence(100))
        t.cancel()
        return ticks

    ticks = asyncio.run(main())
    assert ticks >= 5, (
        "the loop was frozen for the whole feed: %d ticks in 300 ms" % ticks)


@needs_cnn
def test_the_configured_device_reaches_the_model():
    """There was no ACOUSTIC_DEVICE knob at all: config told the operator to
    'use a machine that has a GPU' while AcousticStream took its device='cpu'
    default, so the SSL backend could never actually run on one."""
    st = AcousticStream(model_path=None, stutter_model=CNN_CKPT, device="cpu")
    assert st.device == "cpu"
    assert all(p.device.type == "cpu" for p in st.stutter.parameters())
    if torch.cuda.is_available():
        cu = AcousticStream(model_path=None, stutter_model=CNN_CKPT, device="cuda")
        assert all(p.device.type == "cuda" for p in cu.stutter.parameters())


def test_hop_constant_is_the_one_the_stream_uses():
    st = AcousticStream(model_path=None)
    assert st.hop == int(SR * HOP_MS / 1000)
