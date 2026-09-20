"""Known capture-hardware profiles and the session's active audio source.

Echo's acoustic channel consumes one wire format (PCM16 / 16 kHz / mono on
/ws/audio) from two physically different sources: a wireless lav worn on
the speaker (DJI Mic 2S) and the laptop's own mic array. The bytes look
identical; what differs is what the microphone is pointed at, and that
decides two things the rest of the stack cannot infer from the samples:

  1. Which browser processing to ask for. Chrome's AGC normalises level,
     which is exactly the evidence the level-based speaker gate reads
     (docs/PROTOCOL.md, `word.wearer_conf`); its echo canceller and noise
     suppressor were designed for a mic on a table next to speakers, not a
     capsule 5 cm from the mouth.
  2. Whether the speaker gate is meaningful at all. A lav IS on the speaker,
     so "the wearer is the loudest voice" is a physical fact. A laptop array
     on a table hears wearer and bystander at about the same distance, and
     the gate is validated on synthetic mixes only (backend/config.py).

This module is a table plus a matcher; it changes no default. The frontend
asks for the table (GET /api/audio/sources), picks a device, and reports what
it chose (POST /api/audio/source). The choice is stored on the session and
surfaced in /healthz and /api/config so an operator can see, at a glance,
which microphone the demo is actually listening through.

Everything here is ASCII: device labels on a localised Windows arrive with a
non-ASCII prefix in front of the endpoint name (the CJK word for "microphone"
followed by " (Wireless Mic Rx)"), and the matcher must find the ASCII
endpoint name inside them.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

KINDS = ("lav-wireless", "onboard-array")


@dataclass(frozen=True)
class AudioProfile:
    id: str
    display_name: str
    kind: str                                  # one of KINDS
    match: tuple[str, ...]                     # case-insensitive regexes on the device label
    constraints: dict[str, bool]               # getUserMedia audio constraints
    channel_count: int                         # channelCount to request (worklet reads ch 0)
    wearer_gate: bool                          # is the level-based speaker gate meaningful?
    notes: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["match"] = list(self.match)
        return d


# Order matters: the first profile whose pattern matches wins, so the
# specific commercial mic sits above the generic onboard patterns.
PROFILES: tuple[AudioProfile, ...] = (
    AudioProfile(
        id="dji-mic-2s",
        display_name="DJI Mic 2S (wireless lav)",
        kind="lav-wireless",
        # Windows enumerates the receiver as "Wireless Mic Rx" (USB VID_2CA3
        # PID_4015 MI_01); macOS/Chrome label it "DJI MIC" / "DJI Mic 2S".
        match=(r"wireless\s*mic\s*rx", r"\bdji\b"),
        constraints={
            "echoCancellation": False,
            "noiseSuppression": False,
            "autoGainControl": False,
        },
        channel_count=1,
        wearer_gate=True,
        notes=(
            "Receiver delivers 48 kHz / 2 ch / 24-bit; both channels are "
            "bit-identical with one transmitter, so channel 0 is the whole "
            "signal. The receiver already applies its own DSP, so Chrome's "
            "AEC/NS/AGC are redundant and AGC would erase the level evidence "
            "the speaker gate reads. TTS bleed is handled by the page dropping "
            "PCM while it speaks (frontend flushPCM), not by AEC. Measured "
            "numbers: docs/MICROPHONE.md, 'DJI Mic 2S -- measured compatibility'."
        ),
    ),
    AudioProfile(
        id="laptop-array",
        display_name="Laptop microphone array",
        kind="onboard-array",
        match=(r"realtek", r"mic(rophone)?\s*array", r"built-?in", r"internal\s*mic",
               r"macbook", r"\bsst\b"),
        # What the page requests today (frontend/app.js startAudioChannel):
        # AEC on because Echo's TTS plays out of the same laptop, NS off so a
        # prolongation is not smoothed away, AGC left at Chrome's default.
        constraints={
            "echoCancellation": True,
            "noiseSuppression": False,
            "autoGainControl": True,
        },
        channel_count=1,
        wearer_gate=False,
        notes=(
            "A mic on the table hears the wearer and a bystander at roughly "
            "the same distance, so the level-based speaker gate has no physical "
            "basis here (docs/PROTOCOL.md, 'Honest limitation'). Room tone and "
            "keyboard noise arrive at near-speech level."
        ),
    ),
)

_BY_ID = {p.id: p for p in PROFILES}
_COMPILED = tuple((p, tuple(re.compile(rx, re.IGNORECASE) for rx in p.match))
                  for p in PROFILES)


def get_profile(profile_id: str | None) -> AudioProfile | None:
    if not profile_id:
        return None
    return _BY_ID.get(str(profile_id).strip().lower())


def match_profile(label: str | None) -> AudioProfile | None:
    """First profile whose pattern appears in `label`, or None for unknown
    hardware. Case-insensitive; tolerant of the localised prefixes Windows
    puts in front of the endpoint name."""
    if not label:
        return None
    text = str(label)
    for profile, patterns in _COMPILED:
        for rx in patterns:
            if rx.search(text):
                return profile
    return None


def _opt_float(raw: object) -> float | None:
    """A dBFS floor the page measured, or None. Never raises: a bad number
    from the browser must not reject the whole source report."""
    if raw is None or isinstance(raw, bool):
        return None
    try:
        v = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


@dataclass
class AudioSource:
    """What the frontend is capturing from right now."""
    label: str
    device_id: str | None = None
    profile_id: str | None = None
    floor_dbfs: float | None = None
    set_at: float = field(default_factory=time.time)

    @classmethod
    def from_report(cls, body: dict[str, Any]) -> "AudioSource":
        """Build from the POST /api/audio/source body.

        `profile_id` null (or absent) means "you tell me": the label is run
        through the matcher. A non-null id must name a known profile so a
        typo in the page cannot silently register as unknown hardware.
        """
        label = str(body.get("label") or "").strip()
        if not label:
            raise ValueError("label is required")
        raw_pid = body.get("profile_id")
        if raw_pid is None or str(raw_pid).strip() == "":
            profile = match_profile(label)
        else:
            profile = get_profile(str(raw_pid))
            if profile is None:
                raise ValueError("unknown profile_id %r" % (raw_pid,))
        device_id = body.get("deviceId", body.get("device_id"))
        return cls(
            label=label,
            device_id=str(device_id) if device_id else None,
            profile_id=profile.id if profile else None,
            floor_dbfs=_opt_float(body.get("floor_dbfs")),
        )

    @property
    def profile(self) -> AudioProfile | None:
        return get_profile(self.profile_id)

    def to_dict(self) -> dict[str, Any]:
        p = self.profile
        return {
            "label": self.label,
            "deviceId": self.device_id,
            "profile_id": self.profile_id,
            "kind": p.kind if p else None,
            "display_name": p.display_name if p else None,
            "wearer_gate_recommended": p.wearer_gate if p else None,
            "floor_dbfs": self.floor_dbfs,
            "set_at": self.set_at,
        }


def profiles_payload() -> dict[str, Any]:
    """GET /api/audio/sources body."""
    return {"profiles": [p.to_dict() for p in PROFILES], "kinds": list(KINDS)}
