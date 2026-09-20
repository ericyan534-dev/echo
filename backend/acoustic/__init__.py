"""Acoustic stall channel — hears what transcripts can't.

Consumer ASR (Chrome Web Speech, Whisper, Deepgram defaults) suppresses filled
pauses ("um/uh") and normalizes prolongations ("theeee" -> "the"), so a
transcript-only stall detector is structurally blind to the two most common
word-search signals in aphasic speech. This package analyzes raw mic PCM in
parallel with the transcript:

  features.py      log-mel frontend (shared by training + inference)
  model.py         compact CNN filler classifier (trained on PodcastFillers)
  prolongation.py  rule-based sustained-phone detector (literature-grounded)
  stream.py        sliding-window service: PCM in -> AcousticEvent out
"""
