// AudioWorklet processor: forwards raw float32 mono blocks to the main thread.
// The AudioContext is created at 16 kHz, so no resampling is needed here.
//
// Each message carries the block AND its RMS: {pcm: Float32Array, rms: number}.
// The RMS is computed here, on the audio thread, over the same samples that go
// to /ws/audio -- so the level the speaker gate reasons about is exactly the
// level the server hears, with no resampling or requantisation in between.
// (main thread could recompute it from `pcm`, but doing it here keeps the two
// numbers provably from one buffer and costs one pass over 128 floats.)
//
// The PCM path SHAPE is unchanged: app.js still sends plain Int16 frames to
// /ws/audio, which the server appends verbatim. This message is
// main-thread-only plumbing.
class PCMForwarder extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch && ch.length) {
      // copy — the engine reuses the buffer
      const pcm = new Float32Array(ch);
      let sum = 0;
      for (let i = 0; i < pcm.length; i++) sum += pcm[i] * pcm[i];
      const rms = Math.sqrt(sum / pcm.length);
      // transfer the copy: the worklet has no further use for it
      this.port.postMessage({ pcm, rms }, [pcm.buffer]);
    }
    return true;
  }
}
registerProcessor("pcm-forwarder", PCMForwarder);
