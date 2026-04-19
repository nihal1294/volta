export class PlaybackQueue {
  private readonly audioContext: AudioContext;
  private readonly outputGain: GainNode;

  private cursor = 0;

  private readonly sources = new Set<AudioBufferSourceNode>();

  private onIdle: (() => void) | null = null;

  constructor(audioContext: AudioContext) {
    this.audioContext = audioContext;
    this.outputGain = this.audioContext.createGain();
    this.outputGain.gain.value = 1.35;
    this.outputGain.connect(this.audioContext.destination);
  }

  setOnIdle(callback: (() => void) | null) {
    this.onIdle = callback;
  }

  isIdle() {
    return this.sources.size === 0;
  }

  enqueueMonoPcm16Base64(pcm16b64: string, sampleRate: number) {
    const bytes = Uint8Array.from(atob(pcm16b64), (char) => char.charCodeAt(0));
    const int16 = new Int16Array(bytes.buffer);
    const float32 = new Float32Array(int16.length);
    for (let index = 0; index < int16.length; index += 1) {
      float32[index] = int16[index] / 32768;
    }

    const buffer = this.audioContext.createBuffer(1, float32.length, sampleRate);
    buffer.getChannelData(0).set(float32);

    const source = this.audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(this.outputGain);

    const startAt = Math.max(this.audioContext.currentTime + 0.02, this.cursor);
    this.sources.add(source);
    source.onended = () => {
      this.sources.delete(source);
      if (this.sources.size === 0) {
        this.cursor = this.audioContext.currentTime;
        this.onIdle?.();
      }
    };
    source.start(startAt);
    this.cursor = startAt + buffer.duration;
  }

  stop() {
    for (const source of this.sources) {
      try {
        source.stop();
      } catch {
        // Ignore already-ended sources.
      }
      source.disconnect();
    }
    this.sources.clear();
    this.cursor = this.audioContext.currentTime;
    this.onIdle?.();
  }
}
