const MAGIC = [0x52, 0x56, 0x4c, 0x31] as const; // RVL1
const HEADER_BYTES = 16;

export type AudioFrameInput = {
  sequence: number;
  sampleRate: number;
  pcm: Float32Array;
};

export function encodeAudioFrame({ sequence, sampleRate, pcm }: AudioFrameInput): ArrayBuffer {
  const pcm16 = new Int16Array(pcm.length);
  for (let index = 0; index < pcm.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, pcm[index] ?? 0));
    pcm16[index] = sample < 0 ? sample * 32768 : sample * 32767;
  }

  const frame = new ArrayBuffer(HEADER_BYTES + pcm16.byteLength);
  const bytes = new Uint8Array(frame);
  bytes.set(MAGIC, 0);

  const view = new DataView(frame);
  view.setUint32(4, sequence, true);
  view.setUint32(8, sampleRate, true);
  view.setUint16(12, 1, true); // mono
  view.setUint16(14, 1, true); // encoding 1 = pcm16

  bytes.set(new Uint8Array(pcm16.buffer), HEADER_BYTES);
  return frame;
}
