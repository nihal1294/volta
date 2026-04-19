import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { PlaybackQueue } from "./audio/playback-queue";
import { BrandMark, CloseIcon, InfoIcon, MoonIcon, SunIcon } from "./branding";
import { encodeAudioFrame } from "./realtime/frame";

type Screen = "console" | "persona" | "runs";
type ConnectionState = "disconnected" | "connecting" | "connected";
type InputMode = "mic" | "file";
type ThemeMode = "dark" | "light";

type TurnMetrics = {
  stt_first_partial_ms: number | null;
  stt_first_word_ms: number | null;
  stt_final_ms: number | null;
  stt_words_per_sec: number | null;
  llm_ttft_ms: number | null;
  llm_tokens_per_sec: number | null;
  tts_first_audio_ms: number | null;
  tts_realtime_factor: number | null;
  turn_total_ms: number | null;
};

type LogEntry = {
  time: string;
  type: string;
  detail: string;
};

type RunSummary = {
  id: string;
  updated_at_ms: number;
  transcript: string;
  llm_output: string;
  latest_action: string;
  input_audio_url: string | null;
  tts_audio_url: string | null;
};

type RunTurn = {
  turn_id: string;
  text: string;
  started_at_ms: number | null;
  committed_at_ms: number | null;
  audio_url: string | null;
};

type RunEvent = {
  recorded_at_ms: number;
  type: string;
  detail: string;
};

type RunDetail = RunSummary & {
  metrics: Partial<TurnMetrics>;
  turns: RunTurn[];
  timeline: RunEvent[];
};

type PersonaPayload = {
  name: string;
  source_label?: string;
  text: string;
  excerpt_lines?: string[];
  excerpt?: string;
  source_path?: string;
};

type InputDevice = {
  deviceId: string;
  label: string;
};

type VoiceOption = {
  id: string;
  name: string;
  source?: {
    path_on_server?: string;
  };
};

type CaptureSettings = {
  echoCancellation: boolean;
  noiseSuppression: boolean;
  autoGainControl: boolean;
};

const WS_URL =
  import.meta.env.VITE_VOLTA_WS_URL ??
  import.meta.env.VITE_PIPELINE_WS_URL ??
  "ws://127.0.0.1:8765/v1/realtime";
const API_BASE =
  import.meta.env.VITE_PIPELINE_API_URL ??
  WS_URL.replace(/^ws/, "http").replace(/\/v1\/realtime$/, "");
const AUDIO_FILE_ACCEPT = ".wav,.mp3,.m4a,.aac,.flac,.opus,.ogg,audio/*";
const THEME_STORAGE_KEY = "volta-theme";
const VOLTA_REPO_URL = "https://github.com/nihal1294/volta";

const EMPTY_METRICS: TurnMetrics = {
  stt_first_partial_ms: null,
  stt_first_word_ms: null,
  stt_final_ms: null,
  stt_words_per_sec: null,
  llm_ttft_ms: null,
  llm_tokens_per_sec: null,
  tts_first_audio_ms: null,
  tts_realtime_factor: null,
  turn_total_ms: null,
};

const AUTO_SCROLL_THRESHOLD_PX = 48;

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

function formatMetric(value: number | null, unit: string) {
  if (value === null) {
    return "Pending";
  }
  return `${value.toFixed(2)}${unit}`;
}

function formatTimestamp(valueMs: number) {
  return new Date(valueMs).toLocaleString();
}

function formatTurnDuration(startedAtMs: number | null, committedAtMs: number | null) {
  if (startedAtMs === null || committedAtMs === null) {
    return "Pending";
  }
  const durationMs = Math.max(0, committedAtMs - startedAtMs);
  return `${(durationMs / 1000).toFixed(2)}s`;
}

function snippet(text: string, fallback: string) {
  const trimmed = text.trim();
  if (!trimmed) {
    return fallback;
  }
  if (trimmed.length <= 180) {
    return trimmed;
  }
  return `${trimmed.slice(0, 180).trim()}…`;
}

function normalizePersonaLine(line: string) {
  return line
    .replace(/^#{1,6}\s*/, "")
    .replace(/^[-*+]\s*/, "")
    .replace(/\[(.*?)\]\([^)]+\)/g, "$1")
    .replace(/`([^`]*)`/g, "$1")
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/__(.*?)__/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
}

function buildPersonaPreview(text: string) {
  const sectionHeadings = new Set([
    "persona",
    "core role",
    "behavior",
    "listening and reaction rules",
    "investor lens",
    "style",
    "game framing",
    "conversation flow",
    "output constraints",
  ]);
  const lines = text
    .split(/\r?\n/)
    .map(normalizePersonaLine)
    .filter(Boolean)
    .filter((line) => !sectionHeadings.has(line.toLowerCase()));

  return {
    summary: lines.slice(0, 2),
    highlights: lines.slice(2, 8),
  };
}

function resolveInitialTheme(): ThemeMode {
  if (typeof window === "undefined") {
    return "dark";
  }
  const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
  if (stored === "dark" || stored === "light") {
    return stored;
  }
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function applyTheme(theme: ThemeMode) {
  document.documentElement.dataset.theme = theme;
}

function useAutoScrollPane(content: string) {
  const ref = useRef<HTMLPreElement | null>(null);
  const shouldStickRef = useRef(true);

  const handleScroll = () => {
    const element = ref.current;
    if (!element) {
      return;
    }
    const distanceFromBottom =
      element.scrollHeight - element.clientHeight - element.scrollTop;
    shouldStickRef.current = distanceFromBottom <= AUTO_SCROLL_THRESHOLD_PX;
  };

  useLayoutEffect(() => {
    const element = ref.current;
    if (!element || !shouldStickRef.current) {
      return;
    }
    element.scrollTop = element.scrollHeight;
  }, [content]);

  return { ref, handleScroll };
}

export function App() {
  const [screen, setScreen] = useState<Screen>("console");
  const [connection, setConnection] = useState<ConnectionState>("disconnected");
  const [mode, setMode] = useState<InputMode>("mic");
  const [pipelineState, setPipelineState] = useState("idle");
  const [voice, setVoice] = useState("");
  const [instruct, setInstruct] = useState(
    "Very angry, aggressive, intense, raised voice, confrontational delivery, emotionally expressive delivery.",
  );
  const [sttPartial, setSttPartial] = useState("");
  const [sttFinal, setSttFinal] = useState("");
  const [llmText, setLlmText] = useState("");
  const [llmAction, setLlmAction] = useState("");
  const [metrics, setMetrics] = useState<TurnMetrics>(EMPTY_METRICS);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [recording, setRecording] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedFileUrl, setSelectedFileUrl] = useState<string | null>(null);
  const [showDiagnostics, setShowDiagnostics] = useState(false);
  const [showCaptureTools, setShowCaptureTools] = useState(false);
  const [theme, setTheme] = useState<ThemeMode>(() => resolveInitialTheme());
  const [showAbout, setShowAbout] = useState(false);

  const [defaultPersona, setDefaultPersona] = useState<PersonaPayload | null>(null);
  const [personaDraft, setPersonaDraft] = useState("");
  const [personaStatus, setPersonaStatus] = useState("Loading default persona…");
  const [showPersonaEditor, setShowPersonaEditor] = useState(false);

  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [selectedRunDetail, setSelectedRunDetail] = useState<RunDetail | null>(null);
  const [runsLoading, setRunsLoading] = useState(false);
  const [runDetailLoading, setRunDetailLoading] = useState(false);

  const [availableInputs, setAvailableInputs] = useState<InputDevice[]>([]);
  const [selectedInputDeviceId, setSelectedInputDeviceId] = useState("");
  const [captureSettings, setCaptureSettings] = useState<CaptureSettings>({
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
  });
  const [inputGain, setInputGain] = useState(1.35);
  const [inputLevel, setInputLevel] = useState(0);
  const [inputPeak, setInputPeak] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const connectPromiseRef = useRef<Promise<void> | null>(null);
  const audioFileInputRef = useRef<HTMLInputElement | null>(null);
  const inputPreviewRef = useRef<HTMLAudioElement | null>(null);
  const personaFileInputRef = useRef<HTMLInputElement | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const playbackQueueRef = useRef<PlaybackQueue | null>(null);
  const defaultVoiceRef = useRef("");
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const captureNodeRef = useRef<AudioWorkletNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const highpassRef = useRef<BiquadFilterNode | null>(null);
  const lowpassRef = useRef<BiquadFilterNode | null>(null);
  const compressorRef = useRef<DynamicsCompressorNode | null>(null);
  const gainRef = useRef<GainNode | null>(null);
  const limiterRef = useRef<DynamicsCompressorNode | null>(null);
  const workletLoadedRef = useRef(false);
  const audioSequenceRef = useRef(0);
  const modeRef = useRef<InputMode>(mode);
  const recordingRef = useRef(recording);
  const pendingAudioDoneRef = useRef(false);
  const sttPartialPane = useAutoScrollPane(sttPartial);
  const sttFinalPane = useAutoScrollPane(sttFinal);
  const llmTextPane = useAutoScrollPane(llmText);

  useEffect(() => {
    document.title = "Volta";
    applyTheme(theme);
    void fetchPersona();
    void fetchRuns();
    void refreshInputDevices();

    const handleDeviceChange = () => {
      void refreshInputDevices();
    };

    navigator.mediaDevices?.addEventListener?.("devicechange", handleDeviceChange);

    return () => {
      navigator.mediaDevices?.removeEventListener?.("devicechange", handleDeviceChange);
      void stopMicCapture(false);
      stopInputPreview();
      stopPlayback();
      wsRef.current?.close();
    };
  }, []);

  useEffect(() => {
    applyTheme(theme);
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  useEffect(() => {
    let canceled = false;
    fetchJson<VoiceOption[]>("/v1/voices")
      .then((payload) => {
        if (canceled) {
          return;
        }
        const defaultVoice = payload[0]?.name?.trim();
        if (defaultVoice) {
          setVoice((current) =>
            !current.trim() || current === defaultVoiceRef.current ? defaultVoice : current,
          );
          defaultVoiceRef.current = defaultVoice;
        }
      })
      .catch(() => {
        /* keep current fallback voice if the API is unavailable */
      });
    return () => {
      canceled = true;
    };
  }, []);

  useEffect(() => {
    if (!showAbout) {
      return;
    }
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setShowAbout(false);
      }
    };
    window.addEventListener("keydown", handleEscape);
    return () => {
      window.removeEventListener("keydown", handleEscape);
    };
  }, [showAbout]);

  useEffect(() => {
    if (!selectedFile) {
      setSelectedFileUrl((current) => {
        if (current) {
          URL.revokeObjectURL(current);
        }
        return null;
      });
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setSelectedFileUrl((current) => {
      if (current) {
        URL.revokeObjectURL(current);
      }
      return objectUrl;
    });
    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [selectedFile]);

  useEffect(() => {
    if (gainRef.current) {
      gainRef.current.gain.value = inputGain;
    }
  }, [inputGain]);

  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  useEffect(() => {
    recordingRef.current = recording;
  }, [recording]);

  useEffect(() => {
    if (screen === "runs") {
      void fetchRuns(null);
    }
  }, [screen]);

  useEffect(() => {
    if (screen !== "runs") {
      return;
    }
    if (!selectedRunId) {
      setSelectedRunDetail(null);
      return;
    }
    void fetchRunDetail(selectedRunId);
  }, [screen, selectedRunId]);

  const metricTiles = useMemo(
    () => [
      { label: "STT first partial", value: formatMetric(metrics.stt_first_partial_ms, "ms") },
      { label: "STT first word", value: formatMetric(metrics.stt_first_word_ms, "ms") },
      { label: "STT final", value: formatMetric(metrics.stt_final_ms, "ms") },
      { label: "STT words/sec", value: formatMetric(metrics.stt_words_per_sec, "") },
      { label: "LLM TTFT", value: formatMetric(metrics.llm_ttft_ms, "ms") },
      { label: "LLM tok/sec", value: formatMetric(metrics.llm_tokens_per_sec, "") },
      { label: "TTS first audio", value: formatMetric(metrics.tts_first_audio_ms, "ms") },
      { label: "TTS RTF", value: formatMetric(metrics.tts_realtime_factor, "x") },
      { label: "Turn total", value: formatMetric(metrics.turn_total_ms, "ms") },
    ],
    [metrics],
  );

  const statusTone =
    pipelineState === "speaking"
      ? "Speaking"
      : pipelineState === "thinking"
        ? "Thinking"
        : pipelineState === "listening"
          ? "Listening"
          : "Idle";

  const meterPercent = Math.max(0, Math.min(100, Math.round(inputLevel * 1000)));
  const peakPercent = Math.max(0, Math.min(100, Math.round(inputPeak * 1000)));
  const consoleMetricTiles = [metricTiles[0], metricTiles[4], metricTiles[6], metricTiles[8]];
  const fileRunBusy = mode === "file" && ["listening", "thinking", "speaking"].includes(pipelineState);
  const personaPreview = useMemo(() => buildPersonaPreview(personaDraft), [personaDraft]);
  const nextTheme = theme === "dark" ? "light" : "dark";

  function log(type: string, detail: string) {
    const time = new Date().toLocaleTimeString();
    setLogs((current) => [{ time, type, detail }, ...current].slice(0, 250));
  }

  async function fetchPersona() {
    try {
      const payload = await fetchJson<PersonaPayload>("/v1/persona");
      const previewFallback = buildPersonaPreview(payload.text);
      const fallbackLines =
        payload.excerpt_lines && payload.excerpt_lines.length > 0
          ? payload.excerpt_lines
          : previewFallback.summary.concat(previewFallback.highlights).slice(0, 6);
      const normalizedPayload: PersonaPayload = {
        ...payload,
        source_label: payload.source_label ?? "Bundled default persona",
        excerpt_lines: fallbackLines,
      };
      setDefaultPersona(normalizedPayload);
      setPersonaDraft(normalizedPayload.text);
      setPersonaStatus("Bundled default persona ready.");
    } catch (error) {
      setPersonaStatus(`Unable to load persona: ${String(error)}`);
      log("error", `persona load failed: ${String(error)}`);
    }
  }

  async function fetchRuns(preferredId?: string | null) {
    setRunsLoading(true);
    try {
      const payload = await fetchJson<{ runs: RunSummary[] }>("/v1/runs");
      setRuns(payload.runs);
      const candidateId =
        screen === "runs"
          ? preferredId === undefined
            ? selectedRunId
            : preferredId
          : preferredId ?? payload.runs[0]?.id ?? null;
      const nextSelectedId =
        payload.runs.find((run) => run.id === candidateId)?.id ?? payload.runs[0]?.id ?? null;
      setSelectedRunId(nextSelectedId);
      if (screen === "runs") {
        if (nextSelectedId) {
          await fetchRunDetail(nextSelectedId);
        } else {
          setSelectedRunDetail(null);
        }
      }
    } catch (error) {
      log("error", `runs load failed: ${String(error)}`);
    } finally {
      setRunsLoading(false);
    }
  }

  async function fetchRunDetail(runId: string) {
    setRunDetailLoading(true);
    try {
      const payload = await fetchJson<RunDetail>(`/v1/runs/${runId}`);
      setSelectedRunDetail(payload);
    } catch (error) {
      setSelectedRunDetail(null);
      log("error", `run detail load failed: ${String(error)}`);
    } finally {
      setRunDetailLoading(false);
    }
  }

  async function refreshInputDevices() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const audioInputs = devices
        .filter((device) => device.kind === "audioinput")
        .map((device, index) => ({
          deviceId: device.deviceId,
          label: device.label || `Input ${index + 1}`,
        }));
      setAvailableInputs(audioInputs);
      setSelectedInputDeviceId((current) => {
        if (current && audioInputs.some((device) => device.deviceId === current)) {
          return current;
        }
        return audioInputs[0]?.deviceId ?? "";
      });
    } catch (error) {
      log("error", `device enumeration failed: ${String(error)}`);
    }
  }

  async function ensureAudioContext() {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext({ latencyHint: "interactive" });
      playbackQueueRef.current = new PlaybackQueue(audioContextRef.current);
      playbackQueueRef.current.setOnIdle(() => {
        if (!pendingAudioDoneRef.current) {
          return;
        }
        pendingAudioDoneRef.current = false;
        setPipelineState(modeRef.current === "mic" && recordingRef.current ? "listening" : "idle");
        void fetchRuns();
      });
    }
    if (audioContextRef.current.state === "suspended") {
      await audioContextRef.current.resume();
    }
    return audioContextRef.current;
  }

  async function connect() {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      return;
    }
    if (connectPromiseRef.current) {
      await connectPromiseRef.current;
      return;
    }

    setConnection("connecting");
    connectPromiseRef.current = new Promise<void>((resolve, reject) => {
      let opened = false;
      const ws = new WebSocket(WS_URL, "realtime");
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => {
        opened = true;
        setConnection("connected");
        sendSessionUpdate();
        log("session", "connected");
        connectPromiseRef.current = null;
        resolve();
      };

      ws.onclose = () => {
        wsRef.current = null;
        connectPromiseRef.current = null;
        setConnection("disconnected");
        setPipelineState("idle");
        void stopMicCapture(false);
        log("session", "disconnected");
        if (!opened) {
          reject(new Error("websocket closed before opening"));
        }
      };

      ws.onerror = () => {
        log("error", "websocket error");
        if (!opened) {
          connectPromiseRef.current = null;
          reject(new Error("websocket connection failed"));
        }
      };

      ws.onmessage = (event) => {
        if (typeof event.data !== "string") {
          return;
        }
        const message = JSON.parse(event.data) as Record<string, unknown>;
        handleServerMessage(message);
      };
    });

    await connectPromiseRef.current;
  }

  async function handleConnect() {
    await connect();
    if (mode === "mic" && !recording) {
      await startMicCapture();
    }
  }

  function disconnect() {
    void stopMicCapture(false);
    stopInputPreview();
    stopPlayback();
    wsRef.current?.close();
    wsRef.current = null;
    connectPromiseRef.current = null;
    setConnection("disconnected");
    setPipelineState("idle");
  }

  function send(message: Record<string, unknown>) {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return;
    }
    ws.send(JSON.stringify(message));
  }

  function sendBinary(buffer: ArrayBuffer) {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return;
    }
    if (ws.bufferedAmount > 2_000_000) {
      log("transport", "uplink backpressure; frame dropped");
      return;
    }
    ws.send(buffer);
  }

  function sendSessionUpdate() {
    send({
      type: "session.update",
      session: {
        instructions: { type: "conversation", language: "en" },
        voice,
        allow_recording: false,
        output_audio_format: "pcm16",
        tts_instruct: instruct,
        persona_text: personaDraft,
      },
    });
  }

  function handleServerMessage(message: Record<string, unknown>) {
    const type = String(message.type ?? "");
    if (type === "session.updated") {
      const session = (message.session as Record<string, unknown> | undefined) ?? {};
      setPipelineState("idle");
      const sessionVoice = String(session.voice ?? voice);
      setVoice(sessionVoice);
      if (sessionVoice.trim()) {
        defaultVoiceRef.current = sessionVoice;
      }
      setInstruct(String(session.tts_instruct ?? instruct));
      if (typeof session.persona_text === "string" && session.persona_text.trim()) {
        setPersonaDraft(session.persona_text);
      }
      log("session.updated", String(session.voice ?? voice));
      return;
    }
    if (type === "conversation.item.input_audio_transcription.delta") {
      const delta = String(message.delta ?? "");
      const text = typeof message.text === "string" ? String(message.text) : "";
      const fullText = typeof message.full_text === "string" ? String(message.full_text) : "";
      const visibleText = fullText.trim() ? fullText : text;
      if (visibleText.trim()) {
        setSttPartial(visibleText);
        setSttFinal(visibleText);
      } else {
        setSttPartial((current) => current + delta);
        setSttFinal((current) => current + delta);
      }
      if (visibleText.trim() || delta.trim()) {
        setPipelineState("listening");
      }
      return;
    }
    if (type === "response.created") {
      setPipelineState("thinking");
      log("response.created", String((message.response as Record<string, unknown> | undefined)?.id ?? ""));
      return;
    }
    if (type === "response.action") {
      const action = (message.action as Record<string, unknown> | undefined) ?? {};
      const name = String(action.name ?? "");
      setLlmAction(name);
      if (name === "speak" || name === "continue_speaking") {
        setPipelineState("thinking");
      }
      log("llm.action", name || "none");
      return;
    }
    if (type === "response.text.delta") {
      const text = String(message.delta ?? "");
      setLlmText((current) => current + text);
      return;
    }
    if (type === "response.audio.delta") {
      const pcm16 = String(message.delta ?? "");
      const sampleRate = Number(message.sample_rate ?? 24000);
      pendingAudioDoneRef.current = false;
      setPipelineState("speaking");
      void playPcm16Chunk(pcm16, sampleRate).catch((error) => {
        log("error", `audio playback failed: ${String(error)}`);
      });
      return;
    }
    if (type === "response.audio.done") {
      log("response.audio.done", String(message.response_id ?? ""));
      pendingAudioDoneRef.current = true;
      if (!playbackQueueRef.current || playbackQueueRef.current.isIdle()) {
        pendingAudioDoneRef.current = false;
        setPipelineState(modeRef.current === "mic" && recordingRef.current ? "listening" : "idle");
        void fetchRuns();
      }
      return;
    }
    if (type === "turn.metrics") {
      const payload = (message.metrics as Partial<TurnMetrics> | undefined) ?? {};
      setMetrics((current) => ({
        ...current,
        ...payload,
      }));
      return;
    }
    if (type === "error") {
      log("error", String(message.message ?? "unknown"));
    }
  }

  async function playPcm16Chunk(pcm16b64: string, sampleRate: number) {
    const audioContext = await ensureAudioContext();
    playbackQueueRef.current ??= new PlaybackQueue(audioContext);
    playbackQueueRef.current.enqueueMonoPcm16Base64(pcm16b64, sampleRate);
  }

  function stopPlayback() {
    pendingAudioDoneRef.current = false;
    playbackQueueRef.current?.stop();
  }

  function stopInputPreview() {
    if (!inputPreviewRef.current) {
      return;
    }
    inputPreviewRef.current.pause();
    inputPreviewRef.current.currentTime = 0;
  }

  function downmixAudioBuffer(buffer: AudioBuffer) {
    const mono = new Float32Array(buffer.length);
    for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
      const channelData = buffer.getChannelData(channel);
      for (let index = 0; index < channelData.length; index += 1) {
        mono[index] += channelData[index] / buffer.numberOfChannels;
      }
    }
    return mono;
  }

  async function startMicCapture() {
    if (recording) {
      return;
    }
    if (connection !== "connected") {
      await connect();
    }
    const audioContext = await ensureAudioContext();
    if (!workletLoadedRef.current) {
      await audioContext.audioWorklet.addModule(new URL("./audio/mic-capture.worklet.ts", import.meta.url));
      workletLoadedRef.current = true;
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        deviceId: selectedInputDeviceId ? { exact: selectedInputDeviceId } : undefined,
        echoCancellation: captureSettings.echoCancellation,
        noiseSuppression: captureSettings.noiseSuppression,
        autoGainControl: captureSettings.autoGainControl,
      },
    });
    await refreshInputDevices();

    mediaStreamRef.current = stream;
    const source = audioContext.createMediaStreamSource(stream);
    sourceRef.current = source;

    const highpass = new BiquadFilterNode(audioContext, { type: "highpass", frequency: 90, Q: 0.7 });
    const lowpass = new BiquadFilterNode(audioContext, { type: "lowpass", frequency: 7200, Q: 0.7 });
    const compressor = new DynamicsCompressorNode(audioContext, {
      threshold: -26,
      knee: 18,
      ratio: 3.2,
      attack: 0.004,
      release: 0.16,
    });
    const gain = new GainNode(audioContext, { gain: inputGain });
    const limiter = new DynamicsCompressorNode(audioContext, {
      threshold: -3,
      knee: 0,
      ratio: 20,
      attack: 0.002,
      release: 0.08,
    });

    highpassRef.current = highpass;
    lowpassRef.current = lowpass;
    compressorRef.current = compressor;
    gainRef.current = gain;
    limiterRef.current = limiter;

    const captureNode = new AudioWorkletNode(audioContext, "mic-capture", {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      channelCount: 1,
    });
    captureNodeRef.current = captureNode;

    sendSessionUpdate();
    stopPlayback();

    captureNode.port.onmessage = (event) => {
      const data = event.data;
      if (!(data instanceof Float32Array)) {
        return;
      }
      let sumSquares = 0;
      let peak = 0;
      for (let index = 0; index < data.length; index += 1) {
        const sample = data[index] ?? 0;
        sumSquares += sample * sample;
        peak = Math.max(peak, Math.abs(sample));
      }
      const rms = Math.sqrt(sumSquares / Math.max(1, data.length));
      setInputLevel((current) => current * 0.76 + rms * 0.24);
      setInputPeak((current) => Math.max(current * 0.84, peak));

      const frame = encodeAudioFrame({
        sequence: (audioSequenceRef.current += 1),
        sampleRate: audioContext.sampleRate,
        pcm: data,
      });
      sendBinary(frame);
    };

    source.connect(highpass).connect(lowpass).connect(compressor).connect(gain).connect(limiter).connect(captureNode);
    setMetrics(EMPTY_METRICS);
    setSttPartial("");
    setSttFinal("");
    setLlmText("");
    setLlmAction("");
    setPipelineState("listening");
    setRecording(true);
    log("mic", `capturing at ${audioContext.sampleRate} Hz`);
  }

  async function stopMicCapture(sendEnd = true) {
    if (sendEnd) {
      send({ type: "input_audio_buffer.commit" });
    }
    captureNodeRef.current?.disconnect();
    sourceRef.current?.disconnect();
    highpassRef.current?.disconnect();
    lowpassRef.current?.disconnect();
    compressorRef.current?.disconnect();
    gainRef.current?.disconnect();
    limiterRef.current?.disconnect();
    captureNodeRef.current = null;
    sourceRef.current = null;
    highpassRef.current = null;
    lowpassRef.current = null;
    compressorRef.current = null;
    gainRef.current = null;
    limiterRef.current = null;
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    mediaStreamRef.current = null;
    setInputLevel(0);
    setInputPeak(0);
    setRecording(false);
  }

  async function submitFile() {
    if (!selectedFile) {
      return;
    }
    if (connection !== "connected") {
      await connect();
    }
    sendSessionUpdate();
    stopPlayback();
    setMetrics(EMPTY_METRICS);
    setSttPartial("");
    setSttFinal("");
    setLlmText("");
    setLlmAction("");
    setPipelineState("listening");
    if (selectedFileUrl && inputPreviewRef.current) {
      try {
        inputPreviewRef.current.currentTime = 0;
        await inputPreviewRef.current.play();
      } catch (error) {
        log("audio", `input preview playback failed: ${String(error)}`);
      }
    }

    const audioContext = await ensureAudioContext();
    const arrayBuffer = await selectedFile.arrayBuffer();
    const decoded = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    const mono = downmixAudioBuffer(decoded);
    const chunkSize = Math.max(1, Math.round(decoded.sampleRate * 0.08));
    let nextChunkAtMs = performance.now();
    for (let cursor = 0; cursor < mono.length; cursor += chunkSize) {
      const slice = mono.slice(cursor, cursor + chunkSize);
      const frame = encodeAudioFrame({
        sequence: (audioSequenceRef.current += 1),
        sampleRate: decoded.sampleRate,
        pcm: slice,
      });
      sendBinary(frame);
      nextChunkAtMs += (slice.length / decoded.sampleRate) * 1000;
      const waitMs = Math.max(0, nextChunkAtMs - performance.now());
      if (waitMs > 0) {
        await new Promise((resolve) => setTimeout(resolve, waitMs));
      }
    }
    send({ type: "input_audio_buffer.commit" });
    setPipelineState("thinking");
    log("file", `uploaded ${selectedFile.name}`);
  }

  async function handleModeChange(nextMode: InputMode) {
    if (mode === nextMode) {
      return;
    }
    setMode(nextMode);
    if (nextMode === "file" && recording) {
      await stopMicCapture(false);
      setPipelineState("idle");
      return;
    }
    if (nextMode === "mic" && connection === "connected" && !recording) {
      await startMicCapture();
    }
  }

  function resetPersonaToDefault() {
    if (!defaultPersona) {
      return;
    }
    setPersonaDraft(defaultPersona.text);
    setPersonaStatus("Reset to bundled default persona.");
  }

  async function handlePersonaUpload(file: File | null) {
    if (!file) {
      return;
    }
    const text = await file.text();
    setPersonaDraft(text);
    setShowPersonaEditor(true);
    setPersonaStatus(`Loaded custom persona draft from ${file.name}.`);
  }

  function applyPersonaToSession() {
    sendSessionUpdate();
    setPersonaStatus(connection === "connected" ? "Applied to live session." : "Saved locally for the next session.");
    log("persona", "session prompt updated");
  }

  function renderConsoleScreen() {
    return (
      <div className="screen-grid console-grid">
        <aside className="panel control-panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Console</p>
              <h2>Live session</h2>
            </div>
            <span className={`status-pill subtle ${connection === "connected" ? "state-listening" : "state-idle"}`}>
              {connection}
            </span>
          </div>

          <div className="control-stack">
            <div className="control-card emphasis-card">
              <div className="button-row">
                <button
                  className="button-primary"
                  onClick={() => void handleConnect()}
                  disabled={connection !== "disconnected"}
                >
                  Connect
                </button>
                <button className="button-secondary" onClick={disconnect} disabled={connection === "disconnected"}>
                  Disconnect
                </button>
              </div>
              <div className="mode-switch" role="tablist" aria-label="Input mode">
                <button
                  className={mode === "mic" ? "mode-chip active" : "mode-chip"}
                  onClick={() => {
                    void handleModeChange("mic");
                  }}
                >
                  Live mic
                </button>
                <button
                  className={mode === "file" ? "mode-chip active" : "mode-chip"}
                  onClick={() => {
                    void handleModeChange("file");
                  }}
                >
                  Audio file
                </button>
              </div>
            </div>

            <div className="control-card compact-card">
              <div className="card-label-row">
                <span>Session</span>
                <strong className={`connection-chip ${connection}`}>{statusTone}</strong>
              </div>
              <p className="microcopy">
                Full pipeline only: capture, transcript stabilization, floor-taking, and streamed speech output.
              </p>
            </div>

            {mode === "mic" ? (
              <>
                <div className="control-card">
                  <div className="card-label-row">
                    <span>Mic capture</span>
                    <strong>{recording ? `${meterPercent}%` : "offline"}</strong>
                  </div>
                  <label className="field">
                    <span>Input device</span>
                    <select
                      value={selectedInputDeviceId}
                      onChange={(event) => setSelectedInputDeviceId(event.target.value)}
                      disabled={recording}
                    >
                      {availableInputs.length === 0 ? <option value="">No devices found</option> : null}
                      {availableInputs.map((device) => (
                        <option key={device.deviceId} value={device.deviceId}>
                          {device.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <div className="meter-panel inline-meter">
                    <div className="meter-track">
                      <div className="meter-fill" style={{ width: `${recording ? meterPercent : 0}%` }} />
                      <div className="meter-peak" style={{ left: `${recording ? peakPercent : 0}%` }} />
                    </div>
                    <p className="microcopy">
                      Browser chain: high-pass, low-pass, compressor, gain, limiter, then binary uplink.
                    </p>
                  </div>
                  <button className="button-tertiary" onClick={() => setShowCaptureTools((current) => !current)}>
                    {showCaptureTools ? "Hide input tuning" : "Show input tuning"}
                  </button>
                  {showCaptureTools ? (
                    <div className="stack-block">
                      <label className="field">
                        <span>Input gain</span>
                        <input
                          type="range"
                          min="0.8"
                          max="2.1"
                          step="0.05"
                          value={inputGain}
                          onChange={(event) => setInputGain(Number(event.target.value))}
                        />
                        <small>{inputGain.toFixed(2)}x voice lift</small>
                      </label>

                      <div className="switch-list" role="group" aria-label="Microphone capture processing">
                        {(
                          [
                            ["echoCancellation", "Echo cancellation"],
                            ["noiseSuppression", "Noise suppression"],
                            ["autoGainControl", "Auto gain control"],
                          ] as const
                        ).map(([key, label]) => (
                          <button
                            key={key}
                            type="button"
                            role="switch"
                            aria-checked={captureSettings[key]}
                            className={captureSettings[key] ? "switch-row active" : "switch-row"}
                            disabled={recording}
                            onClick={() =>
                              setCaptureSettings((current) => ({
                                ...current,
                                [key]: !current[key],
                              }))
                            }
                          >
                            <span className="switch-row-label">{label}</span>
                            <span className="switch-control" aria-hidden="true">
                              <span className="switch-thumb" />
                            </span>
                          </button>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </div>
              </>
            ) : (
              <div className="control-card file-panel">
                <div className="card-label-row">
                  <span>Audio file</span>
                  <strong>{selectedFile ? "ready" : "waiting"}</strong>
                </div>
                <input
                  ref={audioFileInputRef}
                  className="visually-hidden-input"
                  hidden
                  type="file"
                  accept={AUDIO_FILE_ACCEPT}
                  onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
                />
                <div className="file-picker-shell">
                  <div className="file-selection-summary">
                    <span>{selectedFile ? "Selected file" : "No file selected"}</span>
                    <strong>{selectedFile ? selectedFile.name : "Choose an audio file to run through the full loop"}</strong>
                    <p>
                      Supported formats include `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, and `.opus`.
                    </p>
                  </div>
                  {selectedFileUrl ? (
                    <audio
                      ref={inputPreviewRef}
                      className="inline-audio-player"
                      controls
                      preload="metadata"
                      src={selectedFileUrl}
                    />
                  ) : null}
                  <div className="button-row file-button-row">
                    <button
                      className="button-secondary"
                      onClick={() => {
                        if (audioFileInputRef.current) {
                          audioFileInputRef.current.value = "";
                          audioFileInputRef.current.click();
                        }
                      }}
                    >
                      Choose audio file
                    </button>
                    <button
                      className="button-tertiary"
                      onClick={() => {
                        stopInputPreview();
                        setSelectedFile(null);
                      }}
                      disabled={!selectedFile || fileRunBusy}
                    >
                      Clear
                    </button>
                  </div>
                </div>
                <div className="file-drop file-drop-hint">
                  <span>Volta only runs the full STT → LLM → TTS pipeline. File mode is not a standalone transcription tool.</span>
                </div>
                <button className="button-primary" onClick={submitFile} disabled={!selectedFile || fileRunBusy}>
                  Run audio file
                </button>
              </div>
            )}

            <div className="control-card">
              <div className="card-label-row">
                <span>Speech output</span>
                <strong>{voice}</strong>
              </div>
              <div className="field-grid speech-output-grid">
                <label className="field compact-field">
                  <span>Voice</span>
                  <input className="compact-input" value={voice} onChange={(event) => setVoice(event.target.value)} />
                </label>
                <label className="field">
                  <span>TTS style</span>
                  <textarea
                    className="tts-style-input"
                    rows={4}
                    value={instruct}
                    onChange={(event) => setInstruct(event.target.value)}
                  />
                </label>
              </div>
            </div>
          </div>
        </aside>

        <section className="panel conversation-panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Conversation</p>
              <h2>Live conversation</h2>
            </div>
            <span className={`status-pill state-${pipelineState}`}>{statusTone}</span>
          </div>

          <div className="conversation-surface">
            <article className="stream-card">
              <div className="card-head">
                <p className="eyebrow">User feed</p>
                <h3>Transcript</h3>
              </div>
              <div className="stream-block">
                <span>Live delta</span>
                <pre ref={sttPartialPane.ref} onScroll={sttPartialPane.handleScroll}>
                  {sttPartial || "Waiting for speech and transcript stabilization…"}
                </pre>
              </div>
              <div className="stream-block">
                <span>Committed transcript</span>
                <pre ref={sttFinalPane.ref} onScroll={sttFinalPane.handleScroll}>
                  {sttFinal || "No committed speech yet."}
                </pre>
              </div>
            </article>

            <article className="stream-card response-tone">
              <div className="card-head">
                <p className="eyebrow">Assistant feed</p>
                <h3>Response output</h3>
              </div>
              <div className="action-banner">
                <span>Current action</span>
                <strong>{llmAction || "Awaiting floor decision"}</strong>
              </div>
              <div className="assistant-copy">
                <pre ref={llmTextPane.ref} onScroll={llmTextPane.handleScroll}>
                  {llmText || "Volta will stream the assistant reply here once the LLM decides to take the floor."}
                </pre>
              </div>
            </article>
          </div>

          <div className="metrics-strip">
            {consoleMetricTiles.map((item) => (
              <div className="metric-chip" key={item.label}>
                <span>{item.label}</span>
                <strong>{item.value}</strong>
              </div>
            ))}
          </div>

          <div className="diagnostics-shell">
            <div className="diagnostics-head">
              <div>
                <p className="eyebrow">Diagnostics</p>
                <h3>Runtime events</h3>
              </div>
              <button className="button-tertiary" onClick={() => setShowDiagnostics((current) => !current)}>
                {showDiagnostics ? "Hide" : "Show"}
              </button>
            </div>
            {showDiagnostics ? (
              <ul className="log-list">
                {logs.length === 0 ? (
                  <li className="empty-log">No events yet. Connect and start a live session.</li>
                ) : (
                  logs.map((entry, index) => (
                    <li key={`${entry.time}-${entry.type}-${index}`}>
                      <span>{entry.time}</span>
                      <strong>{entry.type}</strong>
                      <code>{entry.detail}</code>
                    </li>
                  ))
                )}
              </ul>
            ) : (
              <p className="microcopy diagnostics-placeholder">
                Diagnostics stay tucked away by default so the console keeps its focus on the live interaction.
              </p>
            )}
          </div>
        </section>
      </div>
    );
  }

  function renderPersonaScreen() {
    return (
      <div className="screen-grid persona-grid">
        <aside className="panel persona-sidebar">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Persona</p>
              <h2>Assistant Persona</h2>
            </div>
          </div>

          <section className="persona-sidebar-card">
            <span className="persona-card-label">Default persona</span>
            <h3>{defaultPersona?.name ?? "Loading…"}</h3>
            <ul className="persona-bullet-list compact">
              {(defaultPersona?.excerpt_lines ?? []).length > 0 ? (
                (defaultPersona?.excerpt_lines ?? []).map((line) => <li key={line}>{line}</li>)
              ) : (
                <li>Loading default persona summary…</li>
              )}
            </ul>
          </section>

          <section className="persona-sidebar-card">
            <div className="persona-status-head">
              <div>
                <span className="persona-card-label">Session state</span>
                <h3>{connection === "connected" ? "Live session attached" : "Offline draft"}</h3>
              </div>
              <span className="status-pill subtle">{connection === "connected" ? "live" : "draft"}</span>
            </div>
            <p className="persona-supporting-copy">{personaStatus}</p>
            <p className="persona-meta-copy">{defaultPersona?.source_label ?? "Bundled default persona"}</p>
          </section>

          <section className="persona-sidebar-card">
            <input
              ref={personaFileInputRef}
              className="visually-hidden-input"
              hidden
              type="file"
              accept=".md,.txt,text/markdown,text/plain"
              onChange={(event) => {
                void handlePersonaUpload(event.target.files?.[0] ?? null);
              }}
            />
            <div className="persona-upload-stack">
              <span className="persona-card-label">Persona file</span>
              <h3>Upload `.md` or `.txt` persona</h3>
              <p className="persona-supporting-copy">
                Replace the editable draft without touching the runtime protocol rules that the backend appends later.
              </p>
              <button
                className="button-secondary"
                onClick={() => {
                  if (personaFileInputRef.current) {
                    personaFileInputRef.current.value = "";
                    personaFileInputRef.current.click();
                  }
                }}
              >
                Choose persona file
              </button>
            </div>
            <div className="persona-action-stack">
              <button className="button-secondary" onClick={resetPersonaToDefault} disabled={!defaultPersona}>
                Reset to default
              </button>
              <button className="button-primary" onClick={applyPersonaToSession}>
                Apply to pipeline
              </button>
            </div>
          </section>
        </aside>

        <section className="panel persona-main">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Persona</p>
              <h2>Session Persona</h2>
            </div>
            <span className="status-pill subtle">{connection === "connected" ? "live" : "draft"}</span>
          </div>

          <div className="persona-main-stack">
            <section className="persona-preview-card">
              <div className="card-head">
                <div>
                  <p className="eyebrow">Preview</p>
                  <h3>What the judge sounds like</h3>
                </div>
              </div>
              <div className="persona-summary-copy">
                {personaPreview.summary.map((line) => (
                  <p key={line}>{line}</p>
                ))}
              </div>
              <ul className="persona-bullet-list">
                {personaPreview.highlights.map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
            </section>

            <section className="persona-editor-card">
              <div className="card-head">
                <div>
                  <p className="eyebrow">Source editor</p>
                  <h3>Edit source text</h3>
                </div>
                <button className="button-secondary" onClick={() => setShowPersonaEditor((current) => !current)}>
                  {showPersonaEditor ? "Hide editor" : "Show editor"}
                </button>
              </div>
              {showPersonaEditor ? (
                <div className="persona-editor-shell">
                  <textarea
                    className="persona-editor"
                    value={personaDraft}
                    onChange={(event) => setPersonaDraft(event.target.value)}
                  />
                </div>
              ) : (
                <p className="persona-supporting-copy">
                  The raw source stays hidden by default so this page stays readable. Open the editor when you want to
                  paste, upload, or refine the prompt text directly.
                </p>
              )}
            </section>

            <div className="persona-note-grid">
              <section className="persona-note-card">
                <span className="persona-card-label">Runtime note</span>
                <p>
                  This page customizes the persona used inside the full end-to-end voice loop. It is not a standalone
                  LLM playground.
                </p>
              </section>
              <section className="persona-note-card">
                <span className="persona-card-label">Protocol</span>
                <p>
                  The backend still appends the runtime floor-control rules. You are editing only the persona layer the
                  user sees.
                </p>
              </section>
            </div>
          </div>
        </section>
      </div>
    );
  }

  function renderRunsScreen() {
    const detail = selectedRunDetail;
    return (
      <div className="screen-grid runs-grid">
        <aside className="panel runs-list-panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Runs</p>
              <h2>Saved sessions</h2>
            </div>
            <button className="button-secondary" onClick={() => void fetchRuns()} disabled={runsLoading}>
              {runsLoading ? "Refreshing…" : "Refresh"}
            </button>
          </div>

          <div className="runs-list">
            {runs.length === 0 ? (
              <div className="empty-state">No saved runs yet. Complete a live or file-driven session first.</div>
            ) : (
              runs.map((run) => (
                <button
                  key={run.id}
                  className={selectedRunId === run.id ? "run-card active" : "run-card"}
                  onClick={() => setSelectedRunId(run.id)}
                >
                  <div className="card-label-row">
                    <span>{formatTimestamp(run.updated_at_ms)}</span>
                    <strong>{run.latest_action || "no action"}</strong>
                  </div>
                  <h3>{run.id}</h3>
                  <p>{snippet(run.transcript, "No transcript saved.")}</p>
                </button>
              ))
            )}
          </div>
        </aside>

        <section className="panel run-detail-panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Run detail</p>
              <h2>{detail?.id ?? selectedRunId ?? "Select a run"}</h2>
            </div>
          </div>

          {runDetailLoading ? (
            <div className="empty-state">Loading saved session detail…</div>
          ) : detail ? (
            <div className="run-detail-grid">
              <div className="summary-card">
                <span>When</span>
                <strong>{formatTimestamp(detail.updated_at_ms)}</strong>
                <p>Latest action: {detail.latest_action || "unknown"}</p>
              </div>

              <div className="summary-card">
                <span>Coverage</span>
                <strong>{detail.turns.length} committed turn{detail.turns.length === 1 ? "" : "s"}</strong>
                <p>{detail.timeline.length} saved pipeline events for this session.</p>
              </div>

              <div className="media-card">
                <span>Input audio</span>
                {detail.input_audio_url ? (
                  <audio controls src={`${API_BASE}${detail.input_audio_url}`} />
                ) : (
                  <p>No input audio saved.</p>
                )}
              </div>

              <div className="media-card">
                <span>TTS audio</span>
                {detail.tts_audio_url ? (
                  <audio controls src={`${API_BASE}${detail.tts_audio_url}`} />
                ) : (
                  <p>No TTS output saved.</p>
                )}
              </div>

              <div className="text-card">
                <span>Committed transcript</span>
                <pre>{detail.transcript || "No transcript saved."}</pre>
              </div>

              <div className="text-card">
                <span>LLM output</span>
                <pre>{detail.llm_output || "No LLM output saved."}</pre>
              </div>

              <div className="text-card">
                <span>Committed turns</span>
                {detail.turns.length === 0 ? (
                  <pre>No committed turns saved.</pre>
                ) : (
                  <ul className="detail-list">
                    {detail.turns.map((turn) => (
                      <li key={turn.turn_id}>
                        <div className="detail-list-head">
                          <strong>Turn {turn.turn_id.slice(0, 8)}</strong>
                          <span>{formatTurnDuration(turn.started_at_ms, turn.committed_at_ms)}</span>
                        </div>
                        <p>{turn.text}</p>
                        {turn.audio_url ? <audio controls src={`${API_BASE}${turn.audio_url}`} /> : null}
                      </li>
                    ))}
                  </ul>
                )}
              </div>

              <div className="text-card">
                <span>Pipeline timeline</span>
                {detail.timeline.length === 0 ? (
                  <pre>No pipeline events saved.</pre>
                ) : (
                  <ul className="detail-list">
                    {detail.timeline.map((event, index) => (
                      <li key={`${event.recorded_at_ms}-${event.type}-${index}`}>
                        <div className="detail-list-head">
                          <strong>{event.type}</strong>
                          <span>{formatTimestamp(event.recorded_at_ms)}</span>
                        </div>
                        <p>{event.detail}</p>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          ) : (
            <div className="empty-state">Select a run to inspect the saved conversation artifacts.</div>
          )}
        </section>
      </div>
    );
  }

  function renderAboutDialog() {
    if (!showAbout) {
      return null;
    }
    return (
      <div className="about-backdrop" onClick={() => setShowAbout(false)}>
        <section
          className="about-dialog"
          role="dialog"
          aria-modal="true"
          aria-labelledby="volta-about-title"
          onClick={(event) => event.stopPropagation()}
        >
          <div className="about-head">
            <div className="about-brand">
              <div className="brand-mark-shell large" aria-hidden="true">
                <BrandMark className="brand-mark-svg" />
              </div>
              <div>
                <p className="eyebrow">Volta</p>
                <h2 id="volta-about-title">About Volta</h2>
              </div>
            </div>
            <button
              className="toolbar-icon-button"
              onClick={() => setShowAbout(false)}
              aria-label="Close about panel"
            >
              <CloseIcon className="toolbar-icon" />
            </button>
          </div>

          <p className="about-lede">
            Volta is a local realtime voice app for low-latency STT → LLM → TTS conversations, with persona control,
            barge-in behavior, saved runs, and a browser harness for building and evaluating voice agents on Apple
            Silicon.
          </p>

          <div className="about-grid">
            <section className="about-card">
              <span>Pipeline</span>
              <strong>STT → LLM → TTS</strong>
              <p>One continuous end-to-end loop, not a set of isolated model demos.</p>
            </section>
            <section className="about-card">
              <span>Voice control</span>
              <strong>Persona + floor-taking</strong>
              <p>The LLM decides when to wait, yield, or speak, and every run is saved for inspection.</p>
            </section>
            <section className="about-card">
              <span>Designed for</span>
              <strong>Local agentic voice systems</strong>
              <p>Meeting facilitation, sales/support coaching, accessibility overlays, and realtime NPC-style voice.</p>
            </section>
          </div>

          <div className="about-footer">
            <a className="button-secondary link-button" href={VOLTA_REPO_URL} target="_blank" rel="noreferrer">
              Open GitHub
            </a>
          </div>
        </section>
      </div>
    );
  }

  return (
    <div className={`app-shell screen-${screen}`}>
      <div className="atmosphere atmosphere-a" />
      <div className="atmosphere atmosphere-b" />

      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark-shell" aria-hidden="true">
            <BrandMark className="brand-mark-svg" />
          </div>
          <div className="brand-copy">
            <p className="eyebrow">Volta</p>
            <h1>Local realtime voice orchestration</h1>
            <p className="lede">Premium local STT → LLM → TTS control with persona, barge-in, and saved runs.</p>
          </div>
        </div>

        <div className="topbar-side">
          <nav className="screen-tabs" aria-label="Main navigation">
            {(
              [
                ["console", "Console"],
                ["persona", "Persona"],
                ["runs", "Runs"],
              ] as const
            ).map(([value, label]) => (
              <button
                key={value}
                className={screen === value ? "screen-tab active" : "screen-tab"}
                onClick={() => setScreen(value)}
              >
                {label}
              </button>
            ))}
          </nav>

          <div className="topbar-utility-cluster">
            <div className="hero-summary">
              <span>{mode === "mic" ? "Live mic" : "Audio file"}</span>
              <span>{statusTone}</span>
              <span>{llmAction || "Waiting"}</span>
            </div>
            <div className="toolbar-actions">
              <button
                className="toolbar-icon-button"
                onClick={() => setTheme(nextTheme)}
                aria-label={`Switch to ${nextTheme} theme`}
                title={`Switch to ${nextTheme} theme`}
              >
                {theme === "dark" ? <SunIcon className="toolbar-icon" /> : <MoonIcon className="toolbar-icon" />}
              </button>
              <button className="toolbar-pill-button" onClick={() => setShowAbout(true)}>
                <InfoIcon className="toolbar-icon" />
                <span>About</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="viewport">
        {screen === "console" ? renderConsoleScreen() : null}
        {screen === "persona" ? renderPersonaScreen() : null}
        {screen === "runs" ? renderRunsScreen() : null}
      </main>
      {renderAboutDialog()}
    </div>
  );
}
