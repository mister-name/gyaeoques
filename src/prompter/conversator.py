#!/usr/bin/env python3
"""
conversator.py - Event-driven conversational pipeline using Sherpa-ONNX.

Pipeline:
    Microphone -> [optional GTCRN noise reduction] -> Silero VAD
              -> Speaker Identification (UUID-based, auto-registers unknowns)
              -> Offline ASR (SenseVoice / Paraformer / Whisper / Transducer)
              -> Session text accumulation per speaker
              -> TTS echo response ("You said …")

Events emitted on the EventBus:
  speaking_started     VAD opened a speech segment
  speaker_detected     speaker UUID resolved (known / unknown)
  speaking_ended       VAD closed the segment; audio chunk attached
  transcript_ready     STT result for one speech chunk
  speaking_finished    full accumulated text for the utterance

Usage (minimal – SenseVoice ASR + VITS-piper TTS):
    python3 conversator.py \\
      --silero-vad-model  silero_vad.onnx \\
      --speaker-model     wespeaker_en_lifespeech_eres2net_base.onnx \\
      --sense-voice       model.onnx \\
      --tokens            tokens.txt \\
      --vits-model        vits-piper-en_US-amy-low/en_US-amy-low.onnx \\
      --vits-tokens       vits-piper-en_US-amy-low/tokens.txt \\
      --vits-data-dir     vits-piper-en_US-amy-low/espeak-ng-data

Model downloads handled in prepare-conversator.sh; it includes required model
downloads:
  # Silero VAD
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  # Speaker embeddings (English)
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_lifespeech_eres2net_base.onnx

  # ASR - SenseVoice (multilingual, fast, accurate)
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  tar xf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

  # TTS - VITS-piper Amy (English, fast, ~50 MB)
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
  tar xf vits-piper-en_US-amy-low.tar.bz2

  # (Optional) GTCRN speech denoiser
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx

Optional models: pre-register known speakers with a text file:
  speaker.txt:
      Alice /path/to/alice.wav
      Bob   /path/to/bob.wav
  then pass --speaker-file speaker.txt
"""

import argparse
import collections
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import pathlib
import queue
import threading
import time
import uuid
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import sherpa_onnx

try:
    import sounddevice as sd
except ImportError:
    raise SystemExit(
        "sounddevice is not installed. Run:  pip install sounddevice"
    )

# Constants
SAMPLE_RATE = 16_000    # sherpa-onnx ASR / VAD / embedding models use 16 kHz
READ_CHUNK_MS = 100     # audio capture granularity in milliseconds
READ_CHUNK_SAMPLES = int(SAMPLE_RATE * READ_CHUNK_MS / 1000)
MIN_SPEECH_SAMPLES = int(0.5 * SAMPLE_RATE)  # discard segments shorter than 0.5 s
UNKNOWN_SPEAKER_LABEL = "unknown"

log = logging.getLogger("pipeline")


# Event definitions
class EventKind(str, Enum):
    SPEAKING_STARTED   = "speaking_started"
    SPEAKER_DETECTED   = "speaker_detected"
    SPEAKING_ENDED     = "speaking_ended"
    TRANSCRIPT_READY   = "transcript_ready"
    SPEAKING_FINISHED  = "speaking_finished"


@dataclass
class BaseEvent:
    """All events carry a wall-clock timestamp and absolute sample position."""
    kind: EventKind
    timestamp: float = field(default_factory=time.monotonic)
    sample_pos: int = 0


@dataclass
class SpeakingStartedEvent(BaseEvent):
    """Fired when VAD opens a new speech segment."""
    kind: EventKind = EventKind.SPEAKING_STARTED


@dataclass
class SpeakerDetectedEvent(BaseEvent):
    """Fired after speaker embedding is computed for a segment."""
    kind: EventKind = EventKind.SPEAKER_DETECTED
    speaker_id: str = ""       # UUID string
    is_known: bool = False     # True if matched a pre-registered speaker
    display_name: str = ""     # Human-readable name / "Speaker-<uuid8>"


@dataclass
class SpeakingEndedEvent(BaseEvent):
    """Fired when VAD closes a speech segment."""
    kind: EventKind = EventKind.SPEAKING_ENDED
    speaker_id: str = ""
    duration_s: float = 0.0
    start_sample: int = 0      # abs sample index of segment start
    end_sample: int = 0        # abs sample index of segment end
    samples: Optional[np.ndarray] = None   # float32, 16 kHz, mono


@dataclass
class TranscriptEvent(BaseEvent):
    """Fired when ASR finishes one speech segment."""
    kind: EventKind = EventKind.TRANSCRIPT_READY
    speaker_id: str = ""
    display_name: str = ""
    text: str = ""
    chunk_index: int = 0       # sequential chunk counter per speaker


@dataclass
class SpeakingFinishedEvent(BaseEvent):
    """Fired after TTS echo is queued; carries the full session text so far."""
    kind: EventKind = EventKind.SPEAKING_FINISHED
    speaker_id: str = ""
    display_name: str = ""
    full_text: str = ""        # all transcribed text accumulated this session


# EventBus
Handler = Callable[[BaseEvent], None]

class EventBus:
    """
    Lightweight synchronous event bus.

    Subscribers register per-event-kind callbacks.  publish() calls all
    handlers immediately in the publishing thread and also places the event
    on an asyncio-compatible Queue for any async consumers that want it.
    """

    def __init__(self):
        self._handlers: Dict[EventKind, List[Handler]] = collections.defaultdict(list)

    def subscribe(self, kind: EventKind, handler: Handler) -> None:
        """Register *handler* to be called whenever *kind* is published."""
        self._handlers[kind].append(handler)

    def publish(self, event: BaseEvent) -> None:
        """Dispatch *event* to all registered handlers synchronously."""
        for handler in self._handlers.get(event.kind, []):
            try:
                handler(event)
            except Exception:
                log.exception("EventBus handler raised an exception for %s", event.kind)


# AudioCapture
class AudioCapture:
    """
    Captures 16 kHz mono float32 audio from the system microphone via
    sounddevice callbacks and queues READ_CHUNK_SAMPLES-sized blocks.

    Each queue item is a tuple (samples: np.ndarray, start_sample: int).
    """

    def __init__(self, device: Optional[int] = None):
        self._device = device
        self._q: queue.Queue = queue.Queue(maxsize=200)
        self._total_samples = 0
        self._stream: Optional[sd.InputStream] = None

    def start(self) -> None:
        devices = sd.query_devices()
        if not devices:
            raise RuntimeError("No audio input devices found")
        idx = self._device if self._device is not None else sd.default.device[0]
        log.info("AudioCapture: using device %s – %s", idx, devices[idx]["name"])
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=READ_CHUNK_SAMPLES,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def read(self) -> Tuple[np.ndarray, int]:
        """Blocking read -> (float32 samples, absolute start sample index)."""
        return self._q.get()

    @property
    def total_samples(self) -> int:
        return self._total_samples

    # callback runs in sounddevice background thread
    def _callback(self, indata: np.ndarray, frames: int, t, status):
        if status:
            log.warning("AudioCapture status: %s", status)
        samples = indata[:, 0].copy()
        start = self._total_samples
        self._total_samples += len(samples)
        try:
            self._q.put_nowait((samples, start))
        except queue.Full:
            log.warning("AudioCapture queue full – dropping %d samples", len(samples))


class SpeechEnhancer:
    """
    Wraps the Sherpa-ONNX OnlineSpeechDenoiser (GTCRN model).

    Feed audio sample-by-sample in any chunk size; the enhancer buffers
    internally and outputs denoised audio aligned to the input.

    If the model produces no output for a given chunk (latency), the
    original samples are returned unchanged so the pipeline is not stalled.
    """

    def __init__(self, model_path: str, num_threads: int = 1):
        config = sherpa_onnx.OnlineSpeechDenoiserConfig(
            model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(
                    model=model_path
                ),
                debug=False,
                num_threads=num_threads,
                provider="cpu",
            )
        )
        if not config.validate():
            raise ValueError(f"Invalid SpeechEnhancer config: {config}")
        self._denoiser = sherpa_onnx.OnlineSpeechDenoiser(config)
        self._frame_shift = self._denoiser.frame_shift_in_samples
        self._raw_buf = np.empty(0, dtype=np.float32)
        self._enhanced_buf = np.empty(0, dtype=np.float32)
        log.info("SpeechEnhancer loaded: %s  frame_shift=%d", model_path, self._frame_shift)

    def enhance(self, samples: np.ndarray) -> np.ndarray:
        """
        Push *samples* through the denoiser.  Returns as many denoised samples
        as are available (may be fewer than input due to model latency).
        Accumulated denoised audio is returned on subsequent calls.
        """
        self._raw_buf = np.concatenate([self._raw_buf, samples])
        while len(self._raw_buf) >= self._frame_shift:
            chunk = self._raw_buf[: self._frame_shift]
            self._raw_buf = self._raw_buf[self._frame_shift :]
            result = self._denoiser(chunk, SAMPLE_RATE)
            self._enhanced_buf = np.concatenate(
                [self._enhanced_buf, np.asarray(result.samples, dtype=np.float32)]
            )

        # Return however many enhanced samples are ready
        if len(self._enhanced_buf) >= len(samples):
            out = self._enhanced_buf[: len(samples)]
            self._enhanced_buf = self._enhanced_buf[len(samples) :]
            return out
        # Not enough enhanced samples yet - pad with input samples
        # (graceful degradation; happens only in the first few frames)
        return samples

    def flush(self) -> np.ndarray:
        """Flush any remaining buffered audio from the model."""
        result = self._denoiser.flush()
        tail = np.asarray(result.samples, dtype=np.float32)
        return np.concatenate([self._enhanced_buf, tail])


class SpeakerRegistry:
    """
    Manages speaker identities as UUID strings.

    *Known* speakers are pre-registered from audio files.
    *Unknown* speakers are automatically registered on first encounter and
    given a UUID + a short display label ("Speaker-<uuid8>").

    All lookups use cosine similarity via sherpa_onnx.SpeakerEmbeddingManager.
    """

    def __init__(
        self,
        extractor: sherpa_onnx.SpeakerEmbeddingExtractor,
        threshold: float = 0.5,
    ):
        self._extractor = extractor
        self._threshold = threshold
        self._manager = sherpa_onnx.SpeakerEmbeddingManager(extractor.dim)
        self._id_to_name: Dict[str, str] = {}   # uuid -> display name

    def save(self, path: str) -> None:
        """
        Persist all speaker embeddings and names to *path* (a directory).
        Creates:
            <path>/index.json          - {uuid: display_name}
            <path>/<uuid>.npy          - embedding vector per speaker
        """
        p = pathlib.Path(path)
        p.mkdir(parents=True, exist_ok=True)

        # Write index
        (p / "index.json").write_text(
            json.dumps(self._id_to_name, indent=2), encoding="utf-8"
        )
        # Write one .npy per speaker
        # We need to re-extract the vectors - they're stored opaquely
        # in _manager, so we must keep a shadow dict of the raw arrays.
        for spk_uuid, embedding in self._embeddings.items():
            np.save(str(p / f"{spk_uuid}.npy"), embedding)

        log.info("Saved %d speaker(s) to %s", len(self._id_to_name), path)

    def load(self, path: str) -> None:
        """
        Load previously saved speaker embeddings from *path*.
        Restores UUIDs so the same person gets the same ID across sessions.
        """
        p = pathlib.Path(path)
        if not p.is_dir():
            log.info("Speaker store %s does not exist - starting fresh.", path)
            return

        index_file = p / "index.json"
        if not index_file.exists():
            return

        index: Dict[str, str] = json.loads(
            index_file.read_text(encoding="utf-8"))
        loaded = 0
        for spk_uuid, display_name in index.items():
            npy_file = p / f"{spk_uuid}.npy"
            if not npy_file.exists():
                log.warning("Missing embedding file for %s (%s) - skipping",
                            display_name, spk_uuid)
                continue
            embedding = np.load(str(npy_file))
            if self._manager.add(spk_uuid, embedding):
                self._id_to_name[spk_uuid] = display_name
                self._embeddings[spk_uuid] = embedding
                loaded += 1
            else:
                log.warning("Failed to re-register %s (%s)", display_name,
                            spk_uuid)

        log.info("Loaded %d speaker(s) from %s", loaded, path)

    def register_from_embedding(self, display_name: str,
                                embedding: np.ndarray) -> str:
        spk_uuid = str(uuid.uuid4())
        if not self._manager.add(spk_uuid, embedding):
            raise RuntimeError(
                f"SpeakerEmbeddingManager.add failed for '{display_name}'")
        self._id_to_name[spk_uuid] = display_name
        self._embeddings[spk_uuid] = embedding  # ← add this line
        log.info("Registered speaker '%s' -> %s", display_name, spk_uuid)
        return spk_uuid

    def register_from_samples(

        self,
        display_name: str,
        samples_list: List[Tuple[np.ndarray, int]],
    ) -> str:
        """
        Register a speaker from one or more (samples, sample_rate) tuples.
        Multiple recordings are averaged for a more robust embedding.
        """
        embeddings = [self._compute_embedding(s, sr) for s, sr in samples_list]
        avg = np.mean(embeddings, axis=0).astype(np.float32)
        return self.register_from_embedding(display_name, avg)

    # ---- identification -----------------------------------------------------

    def identify(
        self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE
    ) -> Tuple[str, bool, str]:
        """
        Identify the speaker in *samples*.

        Returns
        -------
        (uuid_str, is_known, display_name)
            is_known is True only when the segment matched a pre-registered speaker.
            Unknown speakers are auto-registered and is_known is False.
        """
        embedding = self._compute_embedding(samples, sample_rate)
        found_id = self._manager.search(embedding, threshold=self._threshold)
        if found_id:
            return found_id, True, self._id_to_name.get(found_id, found_id)

        # Auto-register as a new unknown speaker
        new_uuid = str(uuid.uuid4())
        if self._manager.add(new_uuid, embedding):
            label = f"Speaker-{new_uuid[:8]}"
            self._id_to_name[new_uuid] = label
            self._embeddings[new_uuid] = embedding
            log.info("Auto-registered new speaker -> %s (%s)", new_uuid, label)
            return new_uuid, False, label

        log.warning("Could not auto-register new speaker; using generic label")
        return UNKNOWN_SPEAKER_LABEL, False, UNKNOWN_SPEAKER_LABEL

    def get_display_name(self, speaker_id: str) -> str:
        return self._id_to_name.get(speaker_id, speaker_id)

    def all_speakers(self) -> Dict[str, str]:
        """Returns a copy of the {uuid: display_name} mapping."""
        return dict(self._id_to_name)

    def _compute_embedding(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
        stream.input_finished()
        assert self._extractor.is_ready(stream), "Embedding extractor not ready"
        return np.array(self._extractor.compute(stream), dtype=np.float32)


class ASREngine:
    """Thin wrapper around sherpa_onnx.OfflineRecognizer."""

    def __init__(self, recognizer: sherpa_onnx.OfflineRecognizer):
        self._rec = recognizer

    def transcribe(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        """Transcribe *samples* and return the text string."""
        stream = self._rec.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
        self._rec.decode_stream(stream)
        return stream.result.text.strip()


class TTSEngine:
    """
    Wraps sherpa_onnx.OfflineTts with non-blocking playback.

    speak() is non-blocking: the text is queued and synthesised + played
    in a dedicated background thread so the capture/VAD loop never stalls.
    Multiple calls pile up; they are drained in order.
    """

    def __init__(
        self,
        tts: sherpa_onnx.OfflineTts,
        sid: int = 0,
        speed: float = 1.0,
    ):
        self._tts = tts
        self._sid = sid
        self._speed = speed
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="tts-worker")
        self._thread.start()

    def speak(self, text: str) -> None:
        """Queue *text* for synthesis + playback (returns immediately)."""
        self._q.put(text)

    def stop(self) -> None:
        """Signal the worker thread to exit cleanly."""
        self._stop.set()
        self._q.put(None)  # unblock get()
        self._thread.join(timeout=5)

    def _worker(self) -> None:
        """Background thread: synthesise + play queued text."""
        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=0.3)
            except queue.Empty:
                continue
            if text is None:
                break
            log.info("TTS ▶ %r", text)
            cfg = sherpa_onnx.GenerationConfig()
            cfg.sid = self._sid
            cfg.speed = self._speed
            audio = self._tts.generate(text, cfg)
            if audio.samples is not None and len(audio.samples) > 0:
                self._play(audio.samples, audio.sample_rate)

    def _play(self, samples: np.ndarray, sample_rate: int) -> None:
        """Synchronously play *samples* via sounddevice (blocks _worker)."""
        done = threading.Event()
        pos = [0]

        def callback(outdata: np.ndarray, frames: int, t, status):
            end = pos[0] + frames
            chunk = samples[pos[0] : end]
            n = len(chunk)
            outdata[:n, 0] = chunk
            if n < frames:
                outdata[n:, 0] = 0.0
                done.set()
            pos[0] = end

        with sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
            blocksize=1024,
        ):
            done.wait()


class SessionAccumulator:
    """
    Accumulates transcription chunks by speaker UUID for the current session.

    A "session" is the lifetime of this script's pipeline run.

    Use reset_speaker() to clear one speaker's history, or reset_all() to
    start fresh.
    """

    def __init__(self):
        self._texts: Dict[str, List[str]] = collections.defaultdict(list)
        self._chunk_counts: Dict[str, int] = collections.defaultdict(int)

    def add(self, speaker_id: str, text: str) -> int:
        """Append *text* for *speaker_id*. Returns the chunk index."""
        idx = self._chunk_counts[speaker_id]
        self._texts[speaker_id].append(text)
        self._chunk_counts[speaker_id] += 1
        return idx

    def get_full_text(self, speaker_id: str) -> str:
        """Return all accumulated text for *speaker_id* as a single string."""
        return " ".join(self._texts[speaker_id])

    def get_all(self) -> Dict[str, str]:
        """Return {speaker_id: full_text} for all speakers."""
        return {sid: " ".join(chunks) for sid, chunks in self._texts.items()}

    def reset_speaker(self, speaker_id: str) -> None:
        self._texts[speaker_id].clear()
        self._chunk_counts[speaker_id] = 0

    def reset_all(self) -> None:
        self._texts.clear()
        self._chunk_counts.clear()


class ConversatorPipeline:
    """
    Main orchestrator: ties all components together in an event-driven pipeline

    Thread model:
    - sounddevice callback thread  – pushes raw PCM into AudioCapture queue
    - main/run() thread            – reads PCM, runs enhancer + VAD,
                                     identifies speaker, runs ASR,
                                     publishes events, queues TTS
    - TTS worker thread            – synthesises + plays audio asynchronously

    All sherpa-onnx inference happens in the run() thread to avoid
    thread-safety concerns with ONNX Runtime sessions.
    """

    def __init__(
        self,
        bus: EventBus,
        capture: AudioCapture,
        vad: sherpa_onnx.VoiceActivityDetector,
        vad_window_size: int,
        registry: SpeakerRegistry,
        asr: ASREngine,
        tts: "TTSEngine | _NoOpTTS",
        enhancer: Optional[SpeechEnhancer] = None,
    ):
        self._bus = bus
        self._capture = capture
        self._vad = vad
        self._vad_window_size = vad_window_size
        self._registry = registry
        self._asr = asr
        self._tts = tts
        self._enhancer = enhancer
        self._session = SessionAccumulator()

        # Running counters
        self._total_samples: int = 0   # samples consumed from mic (absolute)
        self._vad_buf = np.empty(0, dtype=np.float32)

    @property
    def session(self) -> SessionAccumulator:
        """Access the session accumulator for external text retrieval."""
        return self._session

    def run(self) -> None:
        """
        Start the pipeline and block until KeyboardInterrupt.
        Call from the main thread.
        """
        self._capture.start()
        log.info("Pipeline running - speak now!  (Ctrl+C to stop)")
        try:
            while True:
                samples, start_pos = self._capture.read()
                self._total_samples = start_pos + len(samples)
                self._ingest(samples, start_pos)
        except KeyboardInterrupt:
            pass
        finally:
            self._capture.stop()
            self._tts.stop()
            log.info("Pipeline stopped.")

    def _ingest(self, raw: np.ndarray, start_pos: int) -> None:
        """
        internal pipeline stages:
        1. denoising (optional)
        2. Feed VAD window by window
        3. Drain completed speech segments
        """
        samples = self._enhancer.enhance(raw) if self._enhancer else raw

        # Buffer and feed VAD in window_size increments
        self._vad_buf = np.concatenate([self._vad_buf, samples])
        while len(self._vad_buf) >= self._vad_window_size:
            self._vad.accept_waveform(self._vad_buf[: self._vad_window_size])
            self._vad_buf = self._vad_buf[self._vad_window_size :]

        # Process any completed speech segments
        while not self._vad.empty():
            seg_samples = np.asarray(self._vad.front.samples, dtype=np.float32)
            self._vad.pop()
            if len(seg_samples) < MIN_SPEECH_SAMPLES:
                log.debug("Skipping short segment (%d samples)", len(seg_samples))
                continue
            self._process_segment(seg_samples)

    def _process_segment(self, samples: np.ndarray) -> None:
        """
        Process one complete VAD speech segment through the full pipeline:
        SpeakingStarted -> SpeakerDetected -> SpeakingEnded ->
        TranscriptReady -> SpeakingFinished -> TTS echo
        """
        duration_s = len(samples) / SAMPLE_RATE
        # Approximate segment start position (segment ends at total_samples)
        start_sample = self._total_samples - len(samples)
        end_sample = self._total_samples
        ts_start = time.monotonic()

        # 1. SpeakingStarted
        self._bus.publish(SpeakingStartedEvent(
            timestamp=ts_start,
            sample_pos=start_sample,
        ))

        # 2. Speaker identification
        speaker_id, is_known, display_name = self._registry.identify(samples)
        self._bus.publish(SpeakerDetectedEvent(
            timestamp=time.monotonic(),
            sample_pos=start_sample,
            speaker_id=speaker_id,
            is_known=is_known,
            display_name=display_name,
        ))

        # 3. SpeakingEnded
        self._bus.publish(SpeakingEndedEvent(
            timestamp=time.monotonic(),
            sample_pos=end_sample,
            speaker_id=speaker_id,
            duration_s=duration_s,
            start_sample=start_sample,
            end_sample=end_sample,
            samples=samples,
        ))

        # 4. ASR transcription
        text = self._asr.transcribe(samples)
        if not text:
            log.debug("ASR returned empty text for segment (%.2fs)", duration_s)
            return

        chunk_idx = self._session.add(speaker_id, text)
        self._bus.publish(TranscriptEvent(
            timestamp=time.monotonic(),
            sample_pos=end_sample,
            speaker_id=speaker_id,
            display_name=display_name,
            text=text,
            chunk_index=chunk_idx,
        ))

        # 5. SpeakingFinished
        full_text = self._session.get_full_text(speaker_id)
        self._bus.publish(SpeakingFinishedEvent(
            timestamp=time.monotonic(),
            sample_pos=end_sample,
            speaker_id=speaker_id,
            display_name=display_name,
            full_text=full_text,
        ))

        # 6. TTS echo
        self._tts.speak(f"You said: {text}")


class _NoOpTTS:
    """No-op TTS stub - for when no TTS model is supplied"""
    def speak(self, text: str) -> None:
        log.info("[TTS disabled] Would say: %r", text)

    def stop(self) -> None:
        pass


def _build_vad(args) -> Tuple[sherpa_onnx.VoiceActivityDetector, int]:
    """Factory helpers: argparse args -> sherpa_onnx objects"""
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = args.silero_vad_model
    cfg.silero_vad.min_silence_duration = args.min_silence_duration
    cfg.silero_vad.min_speech_duration = args.min_speech_duration
    cfg.silero_vad.threshold = args.vad_threshold
    cfg.sample_rate = SAMPLE_RATE
    if not cfg.validate():
        raise ValueError(f"Invalid VAD config: {cfg}")
    window_size = cfg.silero_vad.window_size
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=100)
    log.info("VAD loaded: silero  window_size=%d", window_size)
    return vad, window_size


def _build_recognizer(args) -> sherpa_onnx.OfflineRecognizer:
    common = dict(
        tokens=args.tokens,
        num_threads=args.num_threads,
        debug=args.debug,
        decoding_method=args.decoding_method,
    )
    if args.sense_voice:
        log.info("ASR: SenseVoice  %s", args.sense_voice)
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=args.sense_voice, use_itn=True, **common
        )
    if args.paraformer:
        log.info("ASR: Paraformer  %s", args.paraformer)
        return sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=args.paraformer,
            sample_rate=SAMPLE_RATE,
            feature_dim=args.feature_dim,
            **common,
        )
    if args.whisper_encoder:
        log.info("ASR: Whisper  %s", args.whisper_encoder)
        return sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=args.whisper_encoder,
            decoder=args.whisper_decoder,
            language=args.whisper_language,
            task=args.whisper_task,
            tail_paddings=args.whisper_tail_paddings,
            **common,
        )
    if args.encoder:
        log.info("ASR: Transducer  %s", args.encoder)
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=args.encoder,
            decoder=args.decoder,
            joiner=args.joiner,
            sample_rate=SAMPLE_RATE,
            feature_dim=args.feature_dim,
            **common,
        )
    raise ValueError(
        "No ASR model specified. Use --sense-voice, --paraformer, "
        "--whisper-encoder, or --encoder/--decoder/--joiner."
    )


def _build_tts(args) -> Optional[sherpa_onnx.OfflineTts]:
    """Returns None if no TTS model paths are given."""
    if not any([args.vits_model, args.matcha_acoustic_model, args.kokoro_model]):
        return None

    tts_cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=args.vits_model or "",
                lexicon=args.vits_lexicon or "",
                data_dir=args.vits_data_dir or "",
                tokens=args.vits_tokens or "",
            ),
            matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=args.matcha_acoustic_model or "",
                vocoder=args.matcha_vocoder or "",
                lexicon=args.matcha_lexicon or "",
                tokens=args.matcha_tokens or "",
                data_dir=args.matcha_data_dir or "",
            ),
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=args.kokoro_model or "",
                voices=args.kokoro_voices or "",
                tokens=args.kokoro_tokens or "",
                data_dir=args.kokoro_data_dir or "",
                lexicon=args.kokoro_lexicon or "",
            ),
            provider=args.provider,
            debug=args.debug,
            num_threads=args.num_threads,
        ),
        rule_fsts=args.tts_rule_fsts or "",
        max_num_sentences=1,
    )
    if not tts_cfg.validate():
        raise ValueError("Invalid TTS config - check model paths.")
    log.info("TTS loaded.")
    return sherpa_onnx.OfflineTts(tts_cfg)


def _preregister_speakers(
    speaker_file: str,
    extractor: sherpa_onnx.SpeakerEmbeddingExtractor,
    registry: SpeakerRegistry,
) -> None:
    """
    Parse a text file of the form:
        name  /path/to/file.wav
    and register each speaker (averaging embeddings for multiple files
    with the same name).
    """
    import soundfile as sf

    speaker_wavs: Dict[str, List[str]] = collections.defaultdict(list)
    with open(speaker_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                log.warning("Skipping malformed speaker-file line: %r", line)
                continue
            speaker_wavs[parts[0]].append(parts[1])

    for name, wav_paths in speaker_wavs.items():
        samples_list = []
        for wav in wav_paths:
            data, sr = sf.read(wav, always_2d=True, dtype="float32")
            samples_list.append((np.ascontiguousarray(data[:, 0]), sr))
        registry.register_from_samples(name, samples_list)


def _attach_console_handlers(bus: EventBus, registry: SpeakerRegistry) -> None:
    """
    Console event handlers, basic logging handlers and
    hook point for LLM/RAG/TTS integration.
    """

    def on_speaking_started(e: SpeakingStartedEvent):
        log.debug(
            "[speaking_started]  sample=%d  t=%.3f",
            e.sample_pos, e.timestamp,
        )

    def on_speaker_detected(e: SpeakerDetectedEvent):
        status = "known  " if e.is_known else "NEW    "
        log.info(
            "[speaker_detected]  %s  name=%-20s  id=%s  sample=%d",
            status, e.display_name, e.speaker_id, e.sample_pos,
        )

    def on_speaking_ended(e: SpeakingEndedEvent):
        log.debug(
            "[speaking_ended]    name=%-20s  dur=%.2fs  samples %d–%d",
            registry.get_display_name(e.speaker_id),
            e.duration_s, e.start_sample, e.end_sample,
        )

    def on_transcript(e: TranscriptEvent):
        # This is the primary output line
        print(f"\n  [{e.display_name}] #{e.chunk_index}: {e.text}\n")

    def on_speaking_finished(e: SpeakingFinishedEvent):
        """
        :param e:
        :return:

        e.full_text is the complete transcription so far for this speaker.
        Identified text chunk: (e.speaker_id, e.full_text)

        For speaking LLM responses:
          tts_engine.speak(llm_response)
        """
        log.info(
            "[speaking_finished] name=%-20s  full_text=%r",
            e.display_name, e.full_text,
        )

    bus.subscribe(EventKind.SPEAKING_STARTED,  on_speaking_started)
    bus.subscribe(EventKind.SPEAKER_DETECTED,  on_speaker_detected)
    bus.subscribe(EventKind.SPEAKING_ENDED,    on_speaking_ended)
    bus.subscribe(EventKind.TRANSCRIPT_READY,  on_transcript)
    bus.subscribe(EventKind.SPEAKING_FINISHED, on_speaking_finished)


# CLI arguments getter
def _get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )

    # Required args
    p.add_argument(
        "--silero-vad-model", required=True, metavar="PATH",
        help="Path to silero_vad.onnx",
    )
    p.add_argument(
        "--speaker-model", required=True, metavar="PATH",
        help="Path to speaker-embedding .onnx  (WeSpeaker or 3DSpeaker)",
    )

    # ASR: tokens
    p.add_argument(
        "--tokens", default="", metavar="PATH",
        help="Path to tokens.txt  (required for most ASR backends)",
    )

    # ASR: model
    asr = p.add_argument_group("ASR backend  (specify exactly one)")
    asr.add_argument("--sense-voice",     default="", metavar="PATH",
                     help="SenseVoice model.onnx  (multilingual, recommended)")
    asr.add_argument("--paraformer",      default="", metavar="PATH",
                     help="Paraformer model.onnx")
    asr.add_argument("--whisper-encoder", default="", metavar="PATH",
                     help="Whisper encoder.onnx")
    asr.add_argument("--whisper-decoder", default="", metavar="PATH",
                     help="Whisper decoder.onnx")
    asr.add_argument("--whisper-language", default="en")
    asr.add_argument("--whisper-task",    default="transcribe",
                     choices=["transcribe", "translate"])
    asr.add_argument("--whisper-tail-paddings", type=int, default=-1)
    asr.add_argument("--encoder",         default="", metavar="PATH",
                     help="Transducer encoder.onnx")
    asr.add_argument("--decoder",         default="", metavar="PATH",
                     help="Transducer decoder.onnx")
    asr.add_argument("--joiner",          default="", metavar="PATH",
                     help="Transducer joiner.onnx")

    # TTS: optional; pick one
    tts = p.add_argument_group("TTS backend  (optional; omit to disable TTS)")
    tts.add_argument("--vits-model",          default="", metavar="PATH")
    tts.add_argument("--vits-tokens",         default="", metavar="PATH")
    tts.add_argument("--vits-lexicon",        default="", metavar="PATH")
    tts.add_argument("--vits-data-dir",       default="", metavar="DIR",
                     help="espeak-ng-data directory for VITS-piper")
    tts.add_argument("--matcha-acoustic-model", default="", metavar="PATH")
    tts.add_argument("--matcha-vocoder",      default="", metavar="PATH")
    tts.add_argument("--matcha-lexicon",      default="", metavar="PATH")
    tts.add_argument("--matcha-tokens",       default="", metavar="PATH")
    tts.add_argument("--matcha-data-dir",     default="", metavar="DIR")
    tts.add_argument("--kokoro-model",        default="", metavar="PATH")
    tts.add_argument("--kokoro-voices",       default="", metavar="PATH")
    tts.add_argument("--kokoro-tokens",       default="", metavar="PATH")
    tts.add_argument("--kokoro-data-dir",     default="", metavar="DIR")
    tts.add_argument("--kokoro-lexicon",      default="", metavar="PATH")
    tts.add_argument("--tts-rule-fsts",       default="",
                     help="Comma-separated rule FST paths for TTS normalisation")
    tts.add_argument("--tts-sid",   type=int,   default=0,
                     help="Speaker/voice index for multi-speaker TTS models")
    tts.add_argument("--tts-speed", type=float, default=1.0,
                     help="TTS speed multiplier  (1.0 = normal)")

    # Optional speech denoiser
    p.add_argument(
        "--gtcrn-model", default="", metavar="PATH",
        help="(Optional) GTCRN online speech-denoiser .onnx",
    )

    # Pre-registered speakers
    p.add_argument(
        "--speaker-file", default="", metavar="PATH",
        help=(
            "(Optional) Text file with 'name /path/to/wav' lines. "
            "Multiple lines with the same name are averaged."
        ),
    )

    # VAD tuning
    vad_g = p.add_argument_group("VAD tuning")
    vad_g.add_argument("--min-silence-duration", type=float, default=0.5,
                       help="Seconds of silence to close a speech segment")
    vad_g.add_argument("--min-speech-duration",  type=float, default=0.25,
                       help="Minimum speech seconds to start a segment")
    vad_g.add_argument("--vad-threshold",        type=float, default=0.5,
                       help="Silero VAD probability threshold  (0–1)")

    # Speaker ID
    p.add_argument("--speaker-threshold", type=float, default=0.5,
                   help="Cosine-similarity threshold for speaker matching  (0–1)")

    # General
    p.add_argument("--num-threads", type=int, default=2,
                   help="ONNX Runtime thread count (all models share this value)")
    p.add_argument("--provider", default="cpu", choices=["cpu", "cuda", "coreml"],
                   help="ONNX Runtime execution provider")
    p.add_argument("--feature-dim", type=int, default=80,
                   help="Acoustic feature dimension (must match ASR model)")
    p.add_argument("--decoding-method", default="greedy_search",
                   choices=["greedy_search", "modified_beam_search"])
    p.add_argument("--device", default=None, type=int,
                   help="sounddevice input device index  (default: system default)")
    p.add_argument("--debug", action="store_true",
                   help="Enable sherpa-onnx debug output")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p.parse_args()


SPEAKER_STORE = "./speaker_store"

def main() -> None:
    args = _get_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s | %(message)s",
    )

    log.info("Loading models …")

    # Setup VAD
    vad, vad_window_size = _build_vad(args)

    # Setup speaker embedding extractor + registry
    spk_cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=args.speaker_model,
        num_threads=args.num_threads,
        debug=args.debug,
        provider=args.provider,
    )
    if not spk_cfg.validate():
        raise ValueError(f"Invalid speaker-embedding config: {spk_cfg}")
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(spk_cfg)
    log.info("Speaker extractor loaded: %s  dim=%d", args.speaker_model, extractor.dim)

    registry = SpeakerRegistry(extractor, threshold=args.speaker_threshold)
    registry.load(SPEAKER_STORE)
    if args.speaker_file:
        _preregister_speakers(args.speaker_file, extractor, registry)

    # Setup ASR
    recognizer = _build_recognizer(args)
    asr = ASREngine(recognizer)

    # Setup TTS (optional)
    raw_tts = _build_tts(args)
    tts: "TTSEngine | _NoOpTTS" = (
        TTSEngine(raw_tts, sid=args.tts_sid, speed=args.tts_speed)
        if raw_tts is not None
        else _NoOpTTS()
    )
    if raw_tts is None:
        log.info("No TTS model specified - TTS disabled.")

    # Setup speech enhancer (optional)
    enhancer = SpeechEnhancer(args.gtcrn_model, args.num_threads) if args.gtcrn_model else None

    # Start event bus
    bus = EventBus()
    _attach_console_handlers(bus, registry)

    # Audio capture
    capture = AudioCapture(device=args.device)

    # Create and start pipeline
    pipeline = ConversatorPipeline(
        bus=bus,
        capture=capture,
        vad=vad,
        vad_window_size=vad_window_size,
        registry=registry,
        asr=asr,
        tts=tts,
        enhancer=enhancer,
    )

    log.info("All models ready.  Starting pipeline …")
    pipeline.run()

    registry.save(SPEAKER_STORE)

    # Session summary after Ctrl+C
    print("\n" + "═" * 60)
    print("SESSION TRANSCRIPT SUMMARY")
    print("═" * 60)
    all_text = pipeline.session.get_all()
    if all_text:
        for spk_id, text in all_text.items():
            name = registry.get_display_name(spk_id)
            print(f"  {name} [{spk_id[:8]}]:")
            print(f"    {text}")
    else:
        print("  (no speech captured)")
    print("═" * 60)


if __name__ == "__main__":
    main()
