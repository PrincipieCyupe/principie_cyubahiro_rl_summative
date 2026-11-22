"""Makes the demo sounds once: spoken move cues (offline Windows text to speech) and a scare whistle."""

import wave
from pathlib import Path

import numpy as np

AUDIO_DIR = Path(__file__).resolve().parent / "audio"
WORDS = ["up", "down", "left", "right"]
SAMPLE_RATE = 22050


def make_voice_cues():
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", 200)
    for word in WORDS:
        engine.save_to_file(word, str(AUDIO_DIR / f"{word}.wav"))
    engine.runAndWait()
    for word in WORDS:
        trim_silence(AUDIO_DIR / f"{word}.wav")


def trim_silence(path):
    # the voice files come with long quiet parts before and after the word, cut them off
    # so the word is heard straight away, even when the game moves quickly
    with wave.open(str(path), "rb") as file:
        params = file.getparams()
        samples = np.frombuffer(file.readframes(params.nframes), dtype=np.int16)
    loud = np.where(np.abs(samples) > 0.02 * np.abs(samples).max())[0]
    if len(loud) == 0:
        return
    rate = params.framerate * params.nchannels
    start = max(0, loud[0] - int(0.01 * rate))
    end = min(len(samples), loud[-1] + int(0.1 * rate))
    with wave.open(str(path), "wb") as file:
        file.setparams(params)
        file.writeframes(samples[start:end].tobytes())


def make_scare_sound():
    # a short whistle that slides upward, with a quick fade in and out
    duration = 0.35
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    pitch = np.linspace(500, 1300, t.size)
    phase = 2 * np.pi * np.cumsum(pitch) / SAMPLE_RATE
    tone = np.sin(phase) + 0.3 * np.sin(2 * phase)
    fade = np.minimum(1, np.minimum(t / 0.02, (duration - t) / 0.08))
    samples = (tone / 1.3 * fade * 0.6 * 32767).astype(np.int16)

    with wave.open(str(AUDIO_DIR / "scare.wav"), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(SAMPLE_RATE)
        file.writeframes(samples.tobytes())


if __name__ == "__main__":
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    make_scare_sound()
    print("Saved scare.wav")
    try:
        make_voice_cues()
        print("Saved " + ", ".join(f"{w}.wav" for w in WORDS))
    except Exception as error:
        print(f"Could not make the voice cues: {error}")
    print(f"Audio files are in {AUDIO_DIR}")
