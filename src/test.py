import torchaudio
import soundfile as sf
import torch
import numpy as np
from deepfilternet import DeepFilterNet

# Lade das DeepFilterNet-Modell für Rauschentfernung
denoiser = DeepFilterNet.from_pretrained("facebook/deepfilternet-v2")

# Lade das Silero VAD-Modell für Sprecherfokussierung
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
(get_speech_timestamps, _, _, _, collect_chunks) = utils


def clean_audio(audio_chunk, sample_rate=16000):
    """ Entfernt Hintergrundgeräusche aus einem Audio-Chunk """
    waveform = torch.tensor(audio_chunk, dtype=torch.float32).unsqueeze(0)
    cleaned_audio = denoiser(waveform)
    return cleaned_audio.squeeze(0).numpy()


def isolate_speaker(audio_chunk, sample_rate=16000):
    """ Extrahiert nur den Hauptsprecher aus einem Audio-Chunk """
    waveform = torch.tensor(audio_chunk, dtype=torch.float32).unsqueeze(0)
    speech_timestamps = get_speech_timestamps(waveform, vad_model, sampling_rate=sample_rate)

    if not speech_timestamps:
        return None  # Falls keine Sprache erkannt wird

    cleaned_audio = collect_chunks(speech_timestamps, waveform)
    return cleaned_audio.squeeze(0).numpy()


def process_audio(input_file, output_cleaned, output_isolated):
    """ Führt Cleaning & Speaker Isolation durch und speichert Audios """

    # 1️⃣ Lade Original-Audio
    waveform, sample_rate = torchaudio.load(input_file)
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    waveform = waveform.squeeze(0).numpy()  # In numpy-Array konvertieren

    # Speichere Original für Vergleich
    sf.write("before.wav", waveform, 16000)

    # 2️⃣ Entferne Rauschen
    cleaned = clean_audio(waveform)
    sf.write(output_cleaned, cleaned, 16000)  # Speichere bereinigtes Audio

    # 3️⃣ Isoliere Hauptsprecher
    isolated = isolate_speaker(cleaned)
    if isolated is not None:
        sf.write(output_isolated, isolated, 16000)  # Speichere Sprecher-isoliertes Audio
    else:
        print("⚠ Kein Sprecher erkannt!")

    print("✅ Verarbeitung abgeschlossen. Vergleiche die Audiodateien!")


# 🔥 TEST AUSFÜHREN
process_audio("test.wav", "cleaned.wav", "isolated.wav")
