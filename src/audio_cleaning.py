import torch
import noisereduce as nr

# Lade das Silero VAD-Modell
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)

# Extrahiere nur die benötigten Funktionen aus utils
(get_speech_timestamps, _, _, _, collect_chunks) = utils

def isolate_speaker(audio_chunk, sample_rate=16000):
    """ Extrahiert nur den Hauptsprecher aus einem Audio-Chunk """
    waveform = torch.tensor(audio_chunk, dtype=torch.float32).unsqueeze(0)

    speech_timestamps = get_speech_timestamps(waveform, vad_model, sampling_rate=sample_rate)
    if not speech_timestamps:
        print("keine sprecher erkannt")
        return audio_chunk

    cleaned_audio = collect_chunks(speech_timestamps, waveform)
    return cleaned_audio.squeeze(0).numpy()

def clean_audio_chunk(audio_chunk):
    """
    Entfernt Hintergrundgeräusche aus einem Audio-Chunk mit RNNoise.
    :param audio_chunk: NumPy-Array mit PCM-Audiodaten (float32)
    :return: Bereinigtes NumPy-Array
    """
    return nr.reduce_noise(y=audio_chunk, sr=16000)
