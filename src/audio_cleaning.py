import torchaudio
from deepfilternet import DeepFilterNet

# Lade das Noise-Filter-Modell
denoiser = DeepFilterNet.from_pretrained("facebook/deepfilternet-v2")


def clean_audio(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

    # Wende Noise Suppression an
    filtered_audio = denoiser(waveform)

    # Speichere das gefilterte Audio
    output_path = "cleaned_audio.wav"
    torchaudio.save(output_path, filtered_audio, 16000)
    return output_path