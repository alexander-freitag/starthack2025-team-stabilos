from speechbrain.pretrained import EncoderClassifier
import numpy as np
import torch

# Lade das Speaker-Embedding-Modell
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmpdir")

# Datenbank für bekannte Sprecher (ID → Stimmprofil)
known_speakers = {
    "user_123": np.array([...]),  # Beispiel-Vektor
    "user_456": np.array([...])
}


def identify_speaker(audio_path):
    # Lade das bereinigte Audio
    signal, fs = torchaudio.load(audio_path)

    # Erstelle das Speaker-Embedding
    embedding = classifier.encode_batch(signal).detach().numpy().mean(axis=0)

    # Prüfe, ob der Sprecher bekannt ist
    best_match = None
    best_score = float("inf")

    for user_id, known_embedding in known_speakers.items():
        score = np.linalg.norm(embedding - known_embedding)  # Euklidische Distanz
        if score < best_score:
            best_score = score
            best_match = user_id

    # Schwellenwert setzen (z.B. 0.5 für bekannte Sprecher)
    if best_score < 0.5:
        return best_match  # Bekannter Sprecher → Nutzer-ID zurückgeben

    # Unbekannter Sprecher → Neue UUID generieren
    new_speaker_id = str(uuid.uuid4())
    known_speakers[new_speaker_id] = embedding
    return new_speaker_id
