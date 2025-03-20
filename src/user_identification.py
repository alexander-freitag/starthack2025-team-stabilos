import io
import os
import uuid
import wave
from scipy.signal import resample
import numpy as np
import pveagle

EAGLE_KEY = os.getenv("EAGLE_KEY")

enrollment_progress = {}

def identify_speaker(audio_chunk: bytes, eagle_profiles):

    # Eagle Erkennung
    try:
        eagle = pveagle.create_recognizer(access_key=EAGLE_KEY, speaker_profiles=list(eagle_profiles.values()))
    except pveagle.EagleError:
        print("Fehler bei Eagle-Erstellung")
        return None

    resampled_chunk = convert_wav_bytes_to_pcm(audio_chunk)

    for i in range(0, len(resampled_chunk), eagle.frame_length):
        chunk = resampled_chunk[i:i + eagle.frame_length]  # Schneide 512 Samples aus

        if len(chunk) < eagle.frame_length:
            print(f"⚠️ Warnung: Letzter Chunk hat nur {len(chunk)} statt {eagle.frame_length} Samples. Wird ignoriert.")
            break  # Verlasse die Schleife, da der letzte Chunk zu klein ist

        scores = eagle.process(chunk)  # Sende den Chunk an Eagle
        print(f"🔹 Verarbeitetes Chunk {i // eagle.frame_length + 1}: {scores}")

        max_score = max(scores)
        print("eagle scores for chunk" + str(scores))
        print("max score" + str(max_score))
        best_match_index = np.argmax(scores)
        if max_score > 0.9:
            user_id = list(eagle_profiles.keys())[best_match_index]
            print(f"✅ Erkannter Nutzer: {user_id} (Score: {max_score:.2f})")
            return user_id


    eagle.delete()
    return None

def enroll_speaker(chat_session_id, audio_chunk: bytes):
    global enrollment_progress

    if chat_session_id not in enrollment_progress:
        enrollment_progress[chat_session_id] = (None, 0.0)  # Neue Enrollment-Session

    stored_audio, progress = enrollment_progress[chat_session_id]
    if stored_audio is not None:
        stored_audio = stored_audio + audio_chunk
    else:
        stored_audio = audio_chunk

    # Eagle Profiler erstellen
    try:
        eagle_profiler = pveagle.create_profiler(access_key=EAGLE_KEY)
    except pveagle.EagleError:
        print("Fehler bei Profiler-Erstellung")
        return None, None

    resampled_audio = convert_wav_bytes_to_pcm(stored_audio)

    # Enrollment durchführen
    enroll_percentage, feedback = eagle_profiler.enroll(resampled_audio)
    print(enroll_percentage, feedback)
    enrollment_progress[chat_session_id] = (stored_audio, enroll_percentage)

    print(f"🔄 Enrollment-Fortschritt für {chat_session_id}: {enroll_percentage:.2f}%")

    # Falls 100% erreicht, Profil exportieren & speichern
    if enroll_percentage >= 100.0:
        speaker_profile = eagle_profiler.export()
        print(f"✅ Enrollment abgeschlossen für {speaker_profile}!")
        enrollment_progress[chat_session_id] = (speaker_profile, enroll_percentage)
        return uuid.uuid4(), speaker_profile

    return None, None


def convert_wav_bytes_to_pcm(wav_bytes, target_sample_rate=16000):
    """
    Konvertiert rohe WAV-Bytes in PCM-Format für Eagle.

    - Extrahiert PCM-Daten aus WAV
    - Falls nötig, wird von 8000 Hz auf 16000 Hz hochskaliert
    - Gibt 16-bit PCM-Daten als numpy-Array zurück

    :param wav_bytes: WAV-Datei als Bytes
    :param target_sample_rate: Ziel-Sample-Rate für Eagle (Standard: 16000 Hz)
    :return: PCM-Daten als numpy-Array (int16)
    """

    # 📌 WAV-Bytes als Datei öffnen
    with wave.open(io.BytesIO(wav_bytes), 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        num_channels = wav_file.getnchannels()
        pcm_data = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)

    # 📌 Falls Stereo, nur ersten Kanal nutzen
    if num_channels > 1:
        pcm_data = pcm_data[::num_channels]

    # 📌 Falls Sample-Rate nicht 16000 Hz, resamplen
    if sample_rate != target_sample_rate:
        factor = target_sample_rate / sample_rate
        pcm_data = resample(pcm_data, int(len(pcm_data) * factor)).astype(np.int16)

    return pcm_data
