import numpy as np
import librosa
import sys
from tensorflow.keras.models import load_model
import math

# ========== PARAMETERS (same as training) ==========
sr = 22050
n_fft = 2048
hop_length = 512
n_mels = 128
max_samples = 3 * sr
max_frames = math.ceil(max_samples / hop_length)

# Label maps (EDIT if your order differs)
emotion_map = {
    0: "Neutral",
    1: "Calm",
    2: "Happy",
    3: "Sad",
    4: "Angry",
    5: "Fearful",
    6: "Disgust",
    7: "Surprised"
}

gender_map = {
    0: "Male",
    1: "Female"
}

# ========== FUNCTIONS ==========

def preprocessing(y, sr):
    y, _ = librosa.effects.trim(y)
    y = librosa.util.normalize(y)
    return y

def fix_mel_shape(mel):
    if mel.shape[1] < max_frames:
        pad_width = max_frames - mel.shape[1]
        mel = np.pad(mel, ((0,0),(0,pad_width)), mode='constant')
    else:
        mel = mel[:, :max_frames]
    return mel

def process_file(path):
    y, _ = librosa.load(path, sr=sr)

    y = preprocessing(y, sr)

    if len(y) < max_samples:
        pad_amount = max_samples - len(y)
        y = np.pad(y, (0,pad_amount))
    else:
        start = (len(y) - max_samples)//2
        y = y[start:start+max_samples]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    mel = fix_mel_shape(mel)

    log_mel = librosa.power_to_db(mel, ref=np.max)

    # IMPORTANT: Use SAME normalization you used in training
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)

    log_mel = np.expand_dims(log_mel, axis=-1)
    log_mel = np.expand_dims(log_mel, axis=0)

    return log_mel

# ========== MAIN ==========

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py audio.wav")
        sys.exit()

    audio_path = sys.argv[1]

    model = load_model("src/ser_model.keras", compile=False)

    X = process_file(audio_path)

    emotion_pred, gender_pred = model.predict(X)

    emotion_idx = np.argmax(emotion_pred)
    gender_idx = np.argmax(gender_pred)

    print("\nPrediction:")
    print("Emotion:", emotion_map[emotion_idx])
    print("Gender:", gender_map[gender_idx])
