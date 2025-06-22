import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import soundfile as sf
from streamlit_audiorecorder import audiorecorder

# Load model and preprocessing tools
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, scaler, encoder

model, scaler, encoder = load_model()

# Emotion to emoji mapping
EMOJI_MAP = {
    "angry": "😠", "happy": "😄", "sad": "😢", "neutral": "😐",
    "fear": "😨", "surprise": "😲", "disgust": "🤢", "calm": "😌"
}

# Feature extraction
def extract_features(data, sample_rate):
    result = np.array([])
    data, _ = librosa.effects.trim(data)
    stft = np.abs(librosa.stft(data))

    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    centroid = np.mean(librosa.feature.spectral_centroid(y=data, sr=sample_rate).T, axis=0)
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=data, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sample_rate).T, axis=0)

    result = np.hstack((zcr, chroma_stft, mfcc, rms, mel, centroid, bandwidth, contrast, tonnetz))
    return result

# Prediction function
def predict_emotion(audio_path):
    try:
        data, sr = librosa.load(audio_path, duration=2.5, offset=0.6)
        features = extract_features(data, sr)

        if features.shape[0] != 197:
            return "Feature mismatch (got {} features)".format(features.shape[0])

        features = scaler.transform(features.reshape(1, -1))
        features = np.expand_dims(features, axis=2)

        prediction = model.predict(features)
        predicted_label = encoder.inverse_transform(prediction)
        return predicted_label[0][0]
    except Exception as e:
        return f"Error: {e}"

# Plot spectrogram
def show_spectrogram(file_path):
    y, sr = librosa.load(file_path)
    fig, ax = plt.subplots()
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set(title='Mel-Spectrogram')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    st.pyplot(fig)

# Streamlit App
st.title("🎤 Speech Emotion Recognition")
st.write("Upload a `.wav` file or record your voice to predict emotion.")

# Audio upload
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
recording = audiorecorder("Click to record", "Recording...")

audio_path = None

if uploaded_file:
    audio_path = "temp_uploaded.wav"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.audio(uploaded_file, format="audio/wav")

elif recording is not None and len(recording) > 0:
    audio_path = "temp_recorded.wav"
    with open(audio_path, "wb") as f:
        f.write(recording)
    st.audio(audio_path, format="audio/wav")

# Predict emotion
if audio_path:
    show_spectrogram(audio_path)
    emotion = predict_emotion(audio_path)
    emoji = EMOJI_MAP.get(emotion.lower(), "🎧")
    st.markdown(f"### Predicted Emotion: **{emotion}** {emoji}")
