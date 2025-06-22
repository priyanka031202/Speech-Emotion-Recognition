# app.py
import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

# Load model and preprocessing tools
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, scaler, encoder

model, scaler, encoder = load_model()

# Feature extraction
def extract_features(data, sample_rate):
    result = np.array([])
    data, _ = librosa.effects.trim(data)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfcc))
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    centroid = np.mean(librosa.feature.spectral_centroid(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, centroid))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, bandwidth))
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, contrast))
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sample_rate).T, axis=0)
    result = np.hstack((result, tonnetz))
    return result

# Predict function
def predict_emotion(audio_path):
    try:
        data, sr = librosa.load(audio_path, duration=2.5, offset=0.6)
        features = extract_features(data, sr)
        if features.shape[0] != 197:
            return "❌ Feature size mismatch. Got {} features.".format(features.shape[0])
        features = scaler.transform(features.reshape(1, -1))
        features = np.expand_dims(features, axis=2)
        prediction = model.predict(features)
        predicted_label = encoder.inverse_transform(prediction)
        return predicted_label[0][0], data, sr
    except Exception as e:
        return f"❌ Error: {e}", None, None

# Map emotions to emojis
EMOJI_MAP = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😠",
    "neutral": "😐",
    "fear": "😱",
    "disgust": "🤢",
    "surprise": "😲"
}

# Streamlit UI
st.title("🎤 Speech Emotion Recognition")
st.write("Upload a `.wav` file to detect the emotion and see the spectrogram.")

uploaded_file = st.file_uploader("📂 Choose a WAV audio file", type=["wav"])

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file, format="audio/wav")

    predicted_emotion, audio_data, sr = predict_emotion("temp.wav")

    if audio_data is not None:
        # Show prediction
        emoji = EMOJI_MAP.get(predicted_emotion.lower(), "🎧")
        st.markdown(f"### 🎯 Predicted Emotion: **{predicted_emotion}** {emoji}")

        # Plot spectrogram
        st.markdown("#### 🔊 Spectrogram")
        fig, ax = plt.subplots(figsize=(10, 4))
        S = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
        S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        plt.colorbar(format='%+2.0f dB')
        st.pyplot(fig)
