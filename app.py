import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import tempfile
import os

# Load model and preprocessing tools
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, scaler, encoder

model, scaler, encoder = load_model()

# Extract audio features
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

# Predict emotion
def predict_emotion(audio_file_path):
    try:
        data, sr = librosa.load(audio_file_path, duration=2.5, offset=0.6)
        features = extract_features(data, sr)

        if features.shape[0] != 197:
            return f"Feature size mismatch. Got {features.shape[0]} features.", None, None

        features = scaler.transform(features.reshape(1, -1))
        features = np.expand_dims(features, axis=2)
        prediction = model.predict(features)
        predicted_label = encoder.inverse_transform(prediction)
        return predicted_label[0][0], data, sr
    except Exception as e:
        return f"Error: {e}", None, None

# Emoji dictionary
emoji_dict = {
    'angry': '😠',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢',
    'fearful': '😨',
    'disgust': '🤢',
    'surprise': '😲',
    'calm': '😌'
}

# Streamlit UI
st.title("🎤 Speech Emotion Recognition")
st.write("Upload an audio file (`.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`) to predict the emotion.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "flac", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    # Save uploaded file to a temporary path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_path = temp_audio.name

    # Predict
    result, audio_data, sr = predict_emotion(temp_path)
    os.remove(temp_path)

    if audio_data is not None:
        emoji = emoji_dict.get(result.lower(), "🎭")
        st.markdown(f"###  Predicted Emotion: **{result}** {emoji}")

        # Plot mel spectrogram
        fig, ax = plt.subplots(figsize=(10, 4))
        S = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
        S_DB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel Spectrogram')
        st.pyplot(fig)
    else:
        st.error(result)
