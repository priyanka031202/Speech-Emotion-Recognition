# Updated app.py with audio recording, spectrogram, and emoji support
import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import soundfile as sf
import tempfile
import io

# Emojis for emotions 
EMOJI_MAP = {
    "happy": "😃",
    "sad": "😢",
    "angry": "😠",
    "neutral": "😐",
    "fear": "😱",
    "disgust": "🤮",
    "surprise": "😮"
}

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
def predict_emotion(audio_file):
    try:
        data, sr = librosa.load(audio_file, duration=2.5, offset=0.6)
        features = extract_features(data, sr)
        if features.shape[0] != 197:
            return "Feature size mismatch. Got {} features.".format(features.shape[0]), None
        features = scaler.transform(features.reshape(1, -1))
        features = np.expand_dims(features, axis=2)
        prediction = model.predict(features)
        predicted_label = encoder.inverse_transform(prediction)
        return predicted_label[0][0], (data, sr)
    except Exception as e:
        return f"Error: {e}", None

# Streamlit UI
st.title("Speech Emotion Recognition")
st.write("Upload a `.wav` file or record audio to detect the emotion")

# Audio uploader
uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

# Audio recorder
recorded_audio = st.audio_recorder("Click to record", sample_rate=22050)

# Handling uploaded or recorded audio
audio_path = None
if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    audio_path = "temp.wav"
    st.audio(uploaded_file, format="audio/wav")
elif recorded_audio is not None:
    audio_bytes = recorded_audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        audio_path = tmp.name
    st.audio(audio_bytes, format="audio/wav")

# Prediction and Spectrogram
if audio_path:
    result, audio_data = predict_emotion(audio_path)
    if audio_data:
        signal, sr = audio_data
        st.markdown(f"### Predicted Emotion: **{result}** {EMOJI_MAP.get(result.lower(), '')}")

        # Display spectrogram
        fig, ax = plt.subplots()
        S = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
        S_DB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        ax.set(title='Mel-frequency spectrogram')
        st.pyplot(fig)
    else:
        st.error(result)
