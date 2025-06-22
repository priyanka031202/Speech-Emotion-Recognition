
import numpy as np
import librosa
import tensorflow as tf
import joblib
import os


MODEL_PATH = "models/model.h5"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)


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


def predict_emotion(audio_path):
    try:
        print(f"\n Processing: {audio_path}")
        data, sr = librosa.load(audio_path, duration=2.5, offset=0.6)
        print(f" Audio loaded | Sample rate: {sr} | Duration: {len(data)/sr:.2f} sec")
        features = extract_features(data, sr)
        print(f" Extracted feature shape: {features.shape}")
        if features.shape[0] != 197:
            raise ValueError(f" Feature mismatch: got {features.shape[0]}, expected 197.")
        features_scaled = scaler.transform(features.reshape(1, -1))
        features_scaled = np.expand_dims(features_scaled, axis=2)
        prediction = model.predict(features_scaled)
        predicted_label = encoder.inverse_transform(prediction)
        print(f"\n Predicted Emotion: {predicted_label[0][0]}")
        return predicted_label[0][0]
    except Exception as e:
        print(f" Error during prediction: {e}")
        return None


if __name__ == "__main__":
    test_audio_path = "test.wav" 
    if os.path.exists(test_audio_path):
        predict_emotion(test_audio_path)
    else:
        print(f" File not found: {test_audio_path}")
