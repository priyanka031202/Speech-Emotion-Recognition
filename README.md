# Speech Emotion Recognition 
### This is a simple and interactive web app that can tell what emotion someone is feeling just by listening to their voice. You can upload a .wav audio file, and the app will show the emotion it hears—along with a spectrogram (a colorful sound graph) and a matching emoji.It uses deep learning and audio processing tools like TensorFlow, librosa, and is built with Streamlit to run on the web.



## Supported Emotions
#### 1. Sad
#### 2. Angry	
#### 3. Happy	
#### 4. Neutral	
#### 5. Fearful	
#### 6. Disgust	
#### 7. Surprise	
#### 8. Calm

# How we prepared data
## Data Augmentation
Data augmentation helps solve this by creating more variety in the training data. We made slight changes to the original audio files, such as:
1. Adding background noise 
2. Changing pitch 
3. Time-stretching

## Feature Extraction
1. Used librosa – A Python library for audio analysis, to extract useful features from  audio files.

2. Trimmed Silence – We first removed unnecessary silent parts from the beginning and end of each audio clip using librosa.effects.trim().

3. Extracted Audio Features:

- MFCCs (Mel Frequency Cepstral Coefficients)
  Capture the shape of the sound and are great at identifying tone and timbre.

- Chroma Features
  Represent the 12 different pitch classes (like musical notes) – useful for detecting emotion through tone.

- Zero Crossing Rate
  Measures how often the signal changes sign – helps identify intensity or sharpness in voice.

- Mel Spectrogram
  Shows the energy of different frequencies – visually similar to what our ears hear.

- Spectral Centroid & Bandwidth
  Tell us where most of the sound’s energy is located and how wide it is.

- Spectral Contrast
  Measures the difference between peaks and valleys in the spectrum – useful for emotion shifts.

- Tonnetz (Tonal Centroid Features)
  Capture harmony and tonal characteristics, helping detect mood

  # How we trained model
  We trained a Convolutional Neural Network (CNN) using extracted audio features like MFCCs, Chroma, Mel Spectrogram, and others. Before training, we applied data augmentation techniques such as adding background   noise, changing pitch, and stretching time to make the model more robust and handle real-world variations better.

  The extracted features were scaled using a Standard Scaler, and labels were encoded using a Label Encoder. The model was then trained on this processed data to learn patterns related to different emotions in      speech. Once trained, the model was saved in .h5 format and later used in a Streamlit web app for live predictions.

# After traning the data
  ![image](https://github.com/user-attachments/assets/455cdfbb-cbc4-47bf-aeda-0e92d5806b43) ## This is the confusion matrix
  
  
  ![image](https://github.com/user-attachments/assets/743ce796-7258-4555-8321-1ca018fdaa12) ## Report
  
  
  ![image](https://github.com/user-attachments/assets/3b58f684-cd73-4b13-ab80-227e804c16a4) ## overall accuracy

  
  ![image](https://github.com/user-attachments/assets/d0afe755-6f06-447a-9d10-0df3ca40b34b) ## overall F1 score amd accuracy per class





  # Project Structure
    repositiory : Speech-Emotion-Recognition that contain the below files
    app.py                  
    model.h5                
    scaler.pkl              
    label_encoder.pkl       
    requirements.txt        
    runtime.txt         
    README.md

  # How to run it Locally on windows
  #### 1 Clone the repository
       git clone https://github.com/priyanka031202/Speech-Emotion-Recognition.git
       cd Speech-Emotion-Recognition    
  #### 2 Set up a virtual environment
        python -m venv venv
        .\venv\Scripts\activate.bat
  #### 3 Install dependencies
        pip install -r requirements.txt
  #### Run the Streamlit app
        streamlit run app.py
# Deployed app link
 https://speech-emotion-recognition-pfgqm6ztyukqkizluvnrbp.streamlit.app/

 # Requirements
 streamlit>=1.32.0
 numpy>=1.26.0
 librosa>=0.10.1
 tensorflow>=2.16.0
 plotly>=5.18.0
 pandas>=2.1.3
 soundfile>=0.12.1
 scikit-learn>=1.3.2
 pickle-mixin>=1.0.2
 matplotlib==3.7.1


 - Made by using python and streamlit cloud 

  






