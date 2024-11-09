# Importing project dependencies
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.utils import register_keras_serializable, get_custom_objects
import time
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import matplotlib as mpl
import tempfile

mpl.rcParams['agg.path.chunksize'] = 10000

# Required Parameters for audio
sample_rate = 8000
min_duration = 1.0  
frame_length = 8064
hop_length_frame = 8064
hop_length_frame_noise = 5000
nb_samples = 500
n_fft = 255
hop_length_fft = 63

# Define any custom classes/layers if used in the model
# Example:
# @register_keras_serializable()
# class CustomLayer(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(CustomLayer, self).__init__(**kwargs)
#         # Initialize parameters
#     
#     def call(self, inputs):
#         return inputs  # Replace with actual computation

# Load the model once and cache it
@st.cache_resource
def load_denoising_model():
    try:
        # Option 1: Using SavedModel format
        # model = load_model('model_unet')  # Ensure 'model_unet' directory is present
        # Option 2: Using JSON and weights
        with open('model_unet.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json, custom_objects=get_custom_objects())
        model.load_weights('model_unet.h5')
        st.success("Loaded model from disk")
        return model
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# Scaling functions
def scaled_in(matrix_spec):
    return (matrix_spec + 46) / 50

def scaled_ou(matrix_spec):
    return (matrix_spec - 6) / 82

def inv_scaled_ou(matrix_spec):
    return matrix_spec * 82 + 6

# Audio processing functions
def audio_to_audio_frame_stack(sound_data, frame_length, hop_length_frame):
    sequence_sample_length = sound_data.shape[0]
    sound_data_list = [sound_data[start:start + frame_length] 
                       for start in range(0, sequence_sample_length - frame_length + 1, hop_length_frame)]
    return np.vstack(sound_data_list)

def audio_files_to_numpy(audio_dir, list_audio_files, sample_rate, frame_length, hop_length_frame, min_duration):
    list_sound_array = []
    for file in list_audio_files:
        try:
            y, sr = librosa.load(os.path.join(audio_dir, file), sr=sample_rate)
            total_duration = librosa.get_duration(y=y, sr=sr)
            if total_duration >= min_duration:
                list_sound_array.append(audio_to_audio_frame_stack(y, frame_length, hop_length_frame))
            else:
                print(f"File {file} is below min duration.")
        except ZeroDivisionError:
            continue
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    if list_sound_array:
        return np.vstack(list_sound_array)
    else:
        return np.empty((0, frame_length))

def audio_to_magnitude_db_and_phase(n_fft, hop_length_fft, audio):
    stftaudio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length_fft)
    stftaudio_magnitude, stftaudio_phase = librosa.magphase(stftaudio)
    stftaudio_magnitude_db = librosa.amplitude_to_db(stftaudio_magnitude, ref=np.max)
    return stftaudio_magnitude_db, stftaudio_phase

def numpy_audio_to_matrix_spectrogram(numpy_audio, dim_square_spec, n_fft, hop_length_fft):
    nb_audio = numpy_audio.shape[0]
    m_mag_db = np.zeros((nb_audio, dim_square_spec, dim_square_spec))
    m_phase = np.zeros((nb_audio, dim_square_spec, dim_square_spec), dtype=complex)
    for i in range(nb_audio):
        m_mag_db[i, :, :], m_phase[i, :, :] = audio_to_magnitude_db_and_phase(n_fft, hop_length_fft, numpy_audio[i])
    return m_mag_db, m_phase

def magnitude_db_and_phase_to_audio(frame_length, hop_length_fft, stftaudio_magnitude_db, stftaudio_phase):
    stftaudio_magnitude_rev = librosa.db_to_amplitude(stftaudio_magnitude_db, ref=1.0)
    audio_reverse_stft = stftaudio_magnitude_rev * stftaudio_phase
    audio_reconstruct = librosa.core.istft(audio_reverse_stft, hop_length=hop_length_fft, length=frame_length)
    return audio_reconstruct

def matrix_spectrogram_to_numpy_audio(m_mag_db, m_phase, frame_length, hop_length_fft):
    list_audio = [magnitude_db_and_phase_to_audio(frame_length, hop_length_fft, m_mag_db[i], m_phase[i])
                  for i in range(m_mag_db.shape[0])]
    return np.vstack(list_audio)

# Prediction function
def prediction(model, audio_dir_prediction, dir_save_prediction, audio_input_prediction, audio_output_prediction):
    if model is None:
        st.error("Model could not be loaded. Please check the model files.")
        return
    
    try:
        audio = audio_files_to_numpy(audio_dir_prediction, audio_input_prediction, sample_rate,
                                     frame_length, hop_length_frame, min_duration)
        if audio.size == 0:
            st.error("No valid audio data found for prediction.")
            return
        dim_square_spec = int(n_fft / 2) + 1
        m_amp_db_audio, m_pha_audio = numpy_audio_to_matrix_spectrogram(audio, dim_square_spec, n_fft, hop_length_fft)
        X_in = scaled_in(m_amp_db_audio).reshape(-1, dim_square_spec, dim_square_spec, 1)
        
        X_pred = model.predict(X_in)
        inv_sca_X_pred = inv_scaled_ou(X_pred)
        X_denoise = m_amp_db_audio - inv_sca_X_pred[:, :, :, 0]
        
        audio_denoise_recons = matrix_spectrogram_to_numpy_audio(X_denoise, m_pha_audio, frame_length, hop_length_fft)
        nb_samples = audio_denoise_recons.shape[0]
        denoise_long = audio_denoise_recons.reshape(1, nb_samples * frame_length) * 10
        output_path = os.path.join(dir_save_prediction, audio_output_prediction)
        sf.write(output_path, denoise_long[0, :], 8000, 'PCM_24')
        
        st.success("Denoised audio saved successfully.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Streamlit interface
st.title("Audio Denoising")
model = load_denoising_model()

# Clean up existing files
for file in ['input.wav', 'denoised.wav']:
    if os.path.exists(file):
        try:
            os.remove(file)
        except Exception as e:
            st.warning(f"Could not remove existing file {file}: {e}")

st.subheader("Choose an Audio file")
uploaded_file = st.file_uploader("Choose an audio file", type=['wav'])

if uploaded_file is not None:
    file_details = {
        "File Name": uploaded_file.name,
        "File Type": uploaded_file.type,
        "File Size": uploaded_file.size
    }
    st.write(file_details)
    
    if file_details['File Type'] == 'audio/wav':
        input_path = 'input.wav'
        try:
            with open(input_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            st.subheader("Input")
            st.audio(input_path)
        except Exception as e:
            st.error(f"Failed to save uploaded audio: {e}")
    
        st.subheader("Input Audio Time Series Plot")
        try:
            x, sr = sf.read(input_path)
            fig, ax = plt.subplots(figsize=(40, 15))
            ax.plot(x)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error reading input audio: {e}")
    
        prediction(model, '', '', ['input.wav'], 'denoised.wav')
        
        with st.spinner("Denoising the audio..."):
            # Assuming prediction is synchronous
            time.sleep(2)  # Adjust based on actual processing time
            if os.path.exists('denoised.wav'):
                st.success("Denoised!!")
                st.subheader("Output Audio Time Series Plot")
                try:
                    x, sr = sf.read('denoised.wav')
                    fig, ax = plt.subplots(figsize=(40, 15))
                    ax.plot(x)
                    st.pyplot(fig)
                    st.audio('denoised.wav')
                except Exception as e:
                    st.error(f"Error reading denoised audio: {e}")
            else:
                st.error("Denoised audio file was not created.")