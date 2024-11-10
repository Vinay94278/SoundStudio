from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import numpy as np
import librosa
import soundfile as sf
from tensorflow.keras.models import model_from_json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model once and cache it
def load_denoising_model():
    with open('model_unet.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights('model_unet.h5')
    print("Loaded model from disk")
    return model

model = load_denoising_model()

# Required Parameters for audio
sample_rate = 8000
min_duration = 1.0  
frame_length = 8064
hop_length_frame = 8064
hop_length_frame_noise = 5000
nb_samples = 500
n_fft = 255
hop_length_fft = 63

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
    return np.vstack(list_sound_array)

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

@app.route('/denoise', methods=['POST'])
def denoise_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        audio = audio_files_to_numpy('', [file_path], sample_rate, frame_length, hop_length_frame, min_duration)
        dim_square_spec = int(n_fft / 2) + 1
        m_amp_db_audio, m_pha_audio = numpy_audio_to_matrix_spectrogram(audio, dim_square_spec, n_fft, hop_length_fft)
        X_in = scaled_in(m_amp_db_audio).reshape(-1, dim_square_spec, dim_square_spec, 1)

        X_pred = model.predict(X_in)
        inv_sca_X_pred = inv_scaled_ou(X_pred)
        X_denoise = m_amp_db_audio - inv_sca_X_pred[:, :, :, 0]

        audio_denoise_recons = matrix_spectrogram_to_numpy_audio(X_denoise, m_pha_audio, frame_length, hop_length_fft)
        nb_samples = audio_denoise_recons.shape[0]
        denoise_long = audio_denoise_recons.reshape(1, nb_samples * frame_length) * 10
        output_file_path = os.path.join('outputs', 'denoised_' + file.filename)
        sf.write(output_file_path, denoise_long[0, :], 8000, 'PCM_24')

        return send_file(output_file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)