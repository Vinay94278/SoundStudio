# Audio Denoising

## Methodology

- **Loading clean and noisy audio data:** The first step is to load the audio files that contain clean speech and noise, respectively.
- **Blending a random noise to clean speech audio:** The second step is to add a random noise to the clean speech audio, creating a mixed audio file that simulates a noisy environment.
- **Extracting STFT features from the audio:** The third step is to convert the audio signals into spectrograms using Short-Time Fourier Transform (STFT), which capture the frequency and time characteristics of the sound.
- **Training the model and saving it:** The fourth step is to train a convolutional neural network (CNN) model that takes the noisy spectrogram as input and outputs a mask that indicates the ratio of speech to noise in each frequency bin. The model is trained using the clean and mixed spectrograms, as well as the user input, as the data. The trained model is then saved for future use.
- **Predicting noise:** The fifth step is to use the trained model to predict the noise from the mixed audio file, by applying the mask to the noisy spectrogram.
Removing noise from the input: The final step is to subtract the predicted noise from the mixed audio, resulting in a denoised or cleaned audio as the output.



App - https://audiodenoising.streamlit.app/

**Screenshot**

![image](https://github.com/aravindsriraj/Audio_Denoising/assets/60252521/65ab91d6-d339-48c8-a12a-3b02ae9728ef)

