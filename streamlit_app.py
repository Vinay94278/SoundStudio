import streamlit as st
import requests

st.title("Audio Denoising")

uploaded_file = st.file_uploader("Choose an audio file", type=['wav'])

if uploaded_file is not None:
    file_details = {"File Name": uploaded_file.name, "File Type": uploaded_file.type, "File Size": uploaded_file.size}
    st.write(file_details)

    if file_details['File Type'] == 'audio/wav':
        st.subheader("Input")
        st.audio(uploaded_file.read())

        with st.spinner("Denoising the audio..."):
            response = requests.post("http://127.0.0.1:5000/denoise", files={'file': uploaded_file})

            if response.status_code == 200:
                st.success("Denoised!!")
                st.subheader("Output Audio")
                st.audio(response.content)
            else:
                st.error("Failed to denoise audio")