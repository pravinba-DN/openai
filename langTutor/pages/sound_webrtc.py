import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import numpy as np
import wave

# Audio processor to handle audio frames
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []
        self.recording = False

    def recv(self, frame):
        if self.recording:
            self.audio_frames.append(frame.to_ndarray())
        return frame

    def start_recording(self):
        self.audio_frames = []  # Reset frames
        self.recording = True

    def stop_recording(self):
        self.recording = False
        return self.audio_frames

def app():
    st.title("Audio Recorder")

    audio_processor = AudioProcessor()

    # Start recording button
    if st.button("Start Recording"):
        audio_processor.start_recording()
        st.write("Recording...")

    # Stop recording button
    if st.button("Stop Recording"):
        frames = audio_processor.stop_recording()
        if frames:
            st.write("Stopped Recording.")
            save_audio(frames)
        else:
            st.write("No audio recorded.")

    # Streamlit WebRTC component
    try:
        webrtc_streamer(key="audio", audio_processor_factory=lambda: audio_processor, media_stream_constraints={"audio": True})
    except Exception as e:
        st.error(f"Error initializing WebRTC: {e}")

def save_audio(frames):
    try:
        audio_file = "recorded_audio.wav"
        
        with wave.open(audio_file, 'wb') as wf:
            wf.setnchannels(1)  # Mono audio
            wf.setsampwidth(2)  # 16 bits
            wf.setframerate(16000)  # Sample rate
            for frame in frames:
                wf.writeframes(frame.tobytes())
        
        st.success(f"Recording saved as {audio_file}.")
        st.audio(audio_file)
    except Exception as e:
        st.error(f"Error saving audio: {e}")

