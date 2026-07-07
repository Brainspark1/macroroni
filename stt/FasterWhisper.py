import torch
import speech_recognition as sr
import io
import os
import time

from faster_whisper import WhisperModel

device = "cuda" if torch.cuda.is_available() else "cpu"

compute_type = (
    "float16"
    if torch.cuda.is_available()
    else "int8"
)

whisper_model = WhisperModel(
    "base",
    device=device,
    compute_type=compute_type
)

recognizer = sr.Recognizer()
microphone = sr.Microphone(sample_rate=16000)

with microphone as source:

    print("Adjusting for background noise")
    recognizer.adjust_for_ambient_noise(source, duration=7)
    print("Completed adjusting for background noise")


print("Start speaking into your microphone. Press Ctrl+C to stop.\n")

def audio_callback(recognizer, audio):
    try:
        wav_data = io.BytesIO(
            audio.get_wav_data()
        )

        with open("temp_chunk.wav", "wb") as f:
            f.write(wav_data.read())

        segments, info = whisper_model.transcribe(
            "temp_chunk.wav",
            beam_size=1, # number of possible transcriptions considered before choosing next word - lower value = lower computation time/latency
            vad_filter=True,
            condition_on_previous_text=False
        )

        text = ""

        for segment in segments:
            text += segment.text # adding to transcription text the text gathered from each segment recorded by the whisper model

        text = text.strip()

        if text:
            print(f"Transcription: {text}")

        if os.path.exists("temp_chunk.wav"):
            os.remove("temp_chunk.wav")

    except Exception as e:
        print(f"Error during transcription segment: {e}")

stop_listening = recognizer.listen_in_background(
    microphone,
    audio_callback,
    phrase_time_limit=3 # still forced to set how long chunk of time records
)

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nStopping live transcription.")
    stop_listening(wait_for_stop=False)