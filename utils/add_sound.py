""" Handles providing a voice to the generation/text using SARVAM's text-to-speech api"""
import os
import requests
import base64
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


def play_sound(text):
    """
    Sends a text input to the Sarvam API for text-to-speech conversion, receives the audio response, 
    and plays it if the request is successful.


    :param text: The input text to be converted to speech.
    :type text: str
    :return: None
    :rtype: None
    """

# Sarvam API endpoint and request details
    url = "https://api.sarvam.ai/text-to-speech"
    payload = {
        "inputs": [text],
        "target_language_code": "en-IN",
        "enable_preprocessing": True,
        "pace": 1
    }

    headers = {
        "api-subscription-key": os.getenv("SARVAM_API_KEY"),
        "Content-Type": "application/json",
    }

    def is_docker():
        """Check if the code is running inside a Docker container."""
        path = '/.dockerenv'
        return os.path.exists(path)

    def play_audio_if_possible(audio_segment):
        if not is_docker():
            play(audio_segment)
        else:
            print("Skipping audio playback inside Docker.")

    # Request to Sarvam API
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        response_data = response.json()

        # Check if 'audios' field is present in the response
        if "audios" in response_data and len(response_data["audios"]) > 0:
            base64_string = response_data["audios"][0]

            # Convert Base64 string to bytes
            wav_bytes = base64.b64decode(base64_string)

            # Use BytesIO to handle the audio in memory
            audio_io = BytesIO(wav_bytes)

            # Play the audio directly from memory
            sound = AudioSegment.from_wav(audio_io)
            play_audio_if_possible(sound)

        else:
            print("Error: No audio generated in the response.")
    else:
        print(f"Error: {response.status_code}, {response.text}")

# play_sound(" How are you doing ?")