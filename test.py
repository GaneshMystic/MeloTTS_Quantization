from flask import Flask, send_file, request, render_template
from melo.api import TTS
import torch

app = Flask(__name__)

# CPU is sufficient for real-time inference.
# You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
device = "cuda"  # Will automatically use GPU if available
model = TTS(language="EN", device=device)
speaker_ids = model.hps.data.spk2id


@app.route("/tts", methods=["GET"])
def generate_tts():
    # Get the text message and speed parameters from the query parameters
    text_message = request.args.get("text")
    speed = float(
        request.args.get("speed", 1.0)
    )  # Default speed is 1.0 if not provided
    print(text_message)

    # English
    text = text_message
    # American accent - save the file locally before sending it
    output_path = "en-us.wav"
    model.tts_to_file(text, speaker_ids["EN-US"], output_path, speed=speed)

    # Return the audio file
    return send_file("en-us.wav", mimetype="audio/wav")


if __name__ == "__main__":
    app.run(debug=True, port=5002)
