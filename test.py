from flask import Flask, send_file, request
from melo.api import TTS


app = Flask(__name__)
device = "auto"  # Will automatically use GPU if available

# English
model = TTS(language="EN", device=device)
speaker_ids = model.hps.data.spk2id

# American accent - save the file locally before sending it
output_path = "en-us.wav"


@app.route("/tts", methods=["GET"])
def generate_tts():
    # Get the text message and speed parameters from the query parameters
    text_message = request.args.get("text")
    speed = float(
        request.args.get("speed", 1.0)
    )  # Default speed is 1.0 if not provided
    print(text_message)
    text = text_message
    # CPU is sufficient for real-time inference.
    # You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
    model.tts_to_file(text, speaker_ids["EN-US"], output_path, speed=speed)

    # Return the audio file
    return send_file("en-us.wav", mimetype="audio/wav")


if __name__ == "__main__":
    app.run(debug=True, port=5002)
