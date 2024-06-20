from flask import Flask, request
from flask import Flask, send_file, request, render_template
from melo.api import TTS
from datetime import datetime

app = Flask(__name__)
device = "cuda" 
model = TTS(language="EN", device=device)
speaker_ids = model.hps.data.spk2id


@app.route("/tts", methods=["GET"])
def generate_tts():
    input_text = request.args.get("text")
    speed = float(
        request.args.get("speed", 1.0)
    )  
    
    audio = model.tts_to_file(
        text=input_text, speaker_id=speaker_ids['EN-US'],  speed=speed, buffer=True)
    print(audio)
    return send_file(
        audio,
        mimetype='audio/wav',
        as_attachment=True,
        download_name='audio.wav',
        conditional=True,
        etag=True,
        last_modified=datetime.now(),
        max_age=0
    )


if __name__ == "__main__":
    app.run(port=5002)
