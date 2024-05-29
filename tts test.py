
from melo.api import TTS
import torch


device = "cuda"  # Will automatically use GPU if available
model = TTS(language="EN_V3", device=device)
speaker_ids = model.hps.data.spk2id


text_message = "Something Interesting"
speed = 1.0
print(text_message)

text = text_message

# American accent - save the file locally before sending it
output_path = "en-us.wav"
model.tts_to_file(text, speaker_ids["EN-US"], output_path, speed=speed)
