import torch
import os
from . import utils
# from cached_path import cached_path
# from huggingface_hub import hf_hub_download

DOWNLOAD_CKPT_URLS = {
    "EN": "C:/Users/MYSTIC Ganesh/.cache/huggingface/hub/models--myshell-ai--MeloTTS-English/snapshots/0f6ecee6633a651c0d6e6629274e6347266e024d/checkpoint.pth",
    "EN_V3": "C:/Users/MYSTIC Ganesh/.cache/huggingface/hub/models--myshell-ai--MeloTTS-English-v3/snapshots/f7c4a35392c0e9be24a755f1edb4c3f63040f759/checkpoint.pth",
}

DOWNLOAD_CONFIG_URLS = {
    "EN": "C:/Users/MYSTIC Ganesh/.cache/huggingface/hub/models--myshell-ai--MeloTTS-English/snapshots/0f6ecee6633a651c0d6e6629274e6347266e024d/config.json",
    "EN_V3": "C:/Users/MYSTIC Ganesh/.cache/huggingface/hub/models--myshell-ai--MeloTTS-English-v3/snapshots/f7c4a35392c0e9be24a755f1edb4c3f63040f759/config.json",
}

PRETRAINED_MODELS = {
    "G.pth": "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/pretrained/G.pth",
    "D.pth": "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/pretrained/D.pth",
    "DUR.pth": "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/pretrained/DUR.pth",
}

# LANG_TO_HF_REPO_ID = {
#     "EN": "myshell-ai/MeloTTS-English",
#     "EN_V3": "myshell-ai/MeloTTS-English-v3",
# }


def load_or_download_config(locale, use_hf=True, config_path=None):
    print(config_path)
    if config_path is None:
        language = locale.split("-")[0].upper()
        if use_hf:
        #     assert language in LANG_TO_HF_REPO_ID
        #     config_path = hf_hub_download(
        #         repo_id=LANG_TO_HF_REPO_ID[language], filename="config.json"
        #     )
        # else:
            assert language in DOWNLOAD_CONFIG_URLS
            config_path = DOWNLOAD_CONFIG_URLS[language]
    return utils.get_hparams_from_file(config_path)


def load_or_download_model(locale, device, use_hf=True, ckpt_path=None):
    print(ckpt_path)
    if ckpt_path is None:
        language = locale.split("-")[0].upper()
        if use_hf:
        #     assert language in LANG_TO_HF_REPO_ID
        #     ckpt_path = hf_hub_download(
        #         repo_id=LANG_TO_HF_REPO_ID[language], filename="checkpoint.pth"
        #     )
        # else:
            assert language in DOWNLOAD_CKPT_URLS
            ckpt_path = DOWNLOAD_CKPT_URLS[language]

    return torch.load(ckpt_path, map_location=device)


def load_pretrain_model():
    return [url for url in PRETRAINED_MODELS.values()]
    # return [cached_path(url) for url in PRETRAINED_MODELS.values()]
