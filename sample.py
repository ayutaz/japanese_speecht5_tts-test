import numpy as np
from transformers import (
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5FeatureExtractor,
    SpeechT5Processor,
)
from speecht5_openjtalk_tokenizer import SpeechT5OpenjtalkTokenizer
import soundfile
import torch

# MPS が利用可能なら "mps"、なければ "cpu" を使用（今回はモデル本体は MPS/CPU で動作）
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# vocoder を CPU で実行するためのラッパークラス
class VocoderCPUWrapper(torch.nn.Module):
    def __init__(self, vocoder):
        super().__init__()
        self.vocoder = vocoder  # この vocoder は全パラメータが CPU 上にある必要があります
    def forward(self, x):
        # 入力を CPU に移動してから vocoder を実行
        x_cpu = x.cpu()
        return self.vocoder(x_cpu)

model_name = "esnya/japanese_speecht5_tts"
with torch.no_grad():
    # SpeechT5ForTextToSpeech を float32 でロードし、device に移動
    model = SpeechT5ForTextToSpeech.from_pretrained(
        model_name, torch_dtype=torch.float32
    )
    model.to(device)

    tokenizer = SpeechT5OpenjtalkTokenizer.from_pretrained(model_name)
    feature_extractor = SpeechT5FeatureExtractor.from_pretrained(model_name)
    processor = SpeechT5Processor(feature_extractor, tokenizer)
    
    # SpeechT5HifiGan (vocoder) は MPS の制限があるため、明示的に CPU にロード
    vocoder = SpeechT5HifiGan.from_pretrained(
        "microsoft/speecht5_hifigan", torch_dtype=torch.float32
    )
    vocoder.to("cpu")  # ここで CPU に移動
    vocoder_wrapper = VocoderCPUWrapper(vocoder)

    input_text = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。"
    # processor により input_ids を生成し、device に転送
    input_ids = processor(text=input_text, return_tensors="pt").input_ids.to(device)

    speaker_embeddings = np.random.uniform(-1, 1, (1, 16))
    speaker_embeddings = torch.FloatTensor(speaker_embeddings).to(device=device, dtype=model.dtype)

    # generate_speech 呼び出し時に vocoder_wrapper を指定
    waveform = model.generate_speech(
        input_ids,
        speaker_embeddings,
        vocoder=vocoder_wrapper,
    )

    waveform = waveform / waveform.abs().max()  # 正規化
    waveform = waveform.reshape(-1).cpu().float().numpy()

    soundfile.write(
        "output.wav",
        waveform,
        vocoder.config.sampling_rate,
    )
