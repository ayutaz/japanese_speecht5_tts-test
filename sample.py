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

# CUDA > MPS > CPU の順で利用可能なデバイスを選択
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

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
    # SpeechT5ForTextToSpeech を float32 でロードし、選択した device に移動
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
    vocoder.to("cpu")  # CPU に移動
    vocoder_wrapper = VocoderCPUWrapper(vocoder)

    input_text = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。"
    # processor により input_ids を生成し、選択した device に転送
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
