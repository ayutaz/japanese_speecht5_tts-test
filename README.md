# japanese_speecht5_tts-test

[esnya/japanese_speecht5_tts](https://huggingface.co/esnya/japanese_speecht5_tts)を動かすコード

# env
- python 3.11
- macOS

# setup
```bash
uv venv -p 3.11
source venv/bin/activate
```

# install
```bash
uv pip install -r requirements.txt
```

# convert to onnx(powershell for windows)
```bash
$kwargs='{"vocoder":"microsoft/speecht5_hifigan"}'; optimum-cli export onnx --model esnya/japanese_speecht5_tts --task text-to-audio --model-kwargs $kwargs --trust-remote-code japanese_speecht5_onnx
```