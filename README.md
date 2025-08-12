# toritori

Sample project demonstrating voice chat powered by open-source models.

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Run the voice chat loop:

```bash
python voice_chat.py
```

Press Enter to start recording, speak for a few seconds, and wait for the model to reply.  
Say `quit` or `終了` to end the conversation.

### Models

- [Whisper](https://github.com/openai/whisper) for speech-to-text.
- [DialoGPT-small](https://huggingface.co/microsoft/DialoGPT-small) for language generation.
- [pyttsx3](https://pyttsx3.readthedocs.io/) for text-to-speech output.
