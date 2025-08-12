#!/usr/bin/env python3
"""Voice chat using open-source models.

This script connects speech recognition, an open-source language model,
and text-to-speech so you can talk with your computer.

Components:
- Whisper for speech-to-text.
- DialoGPT-small for conversational responses.
- pyttsx3 for text-to-speech output.
"""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import pyttsx3
import sounddevice as sd
import torch
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer


def record_audio(duration: int, fs: int = 16000) -> np.ndarray:
    """Record audio from the default microphone."""
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Talk with an open-source LLM using your voice."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Seconds to record for each user turn.",
    )
    args = parser.parse_args()

    stt_model = whisper.load_model("base")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    lm_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    tts_engine = pyttsx3.init()

    chat_history_ids: Optional[torch.Tensor] = None
    print("Press Enter, speak, and wait for a response. Say 'quit' or '終了' to exit.")

    while True:
        input("Ready? Press Enter and start speaking...")

        audio = record_audio(args.duration)
        result = stt_model.transcribe(audio, language="ja")
        user_text = result["text"].strip()
        print(f"User: {user_text}")
        if user_text.lower() in {"quit", "exit", "終了"}:
            break

        new_input_ids = tokenizer.encode(
            user_text + tokenizer.eos_token, return_tensors="pt"
        )
        bot_input_ids = (
            torch.cat([chat_history_ids, new_input_ids], dim=-1)
            if chat_history_ids is not None
            else new_input_ids
        )
        chat_history_ids = lm_model.generate(
            bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id
        )
        response_ids = chat_history_ids[:, bot_input_ids.shape[-1] :]
        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        print(f"Bot: {response}")
        tts_engine.say(response)
        tts_engine.runAndWait()

    print("Chat ended.")


if __name__ == "__main__":
    main()
