"""End-to-end smoke test for Gemma 4 E4B audio pipeline on this A4000 box.

What this proves:
  1. Unsloth FastModel loads the 4-bit Gemma 4 E4B checkpoint into VRAM
  2. The AutoProcessor handles Gemma 4's audio chat template correctly
  3. An actual WAV (16 kHz mono from UASpeech) flows through audio encoder -> LM -> text
  4. We measure peak VRAM so we know our headroom on this GPU

We pick one control speaker clip (clean speech, should decode reasonably) and
three dysarthric speaker clips (atypical, may decode poorly — that's fine,
the point is the pipeline runs).

Run:
    .venv/Scripts/python.exe scripts/verify_gemma4_audio.py
"""
from __future__ import annotations
import os, sys, time, json

os.environ.setdefault("HF_HOME", r"D:\Downloads\huggingface_cache")
os.environ.setdefault("HF_HUB_CACHE", r"D:\Downloads\huggingface_cache\hub")
# torch 2.8 + triton-windows 3.6 on Windows has a missing `triton_key` symbol
# that breaks torch.compile/inductor. Disable dynamo for inference — eager is plenty fast.
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

import torch
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True
from unsloth import FastModel
from transformers import AutoProcessor
import soundfile as sf

MODEL_ID = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
MAX_NEW_TOKENS = 64

# A hand-picked mix of clips (all 16 kHz mono, each ~1–5s)
SAMPLES = [
    (r"D:\Downloads\speech_data\control\CF02\CF02_B1_D1_M2.wav",  "control (should be clean)"),
    (r"D:\Downloads\speech_data\control\CM01\CM01_B1_D1_M2.wav",  "control (should be clean)"),
    (r"D:\Downloads\speech_data\F03\F03_B1_D1_M2.wav",            "dysarthric (atypical)"),
    (r"D:\Downloads\speech_data\M08\M08_B1_D1_M2.wav",            "dysarthric (atypical)"),
]

PROMPT = "Transcribe the audio. Output only the spoken words."


def main():
    print(f"[{time.strftime('%H:%M:%S')}] loading {MODEL_ID} ...")
    t0 = time.time()
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=2048,
        load_in_4bit=True,
        full_finetuning=False,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"  loaded in {time.time()-t0:.1f}s")
    print(f"  model device: {next(model.parameters()).device}")
    print(f"  VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB "
          f"(peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB)")

    model.eval()

    results = []
    for path, label in SAMPLES:
        if not os.path.exists(path):
            print(f"  [SKIP] {path} (missing)")
            continue
        audio, sr = sf.read(path)
        if sr != 16000:
            print(f"  [SKIP] {path} (sr={sr}, expected 16000)")
            continue

        messages = [
            {"role": "user", "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": PROMPT},
            ]},
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        t1 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        dur = time.time() - t1

        # Decode only the newly generated portion
        input_len = inputs["input_ids"].shape[1]
        gen_ids = out[0, input_len:]
        text = processor.decode(gen_ids, skip_special_tokens=True).strip()

        results.append({
            "path": os.path.basename(path),
            "label": label,
            "duration_s": len(audio) / sr,
            "gen_time_s": round(dur, 2),
            "output": text,
        })
        print(f"\n-- {os.path.basename(path)} [{label}] "
              f"audio={len(audio)/sr:.2f}s  gen={dur:.2f}s --")
        print(f"   {text!r}")

    print(f"\n[{time.strftime('%H:%M:%S')}] done.")
    print(f"Final VRAM: allocated={torch.cuda.memory_allocated()/1e9:.2f} GB  "
          f"peak={torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # Save to JSON for later inspection
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/verify_gemma4_audio.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL_ID,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "peak_vram_gb": round(torch.cuda.max_memory_allocated()/1e9, 2),
            "results": results,
        }, f, indent=2)


if __name__ == "__main__":
    main()
