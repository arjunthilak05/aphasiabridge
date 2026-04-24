"""Quick diagnostic: load LoRA via FastModel (same as training path) and run a few samples."""
import os, sys, json
os.environ.setdefault("HF_HOME", r"D:\Downloads\huggingface_cache")
os.environ.setdefault("HF_HUB_CACHE", r"D:\Downloads\huggingface_cache\hub")
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
torch._dynamo.config.disable = True
from unsloth import FastModel
import soundfile as sf

MODEL_ID  = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
LORA_DIR  = sys.argv[1] if len(sys.argv) > 1 else r"outputs/aphasiabank_lora_r32_25k/lora_final"
VAL       = r"D:\Downloads\aphasiabank\aphasiabank_val_500.jsonl"
N_SAMPLES = 6

# Load model, THEN load LoRA adapter the way FastModel does
model, processor = FastModel.from_pretrained(
    model_name=LORA_DIR,                  # load the saved checkpoint directly
    dtype=None,
    max_seq_length=4096,
    load_in_4bit=True,
    full_finetuning=False,
)
model.eval()
print(f"[loaded] {LORA_DIR}")

rows = [json.loads(l) for l in open(VAL, encoding="utf-8")][:N_SAMPLES]

SYSTEM_PROMPT = "You are an assistant that transcribes speech accurately."
USER_PROMPT   = "Please transcribe this audio."

for r in rows:
    audio, sr = sf.read(r["audio_path"])
    assert sr == 16000
    msgs = [
        {"role":"system","content":[{"type":"text","text":SYSTEM_PROMPT}]},
        {"role":"user","content":[
            {"type":"audio","audio":audio},
            {"type":"text","text":USER_PROMPT},
        ]},
    ]
    inputs = processor.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v,"to") else v for k,v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                             temperature=None, top_p=None)
    text = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    print("-"*72)
    print(f"SEV:  {r.get('severity')}  CORPUS: {r.get('corpus')}  DUR: {r['duration_ms']}ms")
    print(f"REF:  {r['transcript']!r}")
    print(f"HYP:  {text!r}")
    print(f"HYP len: {len(text.split())} words  |  REF len: {len(r['transcript'].split())}")
