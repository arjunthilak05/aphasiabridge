"""Show how Gemma 4 actually tokenizes an audio-chat sample so we can pick
the right response-start marker for loss masking."""
import os
os.environ.setdefault("HF_HOME", r"D:\Downloads\huggingface_cache")
os.environ.setdefault("HF_HUB_CACHE", r"D:\Downloads\huggingface_cache\hub")
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from transformers import AutoProcessor
import numpy as np

MODEL_ID = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
proc = AutoProcessor.from_pretrained(MODEL_ID)
tok = proc.tokenizer

# Try each way of getting the marker
for variant in ["<start_of_turn>model\n", "<start_of_turn>model", "<start_of_turn>", "model\n"]:
    ids = tok(variant, add_special_tokens=False).input_ids
    print(f"{variant!r:32s} -> {ids}  ({[tok.decode([i]) for i in ids]})")

print("\n--- apply_chat_template output (no audio) ---")
msgs = [
    {"role":"system","content":[{"type":"text","text":"sys"}]},
    {"role":"user","content":[{"type":"text","text":"Q"}]},
    {"role":"assistant","content":[{"type":"text","text":"A"}]},
]
res = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, return_tensors=None)
# res might be list of ints or a BatchEncoding; normalize
if hasattr(res, "input_ids"):
    ids = res.input_ids if isinstance(res.input_ids, list) else res.input_ids[0].tolist()
elif isinstance(res, list):
    ids = res if isinstance(res[0], int) else res[0]
else:
    ids = list(res)
print("ids:", ids)
for idx, tid in enumerate(ids):
    print(f"  [{idx:3d}]  {tid:6d}  {tok.decode([tid])!r}")

print("\n--- apply_chat_template with AUDIO placeholder ---")
dummy_audio = np.zeros(16000, dtype=np.float32)  # 1s silence
msgs2 = [
    {"role":"system","content":[{"type":"text","text":"sys"}]},
    {"role":"user","content":[
        {"type":"audio","audio":dummy_audio},
        {"type":"text","text":"Q"},
    ]},
    {"role":"assistant","content":[{"type":"text","text":"A"}]},
]
out = proc.apply_chat_template(msgs2, tokenize=True, return_dict=True, return_tensors="pt",
                                add_generation_prompt=False)
ids2 = out["input_ids"][0].tolist()
print(f"total tokens: {len(ids2)}  (has input_features: {'input_features' in out})")
# Show token ids where role transitions happen — look for any 'model'/'user'/<start_of_turn> tokens
suspects = []
for idx, tid in enumerate(ids2):
    s = tok.decode([tid])
    if any(k in s for k in ["start_of_turn", "model", "user", "end_of_turn", "\n"]) or tid < 100 or tid > 262000:
        suspects.append((idx, tid, s))
print("first 40 tokens:")
for idx, tid, s in suspects[:40]:
    print(f"  [{idx:4d}]  {tid:6d}  {s!r}")
