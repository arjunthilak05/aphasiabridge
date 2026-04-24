"""Print the actual Linear module names inside the Gemma 4 E4B audio encoder.
Use the output to validate the `target_modules` list in train_gemma4_lora.py.
"""
import os
os.environ.setdefault("HF_HOME", r"D:\Downloads\huggingface_cache")
os.environ.setdefault("HF_HUB_CACHE", r"D:\Downloads\huggingface_cache\hub")
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
torch._dynamo.config.disable = True
from unsloth import FastModel
import torch.nn as nn
from collections import Counter

MODEL_ID = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
model, processor = FastModel.from_pretrained(
    model_name=MODEL_ID, dtype=None, max_seq_length=2048,
    load_in_4bit=True, full_finetuning=False,
)

print("\n=== all module subtrees containing 'audio' ===")
audio_paths = sorted({n for n, _ in model.named_modules() if "audio" in n.lower()})
for p in audio_paths[:15]:
    print(f"  {p}")
print(f"  ...total {len(audio_paths)}")

print("\n=== unique last-segment names under audio subtree ===")
last_segs = Counter()
for n, m in model.named_modules():
    if "audio" not in n.lower():
        continue
    if not isinstance(m, nn.Linear):
        continue
    seg = n.rsplit(".", 1)[-1]
    last_segs[seg] += 1
for seg, c in last_segs.most_common():
    print(f"  {seg:35s}  count={c}")

print("\n=== current target_modules in train_gemma4_lora.py ===")
CURRENT = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "post", "linear_start", "linear_end",
    "embedding_projection",
    "ffw_layer_1", "ffw_layer_2",
    "output_proj",
]
for m in CURRENT:
    in_audio = last_segs.get(m, 0)
    print(f"  {m:25s}  audio-count={in_audio}  {'OK' if in_audio else 'NOT FOUND IN AUDIO'}")

# Also check if any appear in LANGUAGE (non-audio) subtree
print("\n=== language-tower Linear last-seg names (top 20) ===")
lang_segs = Counter()
for n, m in model.named_modules():
    if "audio" in n.lower(): continue
    if isinstance(m, nn.Linear):
        lang_segs[n.rsplit('.',1)[-1]] += 1
for seg, c in lang_segs.most_common(20):
    print(f"  {seg:35s}  count={c}")
