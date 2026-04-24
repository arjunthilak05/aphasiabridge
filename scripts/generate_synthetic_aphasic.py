"""Generate synthetic aphasic utterances by perturbing clean transcripts.

Models four documented aphasia error patterns:
  1. Phonemic paraphasias      — sound-level substitutions ("spoon" -> "stoon")
  2. Semantic paraphasias      — within-category word swaps ("cat" -> "dog")
  3. Agrammatism               — drop function words / tense markers
  4. Word-finding pauses       — insert filler tokens ("the... the... you know")

Output schema (matches align_aprocsa.py manifest so the two corpora can be mixed):
  {source: "synthetic", condition: "synthetic-aphasia", severity: <mild|moderate|severe>,
   transcript_clean: <perturbed>, transcript_raw: <same>, target_intent: <original clean text>,
   paraphasia_count, dropped_word_count, filler_count, word_count}

Usage:
    python scripts/generate_synthetic_aphasic.py \
        --input  data/clean_transcripts.txt \
        --out    data/synthetic_aphasic.jsonl \
        --per-input 3 \
        --seed 42
"""
from __future__ import annotations
import argparse, json, os, random, re, sys
from typing import List, Tuple

FILLERS = ["uh", "um", "er", "mm"]
WORD_FIND_PHRASES = ["the... the...", "you know...", "it's... it's...", "what's the word"]

# Minimal function-word list for agrammatism (drop rate depends on severity)
FUNCTION_WORDS = {
    "a", "an", "the", "is", "was", "were", "are", "am", "be", "been", "being",
    "of", "to", "in", "on", "at", "for", "with", "by", "from", "as",
    "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
    "do", "does", "did", "have", "has", "had", "will", "would", "could", "should",
    "can", "may", "might", "must", "shall",
    "my", "your", "his", "her", "its", "our", "their",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "us", "them",
}

# Simple phonemic substitution classes — voiced<->voiceless and same-place swaps.
# Applied on the spelling level as a cheap surrogate for true phonemic changes.
PHONEMIC_MAP = [
    ("p", "b"), ("b", "p"), ("t", "d"), ("d", "t"), ("k", "g"), ("g", "k"),
    ("f", "v"), ("v", "f"), ("s", "z"), ("z", "s"),
    ("sh", "s"), ("ch", "sh"), ("th", "t"),
    ("m", "n"), ("n", "m"), ("l", "r"), ("r", "l"),
]

# Coarse semantic substitution buckets — small illustrative set.
# Real pipeline should back this with WordNet or a larger lexicon.
SEMANTIC_BUCKETS = [
    ["cat", "dog", "bird", "fish", "rabbit", "horse"],
    ["car", "truck", "bike", "bus", "train", "plane"],
    ["kitchen", "bathroom", "bedroom", "living_room", "office"],
    ["mother", "father", "sister", "brother", "son", "daughter"],
    ["coffee", "tea", "water", "juice", "milk", "soda"],
    ["walk", "run", "jump", "sit", "stand", "lie"],
    ["red", "blue", "green", "yellow", "black", "white"],
    ["happy", "sad", "angry", "tired", "excited", "worried"],
    ["morning", "afternoon", "evening", "night"],
]
SEMANTIC_LOOKUP = {w: bucket for bucket in SEMANTIC_BUCKETS for w in bucket}

# Severity profiles — probabilities per word
SEVERITY_PROFILES = {
    "mild":     dict(p_paraphasia_phon=0.03, p_paraphasia_sem=0.03, p_drop_funcword=0.10, p_filler_before=0.05, p_wordfind=0.02),
    "moderate": dict(p_paraphasia_phon=0.08, p_paraphasia_sem=0.06, p_drop_funcword=0.30, p_filler_before=0.15, p_wordfind=0.08),
    "severe":   dict(p_paraphasia_phon=0.18, p_paraphasia_sem=0.12, p_drop_funcword=0.55, p_filler_before=0.30, p_wordfind=0.20),
}


def phonemic_perturb(word: str, rng: random.Random) -> str:
    candidates = [(a, b) for a, b in PHONEMIC_MAP if a in word]
    if not candidates:
        return word
    a, b = rng.choice(candidates)
    # replace one occurrence at a random match site
    indices = [i for i in range(len(word)) if word[i:i+len(a)] == a]
    if not indices:
        return word
    i = rng.choice(indices)
    return word[:i] + b + word[i+len(a):]


def semantic_perturb(word: str, rng: random.Random) -> str:
    bucket = SEMANTIC_LOOKUP.get(word.lower())
    if not bucket:
        return word
    alts = [w for w in bucket if w != word.lower()]
    return rng.choice(alts) if alts else word


def perturb_utterance(text: str, severity: str, rng: random.Random) -> dict:
    prof = SEVERITY_PROFILES[severity]
    tokens = re.findall(r"[A-Za-z']+|[.,!?;]", text)
    out = []
    para_count = 0
    drop_count = 0
    filler_count = 0
    for tok in tokens:
        if not tok.isalpha():
            out.append(tok)
            continue
        lower = tok.lower()
        # Drop function words (agrammatism)
        if lower in FUNCTION_WORDS and rng.random() < prof["p_drop_funcword"]:
            drop_count += 1
            continue
        # Insert filler before content word
        if rng.random() < prof["p_filler_before"]:
            out.append(rng.choice(FILLERS))
            filler_count += 1
        # Word-finding pause
        if rng.random() < prof["p_wordfind"]:
            out.append(rng.choice(WORD_FIND_PHRASES))
            filler_count += 1
        # Paraphasia
        r = rng.random()
        if r < prof["p_paraphasia_phon"]:
            tok2 = phonemic_perturb(tok, rng)
            if tok2 != tok:
                para_count += 1
            out.append(tok2)
        elif r < prof["p_paraphasia_phon"] + prof["p_paraphasia_sem"]:
            tok2 = semantic_perturb(tok, rng)
            if tok2 != tok:
                para_count += 1
            out.append(tok2)
        else:
            out.append(tok)
    # Re-stitch with spaces but keep punctuation attached
    s = ""
    for tok in out:
        if re.match(r"[.,!?;]", tok):
            s = s.rstrip() + tok + " "
        else:
            s += tok + " "
    return {
        "text": s.strip(),
        "paraphasia_count": para_count,
        "dropped_word_count": drop_count,
        "filler_count": filler_count,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Text file, one clean utterance per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-input", type=int, default=3, help="Perturbations per input line (distributed across severities)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    n = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for i, line in enumerate(lines):
            severities = (["mild", "moderate", "severe"] * args.per_input)[:args.per_input]
            rng.shuffle(severities)
            for sev in severities:
                p = perturb_utterance(line, sev, rng)
                if not p["text"]:
                    continue
                rec = {
                    "source": "synthetic",
                    "condition": "synthetic-aphasia",
                    "severity": sev,
                    "source_index": i,
                    "transcript_clean": p["text"],
                    "transcript_raw": p["text"],
                    "target_intent": line,
                    "paraphasia_count": p["paraphasia_count"],
                    "dropped_word_count": p["dropped_word_count"],
                    "filler_count": p["filler_count"],
                    "word_count": len(p["text"].split()),
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1

    print(f"Wrote {n} synthetic utterances to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
