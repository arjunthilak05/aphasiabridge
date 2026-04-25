"""Wait for all 4 reeval_summary.json files + Whisper baseline, generate
the Phase-1 results section, append to H100_RESULTS.md, and git-commit."""
from __future__ import annotations
import json, os, time, subprocess, sys
from pathlib import Path

ROOT = Path("/data/kl_dev/dataset_wan/data_gemma/aphasiabridge")

REEVALS = {
    "R1-D3":  ROOT / "outputs/h100_lora_daniel_v3/reeval_summary.json",
    "R1-D4a": ROOT / "outputs/h100_lora_daniel_v4/reeval_summary.json",
    "R1-D4b": ROOT / "outputs/h100_lora_daniel_v4b/reeval_summary.json",
    "R1-D4c": ROOT / "outputs/h100_lora_daniel_v4c/reeval_summary.json",
}
WHISPER = ROOT / "results/whisper_baseline/whisper_baseline.json"
DOC     = ROOT / "H100_RESULTS.md"

def wait_for_all():
    while True:
        missing = [tag for tag, p in REEVALS.items() if not p.exists()]
        if not missing:
            return
        print(f"[{time.strftime('%H:%M:%S')}] still waiting on: {', '.join(missing)}", flush=True)
        time.sleep(60)

def load_reeval(p: Path):
    with open(p) as f:
        return json.load(f)

def render_markdown(reevals: dict, whisper: dict) -> str:
    lines = []
    lines.append("## Q. Phase 1 — generation-time fix re-eval (no retraining)")
    lines.append("")
    lines.append("Hypothesis tested: the repeat-loop attractor is a *decoding* failure as much as a")
    lines.append("training failure (same as Whisper's classic problem). Standard fix:")
    lines.append("`repetition_penalty=1.3 + no_repeat_ngram_size=4 + max_new_tokens=200 + beam=1`.")
    lines.append("Re-evaluated all 4 saved LoRAs with the new generate kwargs — no retraining.")
    lines.append("")
    lines.append("### Re-eval scoreboard (all on the same 220-clip speaker-disjoint ss_test)")
    lines.append("")
    lines.append("| Run | OLD WER | **NEW WER** | OLD repeat-loops | **NEW repeat-loops** | OLD hallucinations | **NEW hallucinations** |")
    lines.append("|---|---|---|---|---|---|---|")
    OLD = {  # from prior run summaries (already in this doc)
        "R1-D3":  ("2.093", "15", "9"),
        "R1-D4a": ("0.802", "?",  "?"),
        "R1-D4b": ("0.723", "?",  "?"),
        "R1-D4c": ("0.734", "3",  "2"),
    }
    for tag in ("R1-D3", "R1-D4a", "R1-D4b", "R1-D4c"):
        s = reevals[tag]["result"]
        a = s["after"]
        fm = s["after_failure_modes"]
        old_wer, old_rep, old_hal = OLD[tag]
        lines.append(
            f"| {tag} | {old_wer} | **{a['overall_wer']}** | {old_rep} | **{fm['repeat_loops']}** | {old_hal} | **{fm['protocol_hallucinations']}** |"
        )
    lines.append("")
    # BASE numbers (from any reeval — should be very similar across runs)
    base_wer = reevals["R1-D4b"]["result"].get("base", {}).get("overall_wer")
    if base_wer is not None:
        lines.append(f"**BASE Gemma 4 E4B with new gen kwargs:** {base_wer} WER (vs 0.556 with old gen kwargs).")
    lines.append("")

    lines.append("### Per-severity AFTER (new gen kwargs)")
    lines.append("")
    lines.append("| Severity | BASE | R1-D3 | R1-D4a | R1-D4b | R1-D4c |")
    lines.append("|---|---|---|---|---|---|")
    sevs = sorted(set(
        s for r in reevals.values() for s in r["result"]["after"]["per_severity_wer"]
    ))
    for sev in sevs:
        bs = reevals["R1-D4b"]["result"].get("base", {}).get("per_severity_wer", {}).get(sev, "–")
        row = [str(bs)]
        for tag in ("R1-D3", "R1-D4a", "R1-D4b", "R1-D4c"):
            v = reevals[tag]["result"]["after"]["per_severity_wer"].get(sev, "–")
            row.append(str(v))
        lines.append(f"| {sev} | " + " | ".join(row) + " |")
    lines.append("")

    # Pick the best AFTER WER
    best_tag = min(reevals.keys(), key=lambda t: reevals[t]["result"]["after"]["overall_wer"] or 99)
    best_wer = reevals[best_tag]["result"]["after"]["overall_wer"]
    lines.append(f"### Winner with gen-time fix: **{best_tag}** at WER **{best_wer}**")
    lines.append("")

    # Whisper section
    lines.append("## R. Whisper-Large-v3 zero-shot baseline (architecture decision input)")
    lines.append("")
    w = whisper
    wer_lc = w["wer_lowercase"]; wer_n = w["wer_normalized"]
    lines.append(f"Run on the same 220-clip ss_test, beam=5 language=en, no preprocessing.")
    lines.append("")
    lines.append("| Metric | Whisper-Large-v3 (zero-shot) | BASE Gemma 4 E4B (zero-shot, lowercase) |")
    lines.append("|---|---|---|")
    lines.append(f"| Overall WER (lowercase) | **{wer_lc['wer']}** | **{base_wer if base_wer is not None else 0.556}** |")
    lines.append(f"| Overall CER (lowercase) | {wer_lc['cer']} | – |")
    lines.append(f"| Overall WER (Whisper EnglishNormalizer) | **{wer_n['wer']}** | – |")
    lines.append(f"| Errors | {w['n_errors']}/{w['n_clips']} | – |")
    lines.append("")
    lines.append("Per-severity WER (Whisper, normalized):")
    for sev, v in wer_n["per_severity_wer"].items():
        lines.append(f"- {sev}: {v}")
    lines.append("")
    lines.append(f"Paper reference (Jiang et al. 2026, B1 column): Whisper-Large-v3 = 39.97 % WER on their AphasiaBank test split. Our 220-clip ss_test is harder (different speakers, all `[*p:]` filtered).")
    lines.append("")
    lines.append("### Architecture decision")
    lines.append("")
    if wer_lc["wer"] > base_wer:
        lines.append(f"**Whisper-Large-v3 zero-shot ({wer_lc['wer']}) is WORSE than BASE Gemma 4 E4B ({base_wer}) on this test set.** CONTINGENCY.md Strategy A (Whisper-front-end + Gemma-intent) is OFF the table — using Whisper as the ASR layer would strictly degrade quality. **Ship path: continue with Gemma 4 audio LoRA refinements; BASE Gemma is genuinely strong on aphasic English.**")
    else:
        lines.append(f"Whisper-Large-v3 zero-shot ({wer_lc['wer']}) is better than BASE Gemma 4 E4B ({base_wer}). Strategy A worth considering: Whisper(ASR) → Gemma(intent recovery).")
    return "\n".join(lines) + "\n"

def main():
    print(f"[{time.strftime('%H:%M:%S')}] waiting for all 4 reevals + Whisper baseline ...", flush=True)
    wait_for_all()
    if not WHISPER.exists():
        print("Whisper baseline missing!", file=sys.stderr); sys.exit(1)

    print(f"[{time.strftime('%H:%M:%S')}] all results in. Loading ...", flush=True)
    reevals = {tag: load_reeval(p) for tag, p in REEVALS.items()}
    whisper = load_reeval(WHISPER)

    new_section = render_markdown(reevals, whisper)
    print("\n" + new_section + "\n", flush=True)

    with open(DOC, encoding="utf-8") as f:
        existing = f.read()
    # Append new sections (idempotent: skip if header already there)
    if "## Q. Phase 1 — generation-time fix re-eval" in existing:
        # remove old Q + R sections, append fresh
        idx = existing.index("## Q. Phase 1 — generation-time fix re-eval")
        existing = existing[:idx].rstrip() + "\n\n"
    with open(DOC, "w", encoding="utf-8") as f:
        f.write(existing + "\n" + new_section)

    # Stage + commit
    subprocess.run(["git", "add", str(DOC)], cwd=str(ROOT), check=True)
    msg = (f"Phase-1 re-eval results: gen-time fix (rep_penalty + no_repeat_ngram) on all 4 LoRAs "
           f"+ Whisper-Large-v3 zero-shot baseline on same 220 clips\n\n"
           f"Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>")
    subprocess.run(["git", "commit", "-m", msg], cwd=str(ROOT), check=True)
    print(f"\n[{time.strftime('%H:%M:%S')}] FINALIZE_DONE — committed.", flush=True)


if __name__ == "__main__":
    main()
