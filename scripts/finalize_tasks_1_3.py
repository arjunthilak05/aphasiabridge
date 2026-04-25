"""Wait for Task-1 (beam=5 reevals) + Task-3 (lang-only LoRA) and append results to H100_RESULTS.md."""
from __future__ import annotations
import json, time, subprocess
from pathlib import Path

ROOT = Path("/data/kl_dev/dataset_wan/data_gemma/aphasiabridge")
JOBS = {
    "beam5_d4a": ROOT / "outputs/h100_lora_daniel_v4_beam5/reeval_summary.json",
    "beam5_d4b": ROOT / "outputs/h100_lora_daniel_v4b_beam5/reeval_summary.json",
    "lang_only": ROOT / "outputs/h100_lora_lang_only/run_summary.json",
}
DOC = ROOT / "H100_RESULTS.md"


def wait_for_all():
    while True:
        missing = [k for k, p in JOBS.items() if not p.exists()]
        if not missing:
            return
        print(f"[{time.strftime('%H:%M:%S')}] waiting on: {', '.join(missing)}", flush=True)
        time.sleep(60)


def render(beam5_d4a, beam5_d4b, lang_only):
    lines = []
    lines.append("## S. Task 1 + Task 3 results (beam=5 + freeze-audio)")
    lines.append("")
    lines.append("### Task 1 — beam=5 decoding on top of rep_penalty + no_repeat_ngram")
    lines.append("")
    lines.append("Adds beam search 5 to the Phase-1 gen kwargs. Tests whether beam search closes more gap.")
    lines.append("")
    lines.append("| Run | beam=1 BASE | beam=1 AFTER | **beam=5 BASE** | **beam=5 AFTER** | Δ vs beam=5 BASE |")
    lines.append("|---|---|---|---|---|---|")
    for tag, run, b1_base, b1_after in [
        ("D4a", beam5_d4a, 0.6138, 0.5254),
        ("D4b", beam5_d4b, 0.6138, 0.5415),
    ]:
        b5_base = run["result"]["base"]["overall_wer"]
        b5_after = run["result"]["after"]["overall_wer"]
        delta = round(b5_after - b5_base, 4) if (b5_base and b5_after) else "–"
        lines.append(f"| {tag} | {b1_base} | {b1_after} | **{b5_base}** | **{b5_after}** | {delta} |")
    lines.append("")
    a_after = beam5_d4a["result"]["after"]["overall_wer"]
    b_after = beam5_d4b["result"]["after"]["overall_wer"]
    base = beam5_d4a["result"]["base"]["overall_wer"]
    lines.append(f"### Per-severity AFTER (beam=5)")
    lines.append("")
    lines.append("| Severity | BASE | D4a | D4b |")
    lines.append("|---|---|---|---|")
    sevs = sorted(set(list(beam5_d4a["result"]["after"]["per_severity_wer"]) +
                      list(beam5_d4b["result"]["after"]["per_severity_wer"])))
    for sev in sevs:
        bs = beam5_d4a["result"]["base"]["per_severity_wer"].get(sev, "–")
        a = beam5_d4a["result"]["after"]["per_severity_wer"].get(sev, "–")
        b = beam5_d4b["result"]["after"]["per_severity_wer"].get(sev, "–")
        lines.append(f"| {sev} | {bs} | {a} | {b} |")
    lines.append("")

    lines.append("### Task 3 — language-only LoRA (audio encoder frozen)")
    lines.append("")
    s = lang_only
    ho = s["holdout_eval"]
    lines.append("Smaller LoRA capacity (no audio modules) → lowest over-fitting risk on 3,956 clips.")
    lines.append(f"`target_modules` = `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]` only.")
    lines.append(f"`modules_to_save` = `[lm_head, embed_tokens]`. Audio encoder weights untouched.")
    lines.append("")
    lines.append("```")
    lines.append(f"records: {s['records']}, lora_r/α: {s['lora_r']}/{s['lora_alpha']}, lr: {s['lr']}")
    lines.append(f"elapsed: {s['elapsed_s']/60:.1f} min, peak VRAM: {s['peak_vram_gb']:.2f} GB")
    lines.append(f"final train loss: {s['train_loss']:.3f}")
    if "base" in ho:
        lines.append(f"WER  BASE: {ho['base']['overall_wer']}  AFTER: {ho['after']['overall_wer']}  Δ: {round(ho['after']['overall_wer']-ho['base']['overall_wer'],4)}")
        lines.append(f"per-severity BASE:  {ho['base']['per_severity_wer']}")
    lines.append(f"per-severity AFTER: {ho['after']['per_severity_wer']}")
    lines.append("```")
    lines.append("")

    # Final scoreboard
    lines.append("### Final scoreboard (lower = better)")
    lines.append("")
    lines.append(f"```")
    lines.append(f"BASE Gemma 4 (beam=5 + rep_penalty 1.3 + no_repeat_ngram 4)  = {base}")
    lines.append(f"D4a (beam=5)                                                = {a_after}")
    lines.append(f"D4b (beam=5)                                                = {b_after}")
    if "base" in ho:
        lines.append(f"Lang-only LoRA (beam=1, gen kwargs match training script)   = {ho['after']['overall_wer']}")
    lines.append(f"```")

    return "\n".join(lines) + "\n"


def main():
    print(f"[{time.strftime('%H:%M:%S')}] waiting for Task-1 + Task-3 ...", flush=True)
    wait_for_all()
    data = {k: json.loads(Path(p).read_text()) for k, p in JOBS.items()}

    section = render(data["beam5_d4a"], data["beam5_d4b"], data["lang_only"])
    print("\n" + section + "\n", flush=True)

    existing = DOC.read_text()
    if "## S. Task 1 + Task 3 results" in existing:
        idx = existing.index("## S. Task 1 + Task 3 results")
        existing = existing[:idx].rstrip() + "\n\n"
    DOC.write_text(existing + "\n" + section)

    subprocess.run(["git", "add", str(DOC)], cwd=str(ROOT), check=True)
    subprocess.run(["git", "commit", "-m",
                    "Phase-2 Task 1 (beam=5) + Task 3 (lang-only LoRA) results\n\n"
                    "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"],
                   cwd=str(ROOT), check=True)
    print(f"[{time.strftime('%H:%M:%S')}] FINALIZE_TASKS_1_3_DONE", flush=True)


if __name__ == "__main__":
    main()
