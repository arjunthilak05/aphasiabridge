"""Orchestrator for the full B3/B4 SS-FT sweep across all SS speakers.

Maintains a queue of (speaker, condition) pairs, dispatches them across
the requested GPUs, and waits for completion. Skips any speaker/condition
whose lora_final/ already exists, so re-running is safe.

Usage:
    python scripts/run_b3_b4_sweep.py \\
        --gpus 0,1,2,3,4,5,6,7 \\
        --d4b-dir outputs/h100_lora_daniel_v4b/lora_final \\
        --ss-train ../aphasiabank/aphasiabank_ss_train.jsonl \\
        --output-root outputs \\
        --max-steps 80
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from collections import Counter

REPO = "/data/kl_dev/dataset_wan/data_gemma/aphasiabridge"


def load_speakers(ss_train_path: str, min_train: int = 8):
    with open(ss_train_path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    counts = Counter(r["participant_id"] for r in records)
    return sorted([s for s, n in counts.items() if n >= min_train])


def adapter_exists(root: str, condition: str, speaker: str) -> bool:
    return os.path.isdir(os.path.join(root, f"ss_ft_{condition}", speaker, "lora_final"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7",
                    help="Comma-separated GPU indices to use")
    ap.add_argument("--d4b-dir", required=True)
    ap.add_argument("--ss-train", required=True)
    ap.add_argument("--output-root", default=os.path.join(REPO, "outputs"))
    ap.add_argument("--log-dir", default=os.path.join(REPO, "logs"))
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--min-train", type=int, default=8)
    ap.add_argument("--conditions", default="b3,b4",
                    help="Comma-separated subset of conditions to run")
    ap.add_argument("--limit-speakers", type=int, default=None,
                    help="Only run first N speakers (debug)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    conditions = args.conditions.split(",")
    speakers = load_speakers(args.ss_train, min_train=args.min_train)
    if args.limit_speakers:
        speakers = speakers[: args.limit_speakers]

    # Build queue of (condition, speaker) skipping already-done
    queue = []
    skipped = 0
    for spk in speakers:
        for cond in conditions:
            if adapter_exists(args.output_root, cond, spk):
                skipped += 1
                continue
            queue.append((cond, spk))

    print(f"[orchestrator] {len(speakers)} speakers × {len(conditions)} conds = {len(speakers)*len(conditions)} pairs")
    print(f"  already-done (skipped): {skipped}")
    print(f"  to-do queue:            {len(queue)}")
    print(f"  GPUs:                   {gpus} ({len(gpus)} workers)")

    if args.dry_run:
        for cond, spk in queue[:20]:
            print(f"  WILL: {cond:3s}  {spk}")
        if len(queue) > 20:
            print(f"  ... and {len(queue)-20} more")
        return

    os.makedirs(args.log_dir, exist_ok=True)

    running = {}   # gpu_id -> (popen, cond, spk, start_time)
    queue_iter = iter(queue)

    def launch_next(gpu: int):
        try:
            cond, spk = next(queue_iter)
        except StopIteration:
            return False
        out_dir = os.path.join(args.output_root, f"ss_ft_{cond}", spk)
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(args.log_dir, f"ss_ft_{cond}_{spk}.log")

        cmd = [
            "python", os.path.join(REPO, "scripts", "train_ss_ft.py"),
            "--start-from", "d4b" if cond == "b3" else "base",
            "--manifest", args.ss_train,
            "--participant-id", spk,
            "--output-dir", out_dir,
            "--max-steps", str(args.max_steps),
            "--batch-size", str(args.batch_size),
            "--grad-accum", str(args.grad_accum),
            "--lr", str(args.lr),
        ]
        if cond == "b3":
            cmd += ["--si-lora-dir", args.d4b_dir]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with open(log_path, "w") as logf:
            p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                 cwd=REPO, env=env)
        running[gpu] = (p, cond, spk, time.time())
        print(f"[{time.strftime('%H:%M:%S')}] gpu={gpu} START  {cond:3s} {spk}", flush=True)
        return True

    # Initial fill
    for g in gpus:
        if not launch_next(g):
            break

    # Wait + relaunch loop
    done = 0
    while running:
        time.sleep(2)
        for gpu in list(running.keys()):
            p, cond, spk, t0 = running[gpu]
            if p.poll() is None:
                continue
            elapsed = time.time() - t0
            rc = p.returncode
            adapter_ok = adapter_exists(args.output_root, cond, spk)
            tag = "OK" if (rc == 0 and adapter_ok) else f"FAIL rc={rc} adapter={adapter_ok}"
            print(f"[{time.strftime('%H:%M:%S')}] gpu={gpu} {tag}  {cond:3s} {spk}  ({elapsed/60:.1f} min)", flush=True)
            done += 1
            del running[gpu]
            launch_next(gpu)

    print(f"\n[{time.strftime('%H:%M:%S')}] DONE. {done} jobs completed.", flush=True)


if __name__ == "__main__":
    main()
