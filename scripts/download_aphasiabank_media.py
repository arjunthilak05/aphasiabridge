"""Download AphasiaBank media and extract 16 kHz mono WAV in one step.

Uses ffmpeg to read the MP4 directly over HTTP (with auth cookies) and write
only the audio stream, so nothing video-sized ever touches disk.

    ffmpeg -headers "Cookie: ...\r\n" -i <url>.mp4 -vn -ac 1 -ar 16000 -c:a pcm_s16le out.wav

Output: <root>/wav/<Folder>/<Corpus>/<subpath>/<session>.wav

Reads catalog.csv produced by `download_aphasiabank.py --stage catalog`.
Resume-friendly: skips sessions whose WAV already exists and is non-empty.
"""
from __future__ import annotations
import argparse, csv, os, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Reuse auth + constants from the sibling downloader.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_aphasiabank import make_session, login, DEFAULT_ROOT, MEDIA_BASE  # type: ignore


def cookie_header(session) -> str:
    """Serialize session cookies into a single 'Cookie:' header value."""
    return "; ".join(f"{c.name}={c.value}" for c in session.cookies)


# Module-level auth state, guarded by a lock so worker threads can re-auth safely.
_auth_lock = threading.Lock()
_auth_session = None
_auth_cookie = ""
_auth_last_refresh = 0.0


def auth_init():
    global _auth_session, _auth_cookie, _auth_last_refresh
    with _auth_lock:
        _auth_session = make_session()
        login(_auth_session)
        _auth_cookie = cookie_header(_auth_session)
        _auth_last_refresh = time.time()


def auth_get_cookie() -> str:
    with _auth_lock:
        return _auth_cookie


def auth_refresh(reason: str = "") -> bool:
    """Re-login. If another thread just refreshed, skip (cooldown 15s)."""
    global _auth_cookie, _auth_last_refresh
    with _auth_lock:
        # Cooldown: if someone else just refreshed <15s ago, reuse their cookie
        if time.time() - _auth_last_refresh < 15:
            return True
        try:
            login(_auth_session)
            _auth_cookie = cookie_header(_auth_session)
            _auth_last_refresh = time.time()
            print(f"[auth] refreshed ({reason})", flush=True)
            return True
        except Exception as e:
            print(f"[auth] refresh failed: {e}", flush=True)
            return False


# Corpora that return 401 even after fresh login are gated (separate password).
# Tracked at runtime so we can fast-fail the rest of their files.
_blacklist_lock = threading.Lock()
_blacklisted_corpora: set[str] = set()


def corpus_is_gated(corpus: str) -> bool:
    with _blacklist_lock:
        return corpus in _blacklisted_corpora


def blacklist_corpus(corpus: str) -> bool:
    """Returns True if this is the first time we're blacklisting this corpus."""
    with _blacklist_lock:
        if corpus in _blacklisted_corpora:
            return False
        _blacklisted_corpora.add(corpus)
        return True


def run_ffmpeg(url: str, cookie: str, dst: Path, timeout: int = 1800) -> tuple[bool, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".wav.part")
    headers = f"Cookie: {cookie}\r\nUser-Agent: aphasiabank-dl/1.0\r\n"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        # HTTP options passed before -i
        "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "30",
        "-headers", headers,
        "-i", url,
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        "-f", "wav",           # force wav muxer (output path has .part extension)
        "-y", str(tmp),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return False, "timeout"
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return False, f"exc {e}"
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        err = r.stderr.decode("utf-8", errors="ignore").strip()
        # Keep last line for brevity
        last = err.splitlines()[-1] if err else f"rc={r.returncode}"
        return False, last[:400]
    try:
        tmp.rename(dst)
    except Exception as e:
        return False, f"rename {e}"
    return True, ""


def media_url_to_wav_path(media_url: str, wav_root: Path) -> Path:
    tail = media_url.split(f"{MEDIA_BASE}/", 1)[1]           # aphasia/English/...  (actually just English/...)
    # tail already starts with Protocol/<corpus>/... because MEDIA_BASE ends in /aphasia/English
    stem = tail.rsplit(".", 1)[0]
    return wav_root / (stem + ".wav")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--jobs", type=int, default=2, help="parallel ffmpeg workers")
    ap.add_argument("--limit", type=int, default=0, help="max sessions to process")
    ap.add_argument("--corpora", nargs="*")
    ap.add_argument("--folders", nargs="*")
    ap.add_argument("--severity", nargs="*")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--max-retries", type=int, default=3,
                    help="retry a failed session this many times (with re-auth on 401)")
    ap.add_argument("--skip-corpora", nargs="*", default=["Marshall"],
                    help="corpora to preemptively skip (known to require separate access)")
    args = ap.parse_args()

    cat = args.root / "catalog.csv"
    if not cat.exists():
        sys.exit(f"Missing {cat}. Run --stage catalog first.")
    rows = list(csv.DictReader(open(cat, encoding="utf-8")))
    if args.folders:  rows = [r for r in rows if r["folder"] in args.folders]
    if args.corpora:  rows = [r for r in rows if r["corpus"] in args.corpora]
    if args.severity: rows = [r for r in rows if r["severity"] in args.severity]
    rows = [r for r in rows if r.get("media_url")]
    if args.limit:
        rows = rows[:args.limit]
    print(f"[media] {len(rows)} sessions to process with {args.jobs} workers",
          file=sys.stderr, flush=True)

    wav_root = args.root / "wav"; wav_root.mkdir(parents=True, exist_ok=True)
    fail_log = args.root / "media_failures.txt"

    auth_init()
    if not auth_get_cookie():
        sys.exit("No cookies after login; check credentials.")

    # Preload blacklist from CLI (e.g. Marshall is separately password-gated)
    for c in args.skip_corpora or []:
        blacklist_corpus(c)
    if args.skip_corpora:
        print(f"[media] preemptively skipping gated corpora: {args.skip_corpora}",
              file=sys.stderr, flush=True)

    # Pre-filter: skip already-downloaded
    pending = []
    prefilt_total_bytes = 0
    for r in rows:
        dst = media_url_to_wav_path(r["media_url"], wav_root)
        if dst.exists() and dst.stat().st_size > 0:
            prefilt_total_bytes += dst.stat().st_size
            continue
        pending.append((r, dst))
    print(f"[media] {len(pending)} pending after skip-cached "
          f"({(len(rows)-len(pending))} already done, "
          f"{prefilt_total_bytes/1e9:.2f} GB on disk)",
          file=sys.stderr, flush=True)

    done = 0; fail = 0; total_bytes = prefilt_total_bytes
    t0 = time.time()

    def task(item):
        r, dst = item
        corpus = r["corpus"]
        # Fast path: corpus already known to be gated
        if corpus_is_gated(corpus):
            return (False, "[gated corpus, skipped]", r, dst, 0, 0)

        last_msg = ""
        for attempt in range(1, args.max_retries + 1):
            ok, msg = run_ffmpeg(r["media_url"], auth_get_cookie(), dst,
                                 timeout=args.timeout)
            if ok:
                sz = dst.stat().st_size if dst.exists() else 0
                return (True, "", r, dst, sz, attempt)
            last_msg = msg
            is_auth_err = ("401" in msg or "Unauthorized" in msg
                           or "authorization failed" in msg.lower())
            if is_auth_err and attempt < args.max_retries:
                auth_refresh(reason=f"401 on {corpus}/{r['file']}")
                # If another thread already blacklisted this corpus, bail early
                if corpus_is_gated(corpus):
                    return (False, "[gated corpus, skipped]", r, dst, 0, attempt)
                continue
            if attempt < args.max_retries:
                time.sleep(2 * attempt)
                continue
            break

        # Exhausted retries. If final error was 401, this corpus is gated.
        if ("401" in last_msg or "Unauthorized" in last_msg
                or "authorization failed" in last_msg.lower()):
            if blacklist_corpus(corpus):
                print(f"[gated] corpus {corpus!r} returns 401 after re-auth; "
                      f"blacklisting all remaining files",
                      flush=True)
        return (False, last_msg, r, dst, 0, args.max_retries)

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = {ex.submit(task, it): it for it in pending}
        for i, fut in enumerate(as_completed(futures), 1):
            ok, msg, r, dst, sz, tries = fut.result()
            tag = f"(try {tries})" if tries > 1 else ""
            if ok:
                done += 1; total_bytes += sz
                dt = time.time() - t0
                rate = (done / dt) * 60 if dt else 0
                eta_min = (len(pending) - i) / max(rate, 1e-6) if rate else -1
                print(f"[dl {i:4d}/{len(pending)}] OK {r['corpus']}/{r['file']}  "
                      f"{sz/1e6:.1f}MB  total={total_bytes/1e9:.2f}GB  "
                      f"{rate:.1f}/min  eta={eta_min:.0f}min {tag}",
                      flush=True)
            else:
                fail += 1
                with open(fail_log, "a", encoding="utf-8") as fp:
                    fp.write(f"{r['media_url']}\t{msg}\n")
                print(f"[dl {i:4d}/{len(pending)}] FAIL {r['corpus']}/{r['file']}  "
                      f"{msg[:160]} {tag}",
                      flush=True)

    dt = time.time() - t0
    print(f"[done] ok={done} fail={fail} wav_total={total_bytes/1e9:.2f}GB "
          f"time={dt/60:.1f}min  ({done/(dt/60):.1f}/min)" if dt else f"[done] ok={done}",
          flush=True)


if __name__ == "__main__":
    main()
