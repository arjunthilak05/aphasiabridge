"""Download AphasiaBank (TalkBank) CHA transcripts and linked media.

Auth: cookie-based via POST https://sla2.talkbank.org/logInUser with JSON {email, pswd}.
Data: walks https://talkbank.org/data/aphasia/English/<Folder>/ recursively for .cha files.
Media: linked files live at https://media.talkbank.org/aphasia/English/<Folder>/<path>.<ext>.

Credentials come from env vars TB_USER and TB_PASS (never hardcoded).

Phase A:  --stage cha              (downloads all .cha transcripts; small)
Phase B:  --stage media --filter X (downloads media for matching sessions; large)
Catalog:  --stage catalog          (parses downloaded .cha files into a CSV)

Example:
  export TB_USER='you@example.com'; export TB_PASS='...'
  python scripts/download_aphasiabank.py --stage cha --folders Protocol NonProtocol Script
  python scripts/download_aphasiabank.py --stage catalog
  python scripts/download_aphasiabank.py --stage media --max-sessions 50 --severity mild moderate
"""
from __future__ import annotations
import argparse, csv, json, os, re, sys, time, urllib.parse, zipfile
from http.cookiejar import MozillaCookieJar
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DATA_BASE  = "https://talkbank.org/data/aphasia/English"
MEDIA_BASE = "https://media.talkbank.org/aphasia/English"
LOGIN_URL  = "https://sla2.talkbank.org/logInUser"
DEFAULT_ROOT = Path(r"D:\Downloads\aphasiabank")

HREF_RE = re.compile(r'href="([^"]+)"')


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.5,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET", "HEAD", "POST"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "aphasiabank-dl/1.0 (research fine-tune)"})
    return s


def login(s: requests.Session) -> None:
    user = os.environ.get("TB_USER")
    pwd  = os.environ.get("TB_PASS")
    if not user or not pwd:
        sys.exit("ERROR: set env vars TB_USER and TB_PASS before running.")
    r = s.post(LOGIN_URL,
               json={"email": user, "pswd": pwd},
               headers={"Origin": "https://media.talkbank.org",
                        "Referer": "https://media.talkbank.org/"},
               timeout=30)
    r.raise_for_status()
    j = r.json()
    if not j.get("success") or not j.get("authStatus", {}).get("authorized"):
        sys.exit(f"Login failed: {j}")
    print(f"[auth] logged in as {user}, authorized=True", file=sys.stderr)


def list_dir(s: requests.Session, url: str) -> list[str]:
    """Return child URLs (folders + files) from an nginx-style directory listing."""
    r = s.get(url, timeout=30)
    if r.status_code != 200 or "text/html" not in r.headers.get("Content-Type", ""):
        return []
    # The listing puts each row's link in href="..."; dedupe and drop "parent"/anchor links
    hrefs = HREF_RE.findall(r.text)
    out, seen = [], set()
    for h in hrefs:
        if h.startswith("#") or h.endswith("Parent Directory"):
            continue
        absu = urllib.parse.urljoin(url, h)
        # Drop ?f=save / ?f=open query suffixes (same file served twice per row)
        absu_clean = absu.split("?", 1)[0]
        if absu_clean in seen or absu_clean == url.rstrip("/"):
            continue
        seen.add(absu_clean)
        out.append(absu_clean)
    return out


def walk_cha(s: requests.Session, url: str, out: list[str], depth=0) -> None:
    """DFS the directory tree, collecting every .cha URL."""
    u = url + ("/" if not url.endswith("/") else "")
    t0 = time.time()
    children = list_dir(s, u)
    cha_here = sum(1 for c in children if c.endswith(".cha"))
    subdirs  = [c for c in children if "." not in c.rsplit("/", 1)[-1]]
    print(f"[walk] d={depth} {u}  +{cha_here}cha  {len(subdirs)}subdirs  ({time.time()-t0:.1f}s)",
          file=sys.stderr, flush=True)
    for child in children:
        if child.endswith(".cha"):
            out.append(child)
        elif child.endswith(".cdc") or child.endswith(".pdf") or child.endswith(".txt"):
            continue
        elif "." not in child.rsplit("/", 1)[-1]:
            walk_cha(s, child, out, depth+1)


def save_to(s: requests.Session, url: str, dst: Path, skip_if_exists=True) -> int:
    if skip_if_exists and dst.exists() and dst.stat().st_size > 0:
        return dst.stat().st_size
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with s.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    tmp.rename(dst)
    return dst.stat().st_size


def data_url_to_local(url: str, root: Path) -> Path:
    """https://talkbank.org/data/aphasia/English/<rest>  ->  <root>/cha/<rest>"""
    tail = url.split("/data/aphasia/English/", 1)[1]
    return root / "cha" / tail


def list_corpora(s: requests.Session, folder_url: str) -> list[str]:
    """Return the subdirectories (corpora) immediately under a Folder listing."""
    children = list_dir(s, folder_url if folder_url.endswith("/") else folder_url + "/")
    corpora = []
    for c in children:
        last = c.rstrip("/").rsplit("/", 1)[-1]
        if "." in last:   # a file, not a corpus dir
            continue
        if c.rstrip("/") == folder_url.rstrip("/"):
            continue
        corpora.append(c.rstrip("/"))
    return corpora


def download_zip(s: requests.Session, zip_url: str, zip_dst: Path, label: str) -> bool:
    """Stream a ?f=zip response to disk. Returns True on a valid zip."""
    t0 = time.time()
    try:
        with s.get(zip_url, stream=True, timeout=(30, 1800)) as r:
            if r.status_code != 200:
                print(f"[zip] {label}: HTTP {r.status_code}", file=sys.stderr)
                return False
            tmp = zip_dst.with_suffix(".zip.part")
            zip_dst.parent.mkdir(parents=True, exist_ok=True)
            n = 0; last_log = time.time()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 18):
                    if not chunk: continue
                    f.write(chunk); n += len(chunk)
                    if time.time() - last_log > 10:
                        print(f"[zip] {label}: {n/1e6:.1f} MB...", file=sys.stderr, flush=True)
                        last_log = time.time()
            tmp.rename(zip_dst)
    except Exception as e:
        print(f"[zip] {label}: exception {e}", file=sys.stderr)
        return False
    ok = zipfile.is_zipfile(zip_dst)
    status = "ok" if ok else "BAD"
    print(f"[zip] {label}: {n/1e6:.2f} MB in {time.time()-t0:.1f}s [{status}]",
          file=sys.stderr)
    return ok


def extract_cha(zip_dst: Path, cha_root: Path, rel_prefix: str) -> tuple[int, int]:
    """Extract .cha entries from a corpus zip into cha_root/<rel_prefix>/...
    Returns (files_extracted, bytes_extracted)."""
    files = bytes_ = 0
    with zipfile.ZipFile(zip_dst) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".cha"):
                continue
            parts = Path(name).parts
            # Drop any leading path component that duplicates rel_prefix pieces;
            # simplest: keep the filename + any subdirs below the corpus root.
            # The zip entries look like "BU/PWA/BU01a.cha" — so we can just join.
            dst = cha_root / rel_prefix / Path(*parts)
            if dst.exists() and dst.stat().st_size > 0:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(dst, "wb") as out:
                buf = src.read(); out.write(buf)
            files += 1
            bytes_ += len(buf)
    return files, bytes_


def stage_cha(args):
    """Grab CHA transcripts using TalkBank's per-corpus ?f=zip endpoint.

    Per-folder top-level ?f=zip (e.g. Protocol?f=zip) returns 404. We instead
    list each folder, enumerate its corpora, and pull one zip per corpus.
    """
    s = make_session(); login(s)
    zips_dir = args.root / "zips"; zips_dir.mkdir(parents=True, exist_ok=True)
    cha_root = args.root / "cha"; cha_root.mkdir(parents=True, exist_ok=True)

    grand_files = 0; grand_bytes = 0; failed = []
    for folder in args.folders:
        folder_url = f"{DATA_BASE}/{folder}/"
        corpora = list_corpora(s, folder_url)
        print(f"[plan] {folder}: {len(corpora)} corpora: "
              f"{[c.rsplit('/',1)[-1] for c in corpora]}", file=sys.stderr)
        folder_zips_dir = zips_dir / folder
        folder_zips_dir.mkdir(parents=True, exist_ok=True)
        for cidx, corpus_url in enumerate(corpora, 1):
            corpus = corpus_url.rsplit("/", 1)[-1]
            label = f"{folder}/{corpus} ({cidx}/{len(corpora)})"
            zip_url = f"{corpus_url}?f=zip"
            zip_dst = folder_zips_dir / f"{corpus}.zip"
            if zip_dst.exists() and zipfile.is_zipfile(zip_dst):
                print(f"[zip] {label}: cached", file=sys.stderr)
                ok = True
            else:
                print(f"[zip] {label}: GET {zip_url}", file=sys.stderr)
                ok = download_zip(s, zip_url, zip_dst, label)
            if not ok:
                failed.append(label)
                continue
            files, bytes_ = extract_cha(zip_dst, cha_root, f"{folder}/{corpus}")
            grand_files += files; grand_bytes += bytes_
            print(f"[ext] {label}: +{files} .cha  (running {grand_files} files, "
                  f"{grand_bytes/1e6:.1f} MB)", file=sys.stderr)

    print(f"[done] Phase A: {grand_files} .cha files, {grand_bytes/1e6:.1f} MB. "
          f"failed={len(failed)} {failed}", file=sys.stderr)


# --- CHA parsing for catalog ---

ID_RE    = re.compile(r"^@ID:\s*(?P<body>.*)$")
MEDIA_RE = re.compile(r"^@Media:\s*(?P<name>[^,\s]+)(?:\s*,\s*(?P<kind>\w+))?")
TIME_RE  = re.compile(r"\x15(\d+)_(\d+)\x15")


def parse_cha(path: Path) -> dict:
    """Extract speaker metadata + duration from one .cha file."""
    info = {"cha_path": str(path), "participants": [], "par_wab_aq": None,
            "par_type": None, "par_age": None, "par_sex": None,
            "media_name": None, "media_kind": None,
            "n_par_utts": 0, "duration_ms": 0}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return info
    max_end = 0
    last_speaker = None
    for line in text.splitlines():
        if line.startswith("@ID:"):
            # @ID:	eng|BU|INV|||||Investigator|||
            # @ID:	eng|BU|PAR|62;06.|male|Anomic||Participant||85.7|
            parts = line[4:].strip().split("|")
            if len(parts) >= 3:
                role = parts[2].strip()
                info["participants"].append(role)
                if role == "PAR":
                    info["par_age"] = parts[3] if len(parts) > 3 else None
                    info["par_sex"] = parts[4] if len(parts) > 4 else None
                    info["par_type"] = parts[5] if len(parts) > 5 else None
                    try:
                        info["par_wab_aq"] = float(parts[9]) if len(parts) > 9 and parts[9].strip() else None
                    except ValueError:
                        info["par_wab_aq"] = None
        elif line.startswith("@Media:"):
            m = MEDIA_RE.match(line)
            if m:
                info["media_name"] = m.group("name")
                info["media_kind"] = m.group("kind")
        elif line.startswith("*"):
            # *SPEAKER:\t<text>   \x15<ms>_<ms>\x15
            header, _, rest = line.partition(":")
            last_speaker = header[1:].strip()
            if last_speaker == "PAR":
                info["n_par_utts"] += 1
            for mm in TIME_RE.finditer(line):
                try:
                    end = int(mm.group(2))
                    if end > max_end:
                        max_end = end
                except Exception:
                    pass
    info["duration_ms"] = max_end
    return info


def derive_severity(aq):
    if aq is None: return None
    if aq >= 76: return "mild"
    if aq >= 51: return "moderate"
    if aq >= 26: return "severe"
    return "very_severe"


def stage_catalog(args):
    cha_root = args.root / "cha"
    rows = []
    cha_files = sorted(cha_root.rglob("*.cha"))
    print(f"[catalog] parsing {len(cha_files)} .cha files", file=sys.stderr)
    for p in cha_files:
        info = parse_cha(p)
        rel = p.relative_to(cha_root)
        parts = rel.parts  # (Folder, Corpus, ...possibly subgroup..., file.cha)
        info["folder"]    = parts[0] if len(parts) > 0 else None
        info["corpus"]    = parts[1] if len(parts) > 1 else None
        info["subgroup"]  = "/".join(parts[2:-1]) if len(parts) > 3 else (parts[2] if len(parts) > 2 else None)
        info["file"]      = parts[-1]
        info["severity"]  = derive_severity(info["par_wab_aq"])
        info["media_url"] = None
        if info["media_name"] and info["media_kind"]:
            ext = {"video": "mp4", "audio": "mp3"}.get(info["media_kind"], "mp4")
            media_rel = str(rel.parent).replace("\\", "/")
            info["media_url"] = f"{MEDIA_BASE}/{media_rel}/{info['media_name']}.{ext}"
        info["participants"] = ",".join(info["participants"])
        rows.append(info)

    out_csv = args.root / "catalog.csv"
    fields = ["folder","corpus","subgroup","file","par_type","severity","par_wab_aq",
              "par_age","par_sex","n_par_utts","duration_ms","media_name","media_kind",
              "media_url","participants","cha_path"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k: r.get(k) for k in fields})

    # Summary
    from collections import Counter
    by_corpus = Counter(r["corpus"] for r in rows)
    by_sev    = Counter(r["severity"] for r in rows)
    by_type   = Counter(r["par_type"] for r in rows if r["par_type"])
    total_sec = sum(r["duration_ms"] for r in rows) / 1000.0
    print(f"[catalog] {len(rows)} sessions, total linked duration ~{total_sec/3600:.1f} h", file=sys.stderr)
    print(f"[catalog] by corpus (top 10): {by_corpus.most_common(10)}", file=sys.stderr)
    print(f"[catalog] by severity: {dict(by_sev)}", file=sys.stderr)
    print(f"[catalog] by aphasia type (top 8): {by_type.most_common(8)}", file=sys.stderr)
    print(f"[catalog] wrote {out_csv}", file=sys.stderr)


def stage_media(args):
    import csv as _csv
    cat = args.root / "catalog.csv"
    if not cat.exists():
        sys.exit("Run --stage catalog first.")
    s = make_session(); login(s)
    rows = list(_csv.DictReader(open(cat, encoding="utf-8")))
    # Filters
    if args.severity:
        rows = [r for r in rows if r["severity"] in args.severity]
    if args.corpora:
        rows = [r for r in rows if r["corpus"] in args.corpora]
    if args.folders:
        rows = [r for r in rows if r["folder"] in args.folders]
    rows = [r for r in rows if r.get("media_url")]
    if args.max_sessions:
        rows = rows[:args.max_sessions]
    print(f"[media] will download {len(rows)} sessions", file=sys.stderr)

    total = 0
    for i, r in enumerate(rows, 1):
        url = r["media_url"]
        tail = url.split(f"{MEDIA_BASE}/", 1)[1]
        dst  = args.root / "media" / tail
        try:
            sz = save_to(s, url, dst)
        except Exception as e:
            print(f"[err] {url}: {e}", file=sys.stderr); continue
        total += sz
        print(f"[dl] {i:4d}/{len(rows)}  {tail}  {sz/1e6:.1f} MB  (total {total/1e9:.2f} GB)", file=sys.stderr)
    print(f"[done] Phase B complete: {len(rows)} media files, {total/1e9:.2f} GB", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["cha", "catalog", "media"])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--folders", nargs="+",
                    default=["Protocol", "NonProtocol", "Script"],
                    help="English/<folder>. Default: Protocol NonProtocol Script.")
    ap.add_argument("--corpora", nargs="*", help="Filter by corpus name for --stage media")
    ap.add_argument("--severity", nargs="*", help="Filter: mild / moderate / severe / very_severe")
    ap.add_argument("--max-sessions", type=int, default=0)
    args = ap.parse_args()
    args.max_sessions = args.max_sessions or None

    t0 = time.time()
    if args.stage == "cha":
        stage_cha(args)
    elif args.stage == "catalog":
        stage_catalog(args)
    elif args.stage == "media":
        stage_media(args)
    print(f"[time] {time.time()-t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
