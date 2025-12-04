# src/fetch_templates.py
import os, requests, json, sys
from pathlib import Path

OUT_DIR = Path("data/templates")
OUT_DIR.mkdir(parents=True, exist_ok=True)
JSONL_FILE = OUT_DIR / "templates.jsonl"
IMGFLIP_API = "https://api.imgflip.com/get_memes"

def fetch_templates():
    print("Fetching template list from Imgflip...")
    r = requests.get(IMGFLIP_API, timeout=15)
    r.raise_for_status()
    payload = r.json()
    if not payload.get("success"):
        print("Imgflip API returned failure:", payload)
        sys.exit(1)
    memes = payload["data"]["memes"]
    return memes

def show_list(memes, n=100):
    print(f"\nShowing first {n} templates (index : name (w x h) )\n")
    for i, m in enumerate(memes[:n]):
        print(f"{i:3d}: {m['name']} ({m['width']}x{m['height']}) url={m['url']} box_count={m.get('box_count')}")
    print()

def parse_selection(sel, max_idx):
    sel = sel.strip().lower()
    if sel.startswith("auto"):
        # auto:N or auto (default 20)
        parts = sel.split(":")
        n = int(parts[1]) if len(parts) > 1 else 20
        return list(range(min(n, max_idx+1)))
    if sel in ("all", "a"):
        return list(range(max_idx+1))
    indices = set()
    for part in sel.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            indices.update(range(int(start), int(end)+1))
        else:
            indices.add(int(part))
    return sorted(i for i in indices if 0 <= i <= max_idx)

def download_image(url, dest_path):
    if dest_path.exists():
        print(f" - already downloaded {dest_path.name}")
        return
    print(f"Downloading {url} -> {dest_path.name}")
    r = requests.get(url, stream=True, timeout=20)
    r.raise_for_status()
    with open(dest_path, "wb") as fw:
        for chunk in r.iter_content(1024):
            fw.write(chunk)

def main():
    memes = fetch_templates()
    show_list(memes, n=100)

    sel = input("Enter indices (e.g. 0,2,3 or 0-10) or 'auto:20' or 'all': ").strip()
    if not sel:
        print("No selection, exiting.")
        return

    indices = parse_selection(sel, len(memes)-1)
    if not indices:
        print("No valid indices parsed. Exiting.")
        return

    entries = []
    for i in indices:
        m = memes[i]
        # sanitize filename: id + basename
        fname = f"{m['id']}_{os.path.basename(m['url'])}"
        fpath = OUT_DIR / fname
        try:
            download_image(m['url'], fpath)
        except Exception as e:
            print(f"Failed to download {m['url']}: {e}")
            continue
        entry = {
            "template": m["name"],
            "template_url": m["url"],
            "template_id": m["id"],
            "width": m["width"],
            "height": m["height"],
            "file_path": str(fpath.as_posix())
        }
        entries.append(entry)

    # Append to templates.jsonl
    with open(JSONL_FILE, "a", encoding="utf-8") as out:
        for e in entries:
            out.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\n✅ Done. Wrote {len(entries)} templates to {JSONL_FILE}")
    print("IMPORTANT: Open data/templates/ and manually inspect images. Remove any templates that already contain captions (not blank).")

if __name__ == "__main__":
    main()
