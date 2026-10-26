"""Download pinned Bavarian/English revisions; existing snapshots are immutable."""

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
session = requests.Session()
session.headers["User-Agent"] = (
    "GroundedGenerativeAlignment/0.2 "
    "(https://github.com/miumiu1337/grounded-generative-alignment)"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    """Publish a complete JSON file without replacing a different existing file."""
    path = Path(path)
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise FileExistsError(f"Existing data differs: {path}. Use a new output directory.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        except BaseException:
            temporary_path.unlink()
            raise
    try:
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink()


def request_api(language, params):
    if language not in {"bar", "en"}:
        raise ValueError(f"Unsupported language: {language}")
    response = session.get(
        f"https://{language}.wikipedia.org/w/api.php",
        params={"format": "json", "formatversion": 2, "maxlag": 5, **params},
        timeout=(10, 60),
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError(f"Invalid MediaWiki response for {language}")
    if "error" in data or "warnings" in data:
        raise RuntimeError(f"MediaWiki {language}: {data.get('error', data.get('warnings'))}")
    time.sleep(1)
    return data


def validate_article(article, language, page_id, revision_id):
    if (article.get("language") != language or article.get("page_id") != page_id
            or article.get("revision_id") != revision_id):
        raise ValueError(f"Snapshot identity differs from selection: {language}:{page_id}:{revision_id}")
    for key in ("title", "wikitext", "page_url", "revision_url", "revision_timestamp", "retrieved_at"):
        if not isinstance(article.get(key), str) or not article[key].strip():
            raise ValueError(f"Missing or empty {key} for {language}:{page_id}")


def download_article(language, page_id, revision_id):
    data = request_api(language, {
        "action": "query", "prop": "revisions", "revids": revision_id,
        "rvprop": "ids|timestamp|content", "rvslots": "main",
    })
    pages = data.get("query", {}).get("pages", [])
    if len(pages) != 1 or not pages[0].get("revisions"):
        raise RuntimeError(f"Article unavailable: {language}:{page_id}:{revision_id}")
    page = pages[0]
    revision = page["revisions"][0]
    article = {
        "language": language,
        "page_id": page["pageid"],
        "title": page["title"],
        "page_url": f"https://{language}.wikipedia.org/?curid={page['pageid']}",
        "revision_id": revision["revid"],
        "revision_url": f"https://{language}.wikipedia.org/w/index.php?oldid={revision['revid']}",
        "revision_timestamp": revision["timestamp"],
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "wikitext": revision["slots"]["main"]["content"],
    }
    validate_article(article, language, page_id, revision_id)
    return article


def load_selection(path=ROOT / "data/articles.csv"):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = {"x_page_id", "x_title", "x_revision_id", "en_page_id", "en_title", "en_revision_id"}
        if not fields.issubset(reader.fieldnames or []):
            raise ValueError(f"Selection CSV needs columns: {sorted(fields)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty article selection")
    for row in rows:
        for field in ("x_page_id", "x_revision_id", "en_page_id", "en_revision_id"):
            row[field] = int(row[field])
            if row[field] <= 0:
                raise ValueError(f"Invalid {field}: {row[field]}")
        if not row["x_title"].strip() or not row["en_title"].strip():
            raise ValueError("Selection titles must not be empty")
    for field in ("x_page_id", "en_page_id"):
        if len({row[field] for row in rows}) != len(rows):
            raise ValueError(f"Repeated article in selection: {field}")
    return rows


def load_raw_pairs(rows, directory=ROOT / "data/raw"):
    pairs = {}
    for path in sorted(directory.glob("bar_*.json")):
        pair = json.loads(path.read_text(encoding="utf-8"))
        page_id = pair["bar"]["page_id"]
        if page_id in pairs:
            raise ValueError(f"Duplicate raw article: {page_id}")
        pairs[page_id] = (path, pair)
    selected = {row["x_page_id"] for row in rows}
    if set(pairs) - selected:
        raise ValueError(f"Unselected raw articles: {sorted(set(pairs) - selected)}")
    for row in rows:
        if row["x_page_id"] not in pairs:
            continue
        path, pair = pairs[row["x_page_id"]]
        for language, prefix in (("bar", "x"), ("en", "en")):
            validate_article(pair[language], language, row[f"{prefix}_page_id"], row[f"{prefix}_revision_id"])
            if pair[language]["title"] != row[f"{prefix}_title"]:
                raise ValueError(f"Snapshot title differs from selection: {path}, {language}")
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT / "data/articles.csv")
    args = parser.parse_args()
    rows = load_selection(args.selection)
    pairs = load_raw_pairs(rows)
    for row in rows:
        page_id = row["x_page_id"]
        if page_id in pairs:
            print(f"Validated existing snapshot: {row['x_title']}")
            continue
        pair = {language: download_article(language, row[f"{prefix}_page_id"], row[f"{prefix}_revision_id"])
                for language, prefix in (("bar", "x"), ("en", "en"))}
        save_json(ROOT / "data/raw" / f"bar_{page_id}.json", pair)
        print(f"Saved: {pair['bar']['title']} ↔ {pair['en']['title']}")


if __name__ == "__main__":
    main()
