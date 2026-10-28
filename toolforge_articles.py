"""Collect revision-pinned x_lang/English article contexts on Toolforge.

Run through language_selection.py inside a Toolforge CPU job, for example:
    python language_selection.py --x-lang bar --limit 10 --output-dir runs/bar_test

Use --all for a full corpus. Rerun the same command to resume an interrupted
collection. Raw wikitext is preserved; extraction and sentence splitting are
separate steps. Language links identify comparable topics, not translations.
"""

import hashlib
import json
import os
import re
import tempfile
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen


ENGLISH_SITE = "https://en.wikipedia.org"
USER_AGENT = (
    "WikiParallelCorpus/0.2 "
    "(https://toolsadmin.wikimedia.org/tools/id/wiki-parallel-corpus)"
)
SQL = """
SELECT p.page_id, p.page_title, p.page_latest, ll.ll_title
FROM page AS p
JOIN langlinks AS ll ON ll.ll_from = p.page_id
WHERE ll.ll_lang = %s AND p.page_namespace = 0 AND p.page_is_redirect = 0
ORDER BY p.page_id
"""


class UnavailablePair(ValueError):
    """An explicitly unavailable or inconsistent article link, recorded as skipped."""


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def positive_id(value):
    if type(value) is not int or value <= 0:
        raise ValueError("Article and revision IDs must be positive integers")
    return value


def api(site, **params):
    """One bounded read; dependency failures abort instead of becoming missing pairs."""
    params = {"action": "query", "format": "json", "formatversion": 2, "maxlag": 5, **params}
    request = Request(f"{site}/w/api.php?{urlencode(params)}", headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        payload = response.read(32_000_001)
    if len(payload) > 32_000_000:
        raise ValueError(f"Wikipedia response exceeds 32 MB: {site}, {params['action']}")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError(f"Malformed Wikipedia response: {site}")
    if any(key in data for key in ("error", "warnings", "continue")):
        code = data.get("error", {}).get("code", "incomplete-response")
        raise RuntimeError(f"Wikipedia API failure: {site}, {params['action']}, {code}")
    time.sleep(1)
    return data


def resolve_wiki(x_lang):
    if not isinstance(x_lang, str) or not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", x_lang):
        raise ValueError("x_lang must be a Wikipedia language code, e.g. bar or nds-nl")
    if x_lang == "en":
        raise ValueError("x_lang must differ from English")
    # Wikimedia's mapping also handles codes whose database names differ, e.g. be-tarask.
    matrix = api(ENGLISH_SITE, action="sitematrix", smtype="language",
                 smlangprop="code|site", smsiteprop="code|dbname|url").get("sitematrix")
    if not isinstance(matrix, dict):
        raise ValueError("Wikipedia site matrix is missing")
    for language in matrix.values():
        if not isinstance(language, dict) or language.get("code") != x_lang:
            continue
        for site in language.get("site", []):
            if site.get("code") != "wiki":
                continue
            if any(flag in site for flag in ("closed", "private", "fishbowl")):
                raise ValueError(f"Wikipedia is not an active public wiki: {x_lang}")
            database, url = site.get("dbname", ""), site.get("url", "")
            parsed = urlsplit(url)
            if (not re.fullmatch(r"[a-z][a-z0-9_]*wiki", database)
                    or parsed.scheme != "https"
                    or not re.fullmatch(r"[a-z0-9-]+\.wikipedia\.org", parsed.netloc)
                    or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
                raise ValueError(f"Unexpected Wikipedia site configuration: {x_lang}")
            return {"language": x_lang, "database": database, "url": url.rstrip("/")}
    raise ValueError(f"No Wikipedia found for x_lang={x_lang}")


def linked_articles(wiki, credentials):
    import pymysql

    with closing(pymysql.connect(
        host=f"{wiki['database']}.analytics.db.svc.wikimedia.cloud",
        database=wiki["database"] + "_p", read_default_file=str(credentials),
        charset="utf8mb4", connect_timeout=10, read_timeout=75, write_timeout=10,
        autocommit=True,
    )) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET SESSION max_statement_time = 60")
            cursor.execute(SQL, ("en",))
            rows = cursor.fetchall()
    candidates = []
    for page_id, title, revision_id, en_title in rows:
        title = title.decode("utf-8") if isinstance(title, bytes) else title
        en_title = en_title.decode("utf-8") if isinstance(en_title, bytes) else en_title
        if not isinstance(title, str) or not title or not isinstance(en_title, str) or not en_title:
            raise ValueError("Empty or malformed title returned by Wiki Replicas")
        candidates.append({"page_id": positive_id(page_id), "title": title.replace("_", " "),
                           "revision_id": positive_id(revision_id), "en_title": en_title})
    if len({row["page_id"] for row in candidates}) != len(candidates):
        raise ValueError("Wiki Replicas returned repeated source articles")
    return candidates


def save_json(path, value):
    """Atomically publish complete JSON without overwriting different existing data."""
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise FileExistsError(f"Existing data differs: {path}; use a new output directory")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        staging = Path(temporary.name)
        try:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        except BaseException:
            staging.unlink()
            raise
    try:
        os.link(staging, path)
    finally:
        staging.unlink()


def fetch_article(site, language, **selector):
    data = api(site, prop="info|pageprops|revisions|langlinks", rvprop="ids|timestamp|content",
               rvslots="main", ppprop="wikibase_item|disambiguation", lllang="en", lllimit=1,
               **selector)
    pages = data.get("query", {}).get("pages")
    if not isinstance(pages, list) or len(pages) != 1 or not isinstance(pages[0], dict):
        if "badrevids" in data.get("query", {}):
            raise UnavailablePair(f"Revision no longer publicly available: {language}")
        raise ValueError(f"Expected exactly one article: {language}")
    page = pages[0]
    if any(flag in page for flag in ("missing", "invalid", "redirect")) or page.get("ns") != 0:
        raise UnavailablePair(f"Missing, redirected, or non-article page: {language}")
    revisions = page.get("revisions")
    if not isinstance(revisions, list) or len(revisions) != 1:
        raise ValueError(f"Missing revision metadata: {language}")
    revision = revisions[0]
    slot = revision.get("slots", {}).get("main", {})
    if any(flag in revision or flag in slot for flag in ("texthidden", "suppressed")):
        raise UnavailablePair(f"Revision content is hidden: {language}")
    text = slot.get("content")
    if "contentmodel" not in slot:
        raise ValueError(f"Missing revision content model: {language}")
    if slot["contentmodel"] != "wikitext":
        raise UnavailablePair(f"Article content is not wikitext: {language}")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Missing revision content: {language}")
    page_id, revision_id = positive_id(page.get("pageid")), positive_id(revision.get("revid"))
    if "revids" in selector and revision_id != selector["revids"]:
        raise ValueError(f"API returned a different revision: {language}")
    title, timestamp = page.get("title"), revision.get("timestamp")
    if not isinstance(title, str) or not title.strip() or not isinstance(timestamp, str):
        raise ValueError(f"Missing article title or timestamp: {language}")
    properties = page.get("pageprops", {})
    qid = properties.get("wikibase_item")
    if not isinstance(qid, str) or not re.fullmatch(r"Q[1-9][0-9]*", qid):
        raise UnavailablePair(f"No Wikidata topic ID: {language}")
    article = {
        "language": language, "page_id": page_id, "title": title,
        "page_url": f"{site}/?curid={page_id}", "revision_id": revision_id,
        "revision_url": f"{site}/w/index.php?oldid={revision_id}",
        "revision_timestamp": timestamp, "retrieved_at": now(),
        "wikidata_item": qid, "is_disambiguation": "disambiguation" in properties,
        "wikitext": text, "wikitext_sha256": digest(text.encode("utf-8")),
    }
    return article, page.get("langlinks", [])


def collect_pair(wiki, candidate):
    article, links = fetch_article(wiki["url"], wiki["language"], revids=candidate["revision_id"])
    if article["page_id"] != candidate["page_id"]:
        raise ValueError("API returned a different source page")
    english_titles = [link["title"] for link in links if link.get("lang") == "en"]
    if len(english_titles) != 1 or english_titles[0].replace("_", " ") != candidate["en_title"].replace("_", " "):
        raise UnavailablePair("Language link changed since the replica selection")
    english, _ = fetch_article(ENGLISH_SITE, "en", titles=english_titles[0])
    if article["wikidata_item"] != english["wikidata_item"]:
        raise UnavailablePair("Source and English articles have different Wikidata topic IDs")
    return {"status": "collected", "candidate": candidate, "x": article, "en": english}


def validate_record(record, candidate, x_lang):
    if record.get("candidate") != candidate:
        raise ValueError("Saved pair differs from the frozen selection")
    if record.get("status") == "skipped":
        if not isinstance(record.get("reason"), str) or not record["reason"]:
            raise ValueError("Skipped pair is missing its reason")
        return
    if record.get("status") != "collected":
        raise ValueError("Unknown saved pair status")
    for key, language in (("x", x_lang), ("en", "en")):
        article = record[key]
        positive_id(article["page_id"])
        positive_id(article["revision_id"])
        if (article["language"] != language or not isinstance(article["wikitext"], str)
                or not article["wikitext"].strip()
                or digest(article["wikitext"].encode("utf-8")) != article["wikitext_sha256"]):
            raise ValueError("Saved article language or content checksum differs")
    if (record["x"]["page_id"] != candidate["page_id"]
            or record["x"]["revision_id"] != candidate["revision_id"]
            or record["x"]["wikidata_item"] != record["en"]["wikidata_item"]):
        raise ValueError("Saved article identities differ from the selected pair")


def collect_articles(x_lang, output_dir, limit=10):
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer or None for all candidates")
    credentials = Path.home() / "replica.my.cnf"
    if not credentials.is_file():
        raise FileNotFoundError("Run collection in your Toolforge tool account; replica.my.cnf is missing")
    output_dir = Path(output_dir)
    settings = {"schema_version": 1, "x_lang": x_lang, "limit": limit,
                "code_sha256": digest(Path(__file__).read_bytes())}
    selection_path = output_dir / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection["settings"] != settings:
            raise ValueError("Selection settings or code changed; use a new output directory")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError("Output directory is not empty and has no selection; choose a new directory")
        wiki = resolve_wiki(x_lang)
        candidates = linked_articles(wiki, credentials)
        selection = {"settings": settings, "wiki": wiki, "selected_at": now(),
                     "sql": SQL, "sql_parameters": ["en"], "candidates": candidates}
        save_json(selection_path, selection)
    selection_hash = digest(selection_path.read_bytes())
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if manifest is not None and manifest["selection_sha256"] != selection_hash:
        raise ValueError("Completed selection checksum differs")
    collected, skipped, records = 0, 0, []
    print(f"{x_lang}–en: {len(selection['candidates'])} linked candidates in frozen selection", flush=True)
    for candidate in selection["candidates"]:
        path = output_dir / "raw" / f"{x_lang}_{candidate['page_id']}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
        else:
            if manifest is not None:
                raise FileNotFoundError(f"Completed corpus record is missing: {path}")
            try:
                record = collect_pair(selection["wiki"], candidate)
            except UnavailablePair as error:
                record = {"status": "skipped", "candidate": candidate, "reason": str(error)}
            save_json(path, record)
        validate_record(record, candidate, x_lang)
        collected += record["status"] == "collected"
        skipped += record["status"] == "skipped"
        records.append({"file": str(path.relative_to(output_dir)), "sha256": digest(path.read_bytes()),
                        "status": record["status"]})
        if len(records) % 100 == 0:
            print(f"Processed {len(records)}: {collected} collected, {skipped} skipped", flush=True)
        if limit is not None and collected >= limit:
            break
    if manifest is not None:
        if manifest["records"] != records:
            raise ValueError("Completed corpus record checksums differ")
    else:
        manifest = {"completed_at": now(), "selection_sha256": selection_hash,
                    "linked_candidates": len(selection["candidates"]), "collected": collected,
                    "skipped": skipped, "records": records,
                    "limit_reached": limit is None or collected >= limit,
                    "content_format": "raw MediaWiki wikitext; no extraction or sentence splitting"}
        save_json(manifest_path, manifest)
    print(f"Saved {collected} article pairs, skipped {skipped}; output: {output_dir}", flush=True)
    return manifest
