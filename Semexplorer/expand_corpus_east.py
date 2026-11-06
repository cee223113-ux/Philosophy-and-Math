#!/usr/bin/env python3
"""
Expand the EASTERN side of the philosophy corpus to ~10,000 texts,
with the strongest possible weight on China.

Sources: Project Gutenberg via the Gutendex API — James Legge's
Chinese Classics series, the Sacred Books of the East volumes, Giles's
Taoist translations, plus original-language Chinese texts (論語, 道德經,
莊子, 孟子, 孫子兵法 …) chunked by characters instead of words.

Run ON YOUR OWN COMPUTER (needs internet):

    python3 expand_corpus_east.py            # harvest until East = 10,000
    python3 expand_corpus_east.py --stats    # show East/West counts only

The script is resumable: texts already in the database are skipped
(exact-text dedup), and it stops as soon as the Eastern sphere reaches
TARGET_EAST_TOTAL. It reports real numbers only — no padding.
"""

import argparse
import json
import os
import re
import sqlite3
import time
import urllib.parse
import urllib.request

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "philosophy_corpus.db")

USER_AGENT = (
    "PhilosophySemanticExplorer/1.0 "
    "(independent student research project)"
)

GUTENDEX = "https://gutendex.com/books"

PASSAGE_MIN_WORDS = 120
PASSAGE_MAX_WORDS = 250
ZH_PASSAGE_MIN_CHARS = 200
ZH_PASSAGE_MAX_CHARS = 450
MAX_PASSAGES_PER_BOOK = 1500
TARGET_EAST_TOTAL = 10000
REQUEST_PAUSE = 0.3

EAST_REGIONS = (
    "East Asia", "South Asia", "Southeast Asia", "Central Asia",
    "Middle East & North Africa", "Sub-Saharan Africa",
)

# (gutendex search query, author, work, country, region, tradition, era)
# China first — it carries the most weight by design.
BOOKS = [
    # ---------------- China (English translations) ----------------
    ("Chinese Classics Confucian Analects Legge",
     "Confucius (tr. James Legge)",
     "The Chinese Classics, Vol. 1: Analects, Great Learning, "
     "Doctrine of the Mean", "China", "East Asia", "Confucianism",
     "5th c. BCE"),
    ("Works of Mencius Legge", "Mencius (tr. James Legge)", "Mencius",
     "China", "East Asia", "Confucianism", "4th c. BCE"),
    ("Shoo King Legge", "Unknown (tr. James Legge)",
     "The Shoo King (Book of Documents)", "China", "East Asia",
     "Confucian classics", "c. 1000 BCE"),
    ("She King Legge", "Unknown (tr. James Legge)",
     "The She King (Book of Poetry)", "China", "East Asia",
     "Confucian classics", "c. 800 BCE"),
    ("Ch'un Ts'ew Spring Autumn Legge", "Confucius (tr. James Legge)",
     "The Ch'un Ts'ew (Spring and Autumn Annals)", "China", "East Asia",
     "Confucian classics", "5th c. BCE"),
    ("Sacred Books of the East volume 16 Yi King",
     "Unknown (tr. James Legge)", "The Yi King (I Ching)", "China",
     "East Asia", "Chinese cosmology", "c. 900 BCE"),
    ("Sacred Books of the East volume 27 Li Ki",
     "Unknown (tr. James Legge)", "The Lî Kî (Book of Rites), Part I",
     "China", "East Asia", "Confucian classics", "c. 200 BCE"),
    ("Sacred Books of the East volume 28 Li Ki",
     "Unknown (tr. James Legge)", "The Lî Kî (Book of Rites), Part II",
     "China", "East Asia", "Confucian classics", "c. 200 BCE"),
    ("Sacred Books of the East volume 3 Shu King",
     "Unknown (tr. James Legge)",
     "Sacred Books of China: Shu, Shih and Hsiao King", "China",
     "East Asia", "Confucian classics", "c. 800 BCE"),
    ("Tao Teh King Legge", "Laozi (tr. James Legge)", "Tao Te Ching",
     "China", "East Asia", "Daoism", "4th c. BCE"),
    ("Sacred Books of the East volume 39 Taoism",
     "Zhuangzi (tr. James Legge)", "The Texts of Taoism, Part I",
     "China", "East Asia", "Daoism", "4th c. BCE"),
    ("Sacred Books of the East volume 40 Taoism",
     "Zhuangzi (tr. James Legge)", "The Texts of Taoism, Part II",
     "China", "East Asia", "Daoism", "4th c. BCE"),
    ("Taoist Teachings Lieh Tzu Giles", "Liezi (tr. Lionel Giles)",
     "Taoist Teachings from the Book of Lieh Tzŭ", "China", "East Asia",
     "Daoism", "4th c. BCE"),
    ("Sayings of Lao Tzu Giles", "Laozi (tr. Lionel Giles)",
     "The Sayings of Lao Tzŭ", "China", "East Asia", "Daoism",
     "4th c. BCE"),
    ("Musings Chinese Mystic Chuang", "Zhuangzi (tr. Lionel Giles)",
     "Musings of a Chinese Mystic", "China", "East Asia", "Daoism",
     "4th c. BCE"),
    ("Sayings of Confucius", "Confucius",
     "The Sayings of Confucius", "China", "East Asia", "Confucianism",
     "5th c. BCE"),
    ("Ethical Political Works Motse", "Mozi (tr. Y. P. Mei)",
     "The Ethical and Political Works of Motse", "China", "East Asia",
     "Mohism", "5th c. BCE"),
    ("Hsun Tzu Moulder Ancient Confucianism Dubs",
     "Xunzi (tr. Homer H. Dubs)",
     "Hsün Tzŭ: The Moulder of Ancient Confucianism", "China",
     "East Asia", "Confucianism", "3rd c. BCE"),
    ("Art of War Sunzi Giles", "Sun Tzu (tr. Lionel Giles)",
     "The Art of War", "China", "East Asia", "Strategic philosophy",
     "5th c. BCE"),
    ("History of Chinese Literature Giles", "Herbert A. Giles",
     "A History of Chinese Literature", "China", "East Asia",
     "Chinese literary thought", "1901"),
    ("Religions of Ancient China Giles", "Herbert A. Giles",
     "Religions of Ancient China", "China", "East Asia",
     "Chinese religious thought", "1905"),
    ("Confucianism and its Rivals Giles", "Herbert A. Giles",
     "Confucianism and Its Rivals", "China", "East Asia",
     "Confucianism", "1915"),
    ("Civilization of China Giles", "Herbert A. Giles",
     "The Civilization of China", "China", "East Asia",
     "Chinese culture studies", "1911"),
    ("China and the Chinese Giles", "Herbert A. Giles",
     "China and the Chinese", "China", "East Asia",
     "Chinese culture studies", "1902"),

    # ---------------- India ----------------
    ("Sacred Books of the East volume 1 Upanishads",
     "Unknown (tr. Max Müller)", "The Upanishads, Part I", "India",
     "South Asia", "Vedanta", "c. 700 BCE"),
    ("Sacred Books of the East volume 15 Upanishads",
     "Unknown (tr. Max Müller)", "The Upanishads, Part II", "India",
     "South Asia", "Vedanta", "c. 700 BCE"),
    ("Laws of Manu Buhler", "Manu (tr. Georg Bühler)",
     "The Laws of Manu", "India", "South Asia", "Dharmaśāstra",
     "c. 200 BCE"),
    ("Buddhist Suttas Sacred Books East", "The Buddha (tr. Rhys Davids)",
     "Buddhist Suttas", "India", "South Asia", "Buddhism",
     "3rd c. BCE"),
    ("Sutta-Nipata Fausboll", "The Buddha (tr. Viggo Fausböll)",
     "The Sutta-Nipata", "India", "South Asia", "Buddhism",
     "3rd c. BCE"),
    ("Crest-Jewel of Wisdom Shankara", "Śaṅkarācārya (tr. Charles Johnston)",
     "The Crest-Jewel of Wisdom", "India", "South Asia",
     "Advaita Vedanta", "8th c. CE"),
    ("Jnana Yoga Vivekananda", "Swami Vivekananda", "Jnana Yoga",
     "India", "South Asia", "Vedanta", "1899"),
    ("Karma Yoga Vivekananda", "Swami Vivekananda", "Karma Yoga",
     "India", "South Asia", "Vedanta", "1896"),
    ("Bhakti Yoga Vivekananda", "Swami Vivekananda", "Bhakti Yoga",
     "India", "South Asia", "Vedanta", "1896"),
    ("Creative Unity Tagore", "Rabindranath Tagore", "Creative Unity",
     "India", "South Asia", "Bengali humanism", "1922"),
    ("Personality Tagore", "Rabindranath Tagore", "Personality",
     "India", "South Asia", "Bengali humanism", "1917"),
    ("Stray Birds Tagore", "Rabindranath Tagore", "Stray Birds",
     "India", "South Asia", "Bengali humanism", "1916"),
    ("Gospel of Ramakrishna", "Ramakrishna (rec. Mahendranath Gupta)",
     "The Gospel of Ramakrishna", "India", "South Asia",
     "Hindu mysticism", "1897"),

    # ---------------- Japan ----------------
    ("Religion of the Samurai Nukariya", "Kaiten Nukariya",
     "The Religion of the Samurai: Zen Buddhism", "Japan", "East Asia",
     "Zen Buddhism", "1913"),
    ("Ideals of the East Okakura", "Kakuzō Okakura",
     "The Ideals of the East", "Japan", "East Asia",
     "Japanese aesthetics", "1903"),
    ("Book of Tea Okakura", "Kakuzō Okakura", "The Book of Tea",
     "Japan", "East Asia", "Japanese aesthetics", "1906"),
    ("Gleanings in Buddha-Fields Hearn", "Lafcadio Hearn",
     "Gleanings in Buddha-Fields", "Japan", "East Asia",
     "Buddhist thought", "1897"),
    ("Out of the East Hearn", "Lafcadio Hearn", "Out of the East",
     "Japan", "East Asia", "Japanese culture studies", "1895"),

    # ---------------- Persia / Middle East ----------------
    ("Masnavi I Manavi Rumi", "Rumi (tr. E. H. Whinfield)",
     "The Masnavi I Ma'navi", "Iran", "Middle East & North Africa",
     "Sufism", "13th c. CE"),
    ("Gulistan Saadi", "Saadi", "The Gulistan (Rose Garden)", "Iran",
     "Middle East & North Africa", "Persian ethics", "1258"),
    ("Improvement of Human Reason Ibn Tufail",
     "Ibn Tufail (tr. Simon Ockley)",
     "The Improvement of Human Reason (Hayy ibn Yaqzan)",
     "Spain / Arab world", "Middle East & North Africa",
     "Islamic philosophy", "12th c. CE"),
    ("Confessions of Al Ghazzali", "Al-Ghazali (tr. Claud Field)",
     "The Confessions of Al Ghazzali", "Iran",
     "Middle East & North Africa", "Islamic philosophy", "11th c. CE"),
    ("Arabian Wisdom Wortabet", "John Wortabet (ed.)",
     "Arabian Wisdom: Selections and Translations", "Arab world",
     "Middle East & North Africa", "Arab ethics", "1907"),

    # ---------------- Tibet ----------------
    ("Tibetan Book of the Dead Evans-Wentz",
     "Padmasambhava (tr. Evans-Wentz)", "The Tibetan Book of the Dead",
     "Tibet", "Central Asia", "Tibetan Buddhism", "8th c. CE"),
    ("Voice of the Silence Blavatsky", "H. P. Blavatsky (tr.)",
     "The Voice of the Silence", "Tibet", "Central Asia",
     "Tibetan Buddhist thought", "1889"),
]

# Original-language Chinese classics (Gutenberg Chinese collection).
# Chunked by characters, not words.
ZH_BOOKS = [
    ("論語", "Confucius 孔子", "論語 (The Analects, original text)",
     "China", "East Asia", "Confucianism", "5th c. BCE"),
    ("道德經", "Laozi 老子", "道德經 (Tao Te Ching, original text)",
     "China", "East Asia", "Daoism", "4th c. BCE"),
    ("莊子", "Zhuangzi 莊子", "莊子 (Zhuangzi, original text)",
     "China", "East Asia", "Daoism", "4th c. BCE"),
    ("孟子", "Mencius 孟子", "孟子 (Mencius, original text)",
     "China", "East Asia", "Confucianism", "4th c. BCE"),
    ("孫子兵法", "Sun Tzu 孫子", "孫子兵法 (Art of War, original text)",
     "China", "East Asia", "Strategic philosophy", "5th c. BCE"),
    ("墨子", "Mozi 墨子", "墨子 (Mozi, original text)",
     "China", "East Asia", "Mohism", "5th c. BCE"),
    ("荀子", "Xunzi 荀子", "荀子 (Xunzi, original text)",
     "China", "East Asia", "Confucianism", "3rd c. BCE"),
    ("韓非子", "Han Feizi 韓非子", "韓非子 (Han Feizi, original text)",
     "China", "East Asia", "Legalism", "3rd c. BCE"),
    ("大學", "Zengzi 曾子", "大學 (The Great Learning, original text)",
     "China", "East Asia", "Confucianism", "5th c. BCE"),
    ("中庸", "Zisi 子思", "中庸 (Doctrine of the Mean, original text)",
     "China", "East Asia", "Confucianism", "5th c. BCE"),
]


def http_get(url, timeout=90, retries=3):
    last_exc = None

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:
            last_exc = exc
            wait = min(2 ** attempt * 2, 20)
            print(f"  request failed ({exc}); retry {attempt}/{retries} "
                  f"in {wait}s…")
            time.sleep(wait)

    raise last_exc


def find_book(query, language=None):
    """Search Gutendex; return (title, txt_url) of the best match."""
    params = {"search": query}
    if language:
        params["languages"] = language
    url = GUTENDEX + "?" + urllib.parse.urlencode(params)
    data = json.loads(http_get(url))

    for book in data.get("results", []):
        formats = book.get("formats", {})
        for mime, link in formats.items():
            if mime.startswith("text/plain") and link.endswith(".txt"):
                return book.get("title", query), link

    return None, None


def strip_gutenberg(raw):
    """Remove the Project Gutenberg header/footer boilerplate."""
    text = raw.decode("utf-8", errors="replace")

    start = re.search(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG",
                      text)
    end = re.search(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", text)

    if start:
        text = text[start.end():]
        text = text[text.find("\n"):]
    if end:
        text = text[: end.start()]

    return text


def chunk_book(text):
    """Split an English book into 120–250 word passages."""
    text = re.sub(r"\r\n?", "\n", text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text)]

    passages = []
    current, words = [], 0

    for p in paragraphs:
        p = re.sub(r"\s+", " ", p)
        w = len(p.split())

        if w < 15 or (p.isupper() and w < 12):
            continue
        if "gutenberg" in p.lower()[:40]:
            continue

        current.append(p)
        words += w

        if words >= PASSAGE_MIN_WORDS:
            passage = " ".join(current)
            if len(passage.split()) <= PASSAGE_MAX_WORDS * 1.5:
                passages.append(passage)
            current, words = [], 0

        if len(passages) >= MAX_PASSAGES_PER_BOOK:
            break

    return passages


def chunk_book_zh(text):
    """Split a Chinese book into 200–450 character passages."""
    text = re.sub(r"\r\n?", "\n", text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text)]

    passages = []
    current, chars = [], 0

    for p in paragraphs:
        p = re.sub(r"\s+", "", p)
        c = len(p)

        if c < 30:
            continue
        if "gutenberg" in p.lower()[:40]:
            continue

        current.append(p)
        chars += c

        if chars >= ZH_PASSAGE_MIN_CHARS:
            passage = "".join(current)
            if len(passage) <= ZH_PASSAGE_MAX_CHARS * 1.5:
                passages.append(passage)
            current, chars = [], 0

        if len(passages) >= MAX_PASSAGES_PER_BOOK:
            break

    return passages


def east_count(conn):
    placeholders = ",".join("?" for _ in EAST_REGIONS)
    return conn.execute(
        f"SELECT COUNT(*) FROM texts WHERE region IN ({placeholders})",
        EAST_REGIONS,
    ).fetchone()[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", action="store_true",
                        help="Print East/West statistics and exit.")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)

    if args.stats:
        total = conn.execute("SELECT COUNT(*) FROM texts").fetchone()[0]
        print(f"Texts: {total}")
        east = east_count(conn)
        print(f"East: {east}   West/other: {total - east}")
        for row in conn.execute(
            "SELECT region, COUNT(*) c FROM texts GROUP BY region"
            " ORDER BY c DESC"
        ):
            print(f"  {row[0] or '(unknown)'}: {row[1]}")
        return

    existing = {
        row[0] for row in conn.execute("SELECT text FROM texts")
    }

    current_east = east_count(conn)
    print(f"East corpus currently: {current_east} texts "
          f"(target {TARGET_EAST_TOTAL})")

    total_added = 0

    jobs = [(b, "English", chunk_book, None) for b in BOOKS]
    jobs += [(b, "Chinese", chunk_book_zh, "zh") for b in ZH_BOOKS]

    for (book, language, chunker, lang_filter) in jobs:
        if current_east >= TARGET_EAST_TOTAL:
            print(f"East target {TARGET_EAST_TOTAL} reached — stopping.")
            break

        query, author, work, country, region, tradition, era = book
        print(f"\n=== {work} — {author} ({country}) ===")

        try:
            title, txt_url = find_book(query, language=lang_filter)
        except Exception as exc:
            print(f"  search failed: {exc}")
            continue

        if not txt_url:
            print("  no plain-text edition found; skipping")
            continue

        print(f"  downloading: {title}")
        time.sleep(REQUEST_PAUSE)

        try:
            raw = http_get(txt_url)
        except Exception as exc:
            print(f"  download failed: {exc}")
            continue

        passages = chunker(strip_gutenberg(raw))
        print(f"  {len(passages)} passages extracted")

        rows = []
        for p in passages:
            if p in existing:
                continue
            existing.add(p)
            rows.append((
                p, author, work, country, region, tradition, language,
                era, "Project Gutenberg (public domain)",
            ))

        if rows:
            conn.executemany(
                "INSERT INTO texts (text, author, work, country, region,"
                " tradition, language, era, source)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                rows,
            )
            conn.commit()

        current_east += len(rows)
        total_added += len(rows)
        print(f"  inserted {len(rows)} (East now {current_east})")

    total = conn.execute("SELECT COUNT(*) FROM texts").fetchone()[0]
    print(f"\nDone. Added {total_added} passages.")
    print(f"Corpus now: {total} texts; East: {east_count(conn)} "
          f"(target {TARGET_EAST_TOTAL}).")
    print("Reopen the app to embed the new texts (first run will be "
          "slow, then cached).")

    conn.close()


if __name__ == "__main__":
    main()
