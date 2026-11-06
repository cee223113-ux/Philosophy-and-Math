#!/usr/bin/env python3
"""
Expand the philosophy corpus by harvesting real quotes from
Wikidata + English Wikiquote.

Pipeline:
  1. Wikidata SPARQL → philosophers with a country of citizenship
     and an English Wikiquote page.
  2. Wikiquote API   → pull each philosopher's page, extract the
     bullet-listed quotes.
  3. Clean, dedupe, and insert into philosophy_corpus.db.

Run this ON YOUR OWN COMPUTER (it needs internet access):

    python3 expand_corpus.py                # harvest everything found
    python3 expand_corpus.py --limit 500    # harvest at most 500 authors
    python3 expand_corpus.py --stats        # show DB stats only

The script is resumable: authors already present in the database are
skipped, so you can interrupt and re-run it safely.

All data comes from Wikimedia projects (CC BY-SA). Attribution is
stored in the `source` column of every imported row.
"""

import argparse
import html
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
    "(independent student research project; contact via GitHub)"
)

SPARQL_URL = "https://query.wikidata.org/sparql"
WIKIQUOTE_API = "https://en.wikiquote.org/w/api.php"

SPARQL_QUERY = """
SELECT ?person ?personLabel ?countryLabel ?page WHERE {
  ?person wdt:P106 wd:Q4964182 .
  ?person wdt:P27 ?country .
  ?page schema:about ?person ;
        schema:isPartOf <https://en.wikiquote.org/> .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY ?personLabel
"""

MAX_QUOTES_PER_AUTHOR = 50
MIN_QUOTE_LEN = 25
MAX_QUOTE_LEN = 400
REQUEST_PAUSE = 0.2  # seconds between API calls, be nice to Wikimedia


# ------------------------------------------------------------------
# Country → region mapping (common cases; anything else stays '')
# ------------------------------------------------------------------

COUNTRY_REGION = {
    # Europe
    "Greece": "Europe", "Italy": "Europe", "France": "Europe",
    "Germany": "Europe", "Austria": "Europe", "Denmark": "Europe",
    "Russia": "Europe", "Russian Empire": "Europe",
    "United Kingdom": "Europe", "England": "Europe",
    "Scotland": "Europe", "Wales": "Europe", "Ireland": "Europe",
    "Spain": "Europe", "Portugal": "Europe", "Netherlands": "Europe",
    "Belgium": "Europe", "Switzerland": "Europe", "Sweden": "Europe",
    "Norway": "Europe", "Finland": "Europe", "Iceland": "Europe",
    "Poland": "Europe", "Czech Republic": "Europe", "Czechia": "Europe",
    "Czechoslovakia": "Europe", "Slovakia": "Europe", "Hungary": "Europe",
    "Romania": "Europe", "Bulgaria": "Europe", "Croatia": "Europe",
    "Serbia": "Europe", "Slovenia": "Europe", "Albania": "Europe",
    "Montenegro": "Europe", "North Macedonia": "Europe",
    "Bosnia and Herzegovina": "Europe", "Lithuania": "Europe",
    "Latvia": "Europe", "Estonia": "Europe", "Ukraine": "Europe",
    "Belarus": "Europe", "Moldova": "Europe", "Malta": "Europe",
    "Cyprus": "Europe", "Luxembourg": "Europe", "Georgia": "Europe",
    "Armenia": "Europe", "Azerbaijan": "Europe",
    "Austria-Hungary": "Europe", "Prussia": "Europe",
    "Kingdom of Prussia": "Europe", "Soviet Union": "Europe",
    "Ottoman Empire": "Middle East & North Africa",
    # Asia
    "China": "East Asia", "People's Republic of China": "East Asia",
    "Japan": "East Asia", "South Korea": "East Asia",
    "North Korea": "East Asia", "Mongolia": "East Asia",
    "India": "South Asia", "Pakistan": "South Asia",
    "Bangladesh": "South Asia", "Sri Lanka": "South Asia",
    "Nepal": "South Asia", "Bhutan": "South Asia",
    "British Raj": "South Asia",
    "Thailand": "Southeast Asia", "Vietnam": "Southeast Asia",
    "Myanmar": "Southeast Asia", "Cambodia": "Southeast Asia",
    "Laos": "Southeast Asia", "Malaysia": "Southeast Asia",
    "Singapore": "Southeast Asia", "Indonesia": "Southeast Asia",
    "Philippines": "Southeast Asia", "Brunei": "Southeast Asia",
    "Timor-Leste": "Southeast Asia",
    "Kazakhstan": "Central Asia", "Uzbekistan": "Central Asia",
    "Turkmenistan": "Central Asia", "Kyrgyzstan": "Central Asia",
    "Tajikistan": "Central Asia", "Afghanistan": "Central Asia",
    # Middle East & North Africa
    "Egypt": "Middle East & North Africa",
    "Iran": "Middle East & North Africa",
    "Iraq": "Middle East & North Africa",
    "Turkey": "Middle East & North Africa",
    "Israel": "Middle East & North Africa",
    "Lebanon": "Middle East & North Africa",
    "Jordan": "Middle East & North Africa",
    "Syria": "Middle East & North Africa",
    "Saudi Arabia": "Middle East & North Africa",
    "Yemen": "Middle East & North Africa",
    "United Arab Emirates": "Middle East & North Africa",
    "Qatar": "Middle East & North Africa",
    "Kuwait": "Middle East & North Africa",
    "Oman": "Middle East & North Africa",
    "Bahrain": "Middle East & North Africa",
    "Morocco": "Middle East & North Africa",
    "Algeria": "Middle East & North Africa",
    "Tunisia": "Middle East & North Africa",
    "Libya": "Middle East & North Africa",
    "Sudan": "Middle East & North Africa",
    "State of Palestine": "Middle East & North Africa",
    "Palestine": "Middle East & North Africa",
    # Africa
    "Ethiopia": "Sub-Saharan Africa", "Ghana": "Sub-Saharan Africa",
    "Nigeria": "Sub-Saharan Africa", "Kenya": "Sub-Saharan Africa",
    "South Africa": "Sub-Saharan Africa",
    "Tanzania": "Sub-Saharan Africa", "Uganda": "Sub-Saharan Africa",
    "Senegal": "Sub-Saharan Africa", "Mali": "Sub-Saharan Africa",
    "Cameroon": "Sub-Saharan Africa", "Rwanda": "Sub-Saharan Africa",
    "Zimbabwe": "Sub-Saharan Africa", "Botswana": "Sub-Saharan Africa",
    "Namibia": "Sub-Saharan Africa", "Zambia": "Sub-Saharan Africa",
    "Mozambique": "Sub-Saharan Africa",
    "Madagascar": "Sub-Saharan Africa", "Somalia": "Sub-Saharan Africa",
    "Eritrea": "Sub-Saharan Africa",
    "Democratic Republic of the Congo": "Sub-Saharan Africa",
    "Republic of the Congo": "Sub-Saharan Africa",
    "Angola": "Sub-Saharan Africa",
    "Ivory Coast": "Sub-Saharan Africa",
    "Burkina Faso": "Sub-Saharan Africa", "Niger": "Sub-Saharan Africa",
    # Americas
    "United States": "North America",
    "United States of America": "North America",
    "Canada": "North America", "Mexico": "North America",
    "Brazil": "Latin America & Caribbean",
    "Argentina": "Latin America & Caribbean",
    "Chile": "Latin America & Caribbean",
    "Peru": "Latin America & Caribbean",
    "Colombia": "Latin America & Caribbean",
    "Venezuela": "Latin America & Caribbean",
    "Ecuador": "Latin America & Caribbean",
    "Bolivia": "Latin America & Caribbean",
    "Uruguay": "Latin America & Caribbean",
    "Paraguay": "Latin America & Caribbean",
    "Cuba": "Latin America & Caribbean",
    "Haiti": "Latin America & Caribbean",
    "Dominican Republic": "Latin America & Caribbean",
    "Jamaica": "Latin America & Caribbean",
    "Guatemala": "Latin America & Caribbean",
    "Panama": "Latin America & Caribbean",
    "Costa Rica": "Latin America & Caribbean",
    "Trinidad and Tobago": "Latin America & Caribbean",
    "El Salvador": "Latin America & Caribbean",
    # Oceania
    "Australia": "Oceania", "New Zealand": "Oceania",
    "Fiji": "Oceania", "Papua New Guinea": "Oceania", "Samoa": "Oceania",
}

# Historical states → modern display name (keeps the nation count clean)
COUNTRY_NORMALIZE = {
    "Russian Empire": "Russia", "Soviet Union": "Russia",
    "Prussia": "Germany", "Kingdom of Prussia": "Germany",
    "Austria-Hungary": "Austria", "Czechoslovakia": "Czech Republic",
    "Czechia": "Czech Republic", "Ottoman Empire": "Turkey",
    "British Raj": "India", "People's Republic of China": "China",
    "United States of America": "United States",
    "England": "United Kingdom", "Scotland": "United Kingdom",
    "Wales": "United Kingdom",
    "Democratic Republic of the Congo": "DR Congo",
    "State of Palestine": "Palestine",
}


def http_get(url, params=None, timeout=60, retries=3):
    """GET with retry + exponential backoff (survives flaky proxies)."""
    if params:
        url = url + "?" + urllib.parse.urlencode(params)

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


def fetch_philosophers_sparql():
    """Wikidata SPARQL: philosophers with a country + en-Wikiquote page."""
    data = http_get(SPARQL_URL, {
        "query": SPARQL_QUERY,
        "format": "json",
    })
    results = json.loads(data)["results"]["bindings"]

    people = {}
    for row in results:
        name = row["personLabel"]["value"]
        country = row["countryLabel"]["value"]
        page = row["page"]["value"].rsplit("/", 1)[-1]
        page = urllib.parse.unquote(page).replace("_", " ")

        if name.startswith("Q"):  # unlabeled entity
            continue

        key = row["person"]["value"]
        people.setdefault(key, {
            "name": name,
            "page": page,
            "country": COUNTRY_NORMALIZE.get(country, country),
        })

    return list(people.values())


# Demonym in the category name → country (Wikiquote category tree uses
# "Category:<Demonym> philosophers" under "Category:Philosophers by
# nationality"). Unknown demonyms are skipped and reported.
DEMONYM_COUNTRY = {
    "American": "United States", "Chinese": "China", "Japanese": "Japan",
    "Korean": "South Korea", "South Korean": "South Korea",
    "Indian": "India", "Pakistani": "Pakistan", "Bangladeshi": "Bangladesh",
    "Sri Lankan": "Sri Lanka", "Nepalese": "Nepal", "Bhutanese": "Bhutan",
    "Thai": "Thailand", "Vietnamese": "Vietnam", "Burmese": "Myanmar",
    "Cambodian": "Cambodia", "Laotian": "Laos", "Malaysian": "Malaysia",
    "Singaporean": "Singapore", "Indonesian": "Indonesia",
    "Filipino": "Philippines", "Mongolian": "Mongolia",
    "Greek": "Greece", "Ancient Greek": "Greece", "Italian": "Italy",
    "French": "France", "German": "Germany", "Austrian": "Austria",
    "Danish": "Denmark", "Russian": "Russia", "British": "United Kingdom",
    "English": "United Kingdom", "Scottish": "United Kingdom",
    "Welsh": "United Kingdom", "Irish": "Ireland", "Spanish": "Spain",
    "Portuguese": "Portugal", "Dutch": "Netherlands", "Belgian": "Belgium",
    "Swiss": "Switzerland", "Swedish": "Sweden", "Norwegian": "Norway",
    "Finnish": "Finland", "Icelandic": "Iceland", "Polish": "Poland",
    "Czech": "Czech Republic", "Slovak": "Slovakia",
    "Hungarian": "Hungary", "Romanian": "Romania",
    "Bulgarian": "Bulgaria", "Croatian": "Croatia", "Serbian": "Serbia",
    "Slovenian": "Slovenia", "Albanian": "Albania",
    "Montenegrin": "Montenegro", "Macedonian": "North Macedonia",
    "Bosnian": "Bosnia and Herzegovina", "Lithuanian": "Lithuania",
    "Latvian": "Latvia", "Estonian": "Estonia", "Ukrainian": "Ukraine",
    "Belarusian": "Belarus", "Moldovan": "Moldova", "Maltese": "Malta",
    "Cypriot": "Cyprus", "Luxembourgian": "Luxembourg",
    "Georgian": "Georgia", "Armenian": "Armenia",
    "Azerbaijani": "Azerbaijan", "Kazakh": "Kazakhstan",
    "Uzbek": "Uzbekistan", "Turkmen": "Turkmenistan",
    "Kyrgyz": "Kyrgyzstan", "Tajik": "Tajikistan", "Afghan": "Afghanistan",
    "Egyptian": "Egypt", "Iranian": "Iran", "Persian": "Iran",
    "Iraqi": "Iraq", "Turkish": "Turkey", "Israeli": "Israel",
    "Lebanese": "Lebanon", "Jordanian": "Jordan", "Syrian": "Syria",
    "Saudi Arabian": "Saudi Arabia", "Saudi": "Saudi Arabia",
    "Yemeni": "Yemen", "Emirati": "United Arab Emirates",
    "Qatari": "Qatar", "Kuwaiti": "Kuwait", "Omani": "Oman",
    "Bahraini": "Bahrain", "Moroccan": "Morocco", "Algerian": "Algeria",
    "Tunisian": "Tunisia", "Libyan": "Libya", "Sudanese": "Sudan",
    "Palestinian": "Palestine", "Ethiopian": "Ethiopia",
    "Ghanaian": "Ghana", "Nigerian": "Nigeria", "Kenyan": "Kenya",
    "South African": "South Africa", "Tanzanian": "Tanzania",
    "Ugandan": "Uganda", "Senegalese": "Senegal", "Malian": "Mali",
    "Cameroonian": "Cameroon", "Rwandan": "Rwanda",
    "Zimbabwean": "Zimbabwe", "Namibian": "Namibia", "Zambian": "Zambia",
    "Mozambican": "Mozambique", "Malagasy": "Madagascar",
    "Somali": "Somalia", "Eritrean": "Eritrea", "Congolese": "DR Congo",
    "Angolan": "Angola", "Ivorian": "Ivory Coast",
    "Burkinabé": "Burkina Faso", "Nigerien": "Niger",
    "Canadian": "Canada", "Mexican": "Mexico", "Brazilian": "Brazil",
    "Argentine": "Argentina", "Chilean": "Chile", "Peruvian": "Peru",
    "Colombian": "Colombia", "Venezuelan": "Venezuela",
    "Ecuadorian": "Ecuador", "Bolivian": "Bolivia",
    "Uruguayan": "Uruguay", "Paraguayan": "Paraguay", "Cuban": "Cuba",
    "Haitian": "Haiti", "Dominican": "Dominican Republic",
    "Jamaican": "Jamaica", "Guatemalan": "Guatemala",
    "Panamanian": "Panama", "Costa Rican": "Costa Rica",
    "Trinidadian": "Trinidad and Tobago", "Salvadoran": "El Salvador",
    "Australian": "Australia", "New Zealand": "New Zealand",
    "Fijian": "Fiji", "Samoan": "Samoa",
}


def fetch_category_members(category, namespace=0):
    """List pages (namespace 0) or subcategories (14), paginated."""
    members = []
    cont = {}

    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category if category.startswith("Category:")
                       else f"Category:{category}",
            "cmlimit": "500",
            "cmnamespace": str(namespace),
            "format": "json",
        }
        params.update(cont)

        data = json.loads(http_get(WIKIQUOTE_API, params))
        members.extend(m["title"] for m in data["query"]["categorymembers"])

        if "continue" in data:
            cont = data["continue"]
        else:
            break

    return members


def fetch_philosophers_fallback():
    """
    Wikidata-free path: English Wikiquote organises philosophers under
    "Category:Philosophers by nationality" → "Category:<Demonym>
    philosophers". The demonym gives the country directly.
    """
    print("Falling back to Wikiquote category tree (no Wikidata needed)…")
    subcats = fetch_category_members(
        "Category:Philosophers by nationality", namespace=14
    )
    print(f"Found {len(subcats)} nationality categories.")

    people = []
    seen = set()
    skipped = []

    for cat in subcats:
        demonym = cat.replace("Category:", "").replace(" philosophers", "")
        country = DEMONYM_COUNTRY.get(demonym)

        if country is None:
            skipped.append(demonym)
            continue

        try:
            pages = fetch_category_members(cat, namespace=0)
        except Exception as exc:
            print(f"  {cat}: failed ({exc}); skipping")
            continue

        for page in pages:
            if page in seen:
                continue
            seen.add(page)
            people.append({
                "name": page,
                "page": page,
                "country": country,
            })

        print(f"  {demonym}: {len(pages)} philosophers")
        time.sleep(REQUEST_PAUSE)

    if skipped:
        print("Unmapped categories (skipped):", ", ".join(skipped))

    return people


def fetch_philosophers():
    try:
        return fetch_philosophers_sparql()
    except Exception as exc:
        print(f"Wikidata SPARQL failed ({exc}).")
        return fetch_philosophers_fallback()


def clean_quote(raw):
    """Strip wiki markup from a quote line; return '' if unusable."""
    text = raw.strip()

    # skip meta lines
    if re.match(r"^(file|image|thumb|category|defaultsort)", text, re.I):
        return ""

    # remove refs, html comments, tags
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"</?\w+[^>]*>", "", text)

    # templates: drop lines that are mostly a template
    if text.startswith("{{") and text.endswith("}}"):
        return ""
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)

    # links: [[a|b]] -> b ; [[a]] -> a
    text = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)

    # external links [url label] -> label
    text = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", text)

    # wiki bold/italic
    text = text.replace("'''", "").replace("''", "")

    text = html.unescape(text).strip(" \t\"'“”")

    # discard headings-like junk and attribution-only lines
    if not text or text.startswith(("=", ":", "#")):
        return ""
    if len(text.split()) < 4:
        return ""

    if not (MIN_QUOTE_LEN <= len(text) <= MAX_QUOTE_LEN):
        return ""

    return text


def fetch_quotes(page_title):
    """Pull one philosopher's Wikiquote page and extract quote lines."""
    data = http_get(WIKIQUOTE_API, {
        "action": "parse",
        "page": page_title,
        "prop": "wikitext",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
    })
    parsed = json.loads(data)

    if "parse" not in parsed:
        return []

    wikitext = parsed["parse"]["wikitext"]
    quotes = []

    for line in wikitext.splitlines():
        line = line.rstrip()

        # top-level bullet = a quote; deeper bullets are context
        if re.match(r"^\*\s", line) and not line.startswith("**"):
            q = clean_quote(line[2:])
            if q:
                quotes.append(q)

        if len(quotes) >= MAX_QUOTES_PER_AUTHOR * 2:
            break

    # keep the longest first N (longer quotes carry more semantics)
    quotes.sort(key=len, reverse=True)
    return quotes[:MAX_QUOTES_PER_AUTHOR]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0,
                        help="Max number of authors to harvest this run.")
    parser.add_argument("--stats", action="store_true",
                        help="Print corpus statistics and exit.")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(
        "CREATE TABLE IF NOT EXISTS texts ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL,"
        " author TEXT DEFAULT '', work TEXT DEFAULT '',"
        " country TEXT NOT NULL, region TEXT DEFAULT '',"
        " tradition TEXT DEFAULT '', language TEXT DEFAULT '',"
        " era TEXT DEFAULT '', source TEXT DEFAULT '');"
    )

    if args.stats:
        total = conn.execute("SELECT COUNT(*) FROM texts").fetchone()[0]
        countries = conn.execute(
            "SELECT COUNT(DISTINCT country) FROM texts").fetchone()[0]
        print(f"Texts: {total} | Countries: {countries}")
        return

    print("Fetching philosopher list from Wikidata…")
    people = fetch_philosophers()
    print(f"Found {len(people)} philosophers with Wikiquote pages.")

    existing_authors = {
        row[0] for row in conn.execute(
            "SELECT DISTINCT author FROM texts WHERE author != ''"
        )
    }
    todo = [p for p in people if p["name"] not in existing_authors]
    print(f"{len(todo)} not yet in the database.")

    if args.limit:
        todo = todo[: args.limit]

    total_added = 0

    for i, person in enumerate(todo, 1):
        try:
            quotes = fetch_quotes(person["page"])
        except Exception as exc:
            print(f"[{i}/{len(todo)}] {person['name']}: fetch failed ({exc})")
            time.sleep(1)
            continue

        rows = [
            (
                q,
                person["name"],
                "Wikiquote",
                person["country"],
                COUNTRY_REGION.get(person["country"], ""),
                "Philosophy",
                "English",
                "",
                "English Wikiquote (CC BY-SA)",
            )
            for q in quotes
        ]

        if rows:
            conn.executemany(
                "INSERT INTO texts (text, author, work, country, region,"
                " tradition, language, era, source)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                rows,
            )
            conn.commit()
            total_added += len(rows)

        print(f"[{i}/{len(todo)}] {person['name']} "
              f"({person['country']}): +{len(rows)} quotes")

        time.sleep(REQUEST_PAUSE)

    total = conn.execute("SELECT COUNT(*) FROM texts").fetchone()[0]
    countries = conn.execute(
        "SELECT COUNT(DISTINCT country) FROM texts").fetchone()[0]
    print(f"\nDone. Added {total_added} quotes.")
    print(f"Corpus now: {total} texts from {countries} countries.")
    print("Next time you open the app it will embed the new texts "
          "automatically (this can take a while for thousands of rows).")

    conn.close()


if __name__ == "__main__":
    main()
