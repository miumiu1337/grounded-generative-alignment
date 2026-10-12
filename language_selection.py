
import re
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup

MIN_PAGES = 20_000
MAX_PAGES = 200_000

FINEWIKI_URL = (
    "https://huggingface.co/datasets/HuggingFaceFW/finewiki/"
    "resolve/main/language_subsets.csv"
)
GOOGLE_CLOUD_URL = "https://docs.cloud.google.com/translate/docs/languages"
GOOGLE_TRANSLATE_URL = "https://support.google.com/translate/answer/15139004?hl=en"

#constructed languages
EXCLUDED_LANGUAGES = {
    "io",   # Ido
    "ia",   # Interlingua
    "avk",  # Kotava
    "vo",   # Volapük
    "eo",   # Esperanto
    "ie",   # Interlingue (Occidental)
    "lfn",  # Lingua Franca Nova
    "nov",  # Novial
    "isv",  # Interslavic (zonal auxiliary language)
}


def normalize(name):
    return re.sub(r"\W+", " ", str(name).casefold()).strip()


def get_html(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def main():
    # load Wikipedia languages
    df = pd.read_csv(FINEWIKI_URL)
    df = df[df["pages"].between(MIN_PAGES, MAX_PAGES)].copy()

    #exclude constructed languages
    df = df[~df["subset"].isin(EXCLUDED_LANGUAGES)].copy()

    # Google Cloud translation supported languages
    tables = pd.read_html(StringIO(get_html(GOOGLE_CLOUD_URL)))
    google = next(
        table for table in tables
        if "Language" in table.columns and "ISO-639 code" in table.columns
    )

    google_names = set(google["Language"].map(normalize))

    google_codes = {
        code.strip().split()[0].lower()
        for value in google["ISO-639 code"].dropna()
        for code in str(value).split(" or ")
    }

    #load ISO 639-3 macrolanguage mappings
    MACRO_URL = (
        "https://raw.githubusercontent.com/noumar/iso639/"
        "main/iso639/iso-639-3-macrolanguages.tab"
    )

    macro = pd.read_csv(MACRO_URL, sep="\t")
    macro = macro[macro["I_Status"] == "A"]

    macro_map = dict(zip(macro["I_Id"], macro["M_Id"]))

    # additional UI Google Translate languages
    soup = BeautifulSoup(get_html(GOOGLE_TRANSLATE_URL), "html.parser")
    heading = soup.find(
        string=lambda text: text and "The new languages are:" in text
    )

    if heading is None:
        raise ValueError("Google Translate language list not found")

    language_list = heading.find_parent().find_next("ul")
    if language_list is None:
        raise ValueError("Google Translate language list not found")

    consumer_names = {
        normalize(item.get_text(" ", strip=True))
        for item in language_list.find_all("li", recursive=False)
    }

    if len(consumer_names) < 100:
        raise ValueError("Google Translate language list is incomplete")

    # keep only unsupported languages
    supported_names = google_names | consumer_names

    codes = df["subset"].astype(str).str.strip().str.lower()

    supported = (
        codes.isin(google_codes)
        | codes.map(macro_map).isin(google_codes)
        | df["name"].map(normalize).isin(supported_names)
    )

    df = df[~supported].copy()

    # results
    result = df[
        ["name", "subset", "pages"]
    ].sort_values("pages", ascending=False)

    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
