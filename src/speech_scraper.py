"""
Scraper for the Fed Chair's speeches, testimony, and FOMC press conference
transcripts.

Chairs in scope for this assignment:
    Jerome Powell  -> Feb 2018 through May 21, 2026
    Kevin Warsh    -> May 22, 2026 onward

Speech listing pages: https://www.federalreserve.gov/newsevents/{YEAR}-speeches.htm
Each entry links to a page like:
    https://www.federalreserve.gov/newsevents/speech/{lastname}{YYYYMMDD}a.htm

Press conference transcripts are PDFs at:
    https://www.federalreserve.gov/mediacenter/files/FOMCpresconf{YYYYMMDD}.pdf
(dates match FOMC meeting dates — reuse fomc_scraper.discover_meeting_dates).
"""

import re
import time
import datetime as dt
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

BASE = "https://www.federalreserve.gov"
HEADERS = {"User-Agent": "Mozilla/5.0 (research script; FRE-GY-7871 assignment)"}

SPEECH_LIST_URL = BASE + "/newsevents/{year}-speeches.htm"

# Powell -> chair Feb 5 2018 to May 21 2026 (last day before Warsh sworn in)
# Warsh  -> chair from May 22 2026 onward
CHAIR_PERIODS = [
    ("powell", dt.date(2018, 2, 5), dt.date(2026, 5, 21)),
    ("warsh", dt.date(2026, 5, 22), dt.date(2099, 1, 1)),
]

# Last names as they appear in speech URLs
CHAIR_LASTNAMES = {"powell": "powell", "warsh": "warsh"}


@dataclass
class SpeechDocument:
    doc_type: str       # "speech" or "testimony"
    speaker: str         # "powell" or "warsh"
    release_date: str    # YYYY-MM-DD
    title: str
    url: str
    text: str = ""


def _get(url: str, retries: int = 3, pause: float = 1.0) -> requests.Response:
    last_exc = None
    for _ in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            if resp.status_code == 200:
                return resp
            last_exc = RuntimeError(f"HTTP {resp.status_code} for {url}")
        except requests.RequestException as e:
            last_exc = e
        time.sleep(pause)
    raise last_exc


def _chair_for_date(d: dt.date) -> str | None:
    for name, start, end in CHAIR_PERIODS:
        if start <= d <= end:
            return name
    return None


def discover_chair_speech_links(years: list[int]) -> list[tuple[str, str, str]]:
    """Return list of (speaker, date 'YYYY-MM-DD', url) for speeches/testimony
    given by whoever was Chair on that date, restricted to speech URLs whose
    filename starts with the chair's last name (powell... / warsh...)."""
    out = []
    for year in years:
        url = SPEECH_LIST_URL.format(year=year)
        try:
            resp = _get(url)
        except Exception as e:
            print(f"  [warn] could not fetch {url}: {e}")
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            m = re.search(r"/speech/(powell|warsh)(\d{8})[a-z]?\.htm", href)
            if not m:
                continue
            lastname, yyyymmdd = m.group(1), m.group(2)
            d = dt.date(int(yyyymmdd[0:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]))
            chair_on_date = _chair_for_date(d)
            if chair_on_date is None or CHAIR_LASTNAMES[chair_on_date] != lastname:
                continue  # e.g. skip a Powell speech given as a governor pre-2018
            full_url = href if href.startswith("http") else BASE + href
            out.append((chair_on_date, d.isoformat(), full_url))
        time.sleep(0.5)
    return sorted(set(out), key=lambda t: t[1])


def fetch_speech(speaker: str, date: str, url: str) -> SpeechDocument:
    resp = _get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    title_tag = soup.find("h3") or soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    container = soup.find("div", id="article") or soup.find("main") or soup
    for tag in container.find_all(["nav", "header", "footer", "script", "style"]):
        tag.decompose()
    paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]
    text = "\n\n".join(p for p in paragraphs if p)

    return SpeechDocument(
        doc_type="speech",
        speaker=speaker,
        release_date=date,
        title=title,
        url=url,
        text=text.strip(),
    )


def collect_all_speeches(years: list[int], sleep_between: float = 1.0) -> list[SpeechDocument]:
    links = discover_chair_speech_links(years)
    print(f"Discovered {len(links)} chair speeches/testimony across {years}")
    docs = []
    for speaker, date, url in links:
        try:
            docs.append(fetch_speech(speaker, date, url))
            print(f"  {date} {speaker}: ok")
        except Exception as e:
            print(f"  [warn] {date} {speaker} failed: {e}")
        time.sleep(sleep_between)
    return docs


# Note: Press conference transcripts are PDFs, not HTML. Use fomc_scraper's
# discovered meeting dates and fetch:
#   https://www.federalreserve.gov/mediacenter/files/FOMCpresconf{YYYYMMDD}.pdf
# Extracting text from these requires a PDF text extractor (e.g. pdfplumber).
# See notebooks/01_data_collection.ipynb for that step.
