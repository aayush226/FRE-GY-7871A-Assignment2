"""
Scraper for the Fed Chair's speeches and testimony. (Press conference
transcripts are handled separately in fomc_scraper.py, since they're PDFs
keyed by FOMC meeting date rather than by a speech/testimony listing page.)

Chairs in scope for this assignment:
    Jerome Powell  -> Feb 2018 through May 21, 2026
    Kevin Warsh    -> May 22, 2026 onward

Speech listing pages:   https://www.federalreserve.gov/newsevents/{YEAR}-speeches.htm
  -> entries link to:   https://www.federalreserve.gov/newsevents/speech/{lastname}{YYYYMMDD}a.htm

Testimony listing pages live at a DIFFERENT path than speeches — easy to
miss, and a real gap in an earlier version of this scraper. Recent years
use a hyphenated filename, older years don't:
    https://www.federalreserve.gov/newsevents/testimony/{YEAR}-testimony.htm   (newer)
    https://www.federalreserve.gov/newsevents/testimony/{YEAR}testimony.htm    (older)
  -> entries link to:   https://www.federalreserve.gov/newsevents/testimony/{lastname}{YYYYMMDD}a.htm

Release time: neither speeches nor testimony have a fixed release-time
convention (unlike statements/minutes/press conferences, which are always
2:00-2:30pm ET) — they happen at whatever time the actual event occurs,
anywhere from an 8am breakfast talk to an evening dinner keynote. The HTML
page itself usually doesn't state a clock time, but the companion PDF
almost always has a "For release on delivery H:MM a.m./p.m." header, so we
fetch that PDF just to extract the time.
"""

import io
import re
import time
import datetime as dt
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

BASE = "https://www.federalreserve.gov"
HEADERS = {"User-Agent": "Mozilla/5.0 (research script; FRE-GY-7871 assignment)"}

SPEECH_LIST_URL = BASE + "/newsevents/{year}-speeches.htm"
TESTIMONY_LIST_URL_NEW = BASE + "/newsevents/testimony/{year}-testimony.htm"
TESTIMONY_LIST_URL_OLD = BASE + "/newsevents/testimony/{year}testimony.htm"

# Powell -> chair Feb 5 2018 to May 21 2026 (last day before Warsh sworn in)
# Warsh  -> chair from May 22 2026 onward
CHAIR_PERIODS = [
    ("powell", dt.date(2018, 2, 5), dt.date(2026, 5, 21)),
    ("warsh", dt.date(2026, 5, 22), dt.date(2099, 1, 1)),
]

# Last names as they appear in speech/testimony URLs
CHAIR_LASTNAMES = {"powell": "powell", "warsh": "warsh"}


@dataclass
class SpeechDocument:
    doc_type: str        # "speech" or "testimony"
    speaker: str          # "powell" or "warsh"
    release_date: str     # YYYY-MM-DD
    release_time: str     # e.g. "10:00 a.m." — best effort, "" if not found
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


def _discover_links(list_urls_by_year, url_regex, years: list[int]) -> list[tuple[str, str, str]]:
    """Shared discovery logic for both speeches and testimony.
    list_urls_by_year: function(year) -> list of candidate listing URLs to try in order."""
    out = []
    for year in years:
        html = None
        for candidate_url in list_urls_by_year(year):
            try:
                html = _get(candidate_url).text
                break
            except Exception:
                continue
        if html is None:
            print(f"  [warn] could not fetch listing page for {year}")
            continue

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            m = re.search(url_regex, href)
            if not m:
                continue
            lastname, yyyymmdd = m.group(1), m.group(2)
            d = dt.date(int(yyyymmdd[0:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]))
            chair_on_date = _chair_for_date(d)
            if chair_on_date is None or CHAIR_LASTNAMES[chair_on_date] != lastname:
                continue  # e.g. skip a Powell item given while just a Governor, pre-2018
            full_url = href if href.startswith("http") else BASE + href
            out.append((chair_on_date, d.isoformat(), full_url))
        time.sleep(0.5)
    return sorted(set(out), key=lambda t: t[1])


def discover_chair_speech_links(years: list[int]) -> list[tuple[str, str, str]]:
    return _discover_links(
        lambda year: [SPEECH_LIST_URL.format(year=year)],
        r"/speech/(powell|warsh)(\d{8})[a-z]?\.htm",
        years,
    )


def discover_chair_testimony_links(years: list[int]) -> list[tuple[str, str, str]]:
    return _discover_links(
        lambda year: [
            TESTIMONY_LIST_URL_NEW.format(year=year),
            TESTIMONY_LIST_URL_OLD.format(year=year),
        ],
        r"/testimony/(powell|warsh)(\d{8})[a-z]?\.htm",
        years,
    )


def _fetch_release_time_from_pdf(html_url: str) -> str:
    """Best-effort: fetch the companion PDF (same filename, under a /files/
    subfolder) and look for 'For release on delivery H:MM a.m./p.m.' on the
    first page. Returns "" if anything about this fails — this is a nice-to-
    have, not something that should ever crash the main scrape."""
    import pdfplumber

    pdf_url = re.sub(r"/(speech|testimony)/", r"/\1/files/", html_url)
    pdf_url = re.sub(r"\.htm$", ".pdf", pdf_url)
    try:
        resp = _get(pdf_url, retries=1)
        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
            first_page_text = pdf.pages[0].extract_text() or ""
        m = re.search(
            r"release (?:on delivery|at)\s+([\d:]+\s*[ap]\.m\.[^\n(]*)",
            first_page_text, re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return ""


def _fetch_doc(doc_type: str, speaker: str, date: str, url: str) -> SpeechDocument:
    resp = _get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    title_tag = soup.find("h3") or soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    container = soup.find("div", id="article") or soup.find("main") or soup
    for tag in container.find_all(["nav", "header", "footer", "script", "style"]):
        tag.decompose()
    paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]
    text = "\n\n".join(p for p in paragraphs if p)

    release_time = _fetch_release_time_from_pdf(url)

    return SpeechDocument(
        doc_type=doc_type,
        speaker=speaker,
        release_date=date,
        release_time=release_time,
        title=title,
        url=url,
        text=text.strip(),
    )


def fetch_speech(speaker: str, date: str, url: str) -> SpeechDocument:
    return _fetch_doc("speech", speaker, date, url)


def fetch_testimony(speaker: str, date: str, url: str) -> SpeechDocument:
    return _fetch_doc("testimony", speaker, date, url)


def collect_all_speeches(years: list[int], sleep_between: float = 1.0) -> list[SpeechDocument]:
    """Despite the name (kept for backward compatibility with existing
    notebooks), this now collects BOTH speeches and testimony — testimony
    lives at a separate URL path the original version of this function
    missed entirely."""
    speech_links = discover_chair_speech_links(years)
    testimony_links = discover_chair_testimony_links(years)
    print(f"Discovered {len(speech_links)} chair speeches and "
          f"{len(testimony_links)} chair testimony documents across {years}")

    docs = []
    for speaker, date, url in speech_links:
        try:
            docs.append(fetch_speech(speaker, date, url))
            print(f"  speech    {date} {speaker}: ok")
        except Exception as e:
            print(f"  [warn] speech {date} {speaker} failed: {e}")
        time.sleep(sleep_between)

    for speaker, date, url in testimony_links:
        try:
            docs.append(fetch_testimony(speaker, date, url))
            print(f"  testimony {date} {speaker}: ok")
        except Exception as e:
            print(f"  [warn] testimony {date} {speaker} failed: {e}")
        time.sleep(sleep_between)

    return docs


# Note: Press conference transcripts are PDFs, not HTML, and are handled in
# fomc_scraper.py's fetch_press_conference() / collect_all_press_conferences(),
# since they're keyed by FOMC meeting date rather than a listing page.
