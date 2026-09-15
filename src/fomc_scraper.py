"""
Scraper for FOMC press releases (statements) and meeting minutes.

Strategy
--------
The Fed publishes one "historical materials" page per year at:
    https://www.federalreserve.gov/monetarypolicy/fomchistorical{YEAR}.htm
(for years that have rolled off the main fomccalendars.htm page, i.e. more
than ~5 years old) and the current few years live directly on:
    https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm

Both pages contain, for every meeting that year, a "Statement" link
(newsevents/pressreleases/monetary{YYYYMMDD}a.htm) and a "Minutes" link
(monetarypolicy/fomcminutes{YYYYMMDD}.htm) plus the minutes release date.

This module parses those listing pages to discover every meeting date in a
year, then fetches and cleans the statement/minutes text for each date.

Run this from a machine with normal internet access (not a sandboxed
container) since it needs to reach federalreserve.gov.
"""

import io
import re
import time
import datetime as dt
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup

BASE = "https://www.federalreserve.gov"
HEADERS = {"User-Agent": "Mozilla/5.0 (research script; FRE-GY-7871 assignment)"}

STATEMENT_URL = BASE + "/newsevents/pressreleases/monetary{date}a.htm"
MINUTES_URL = BASE + "/monetarypolicy/fomcminutes{date}.htm"
HISTORICAL_YEAR_URL = BASE + "/monetarypolicy/fomchistorical{year}.htm"
CALENDAR_URL = BASE + "/monetarypolicy/fomccalendars.htm"
PRESSCONF_PDF_URL = BASE + "/mediacenter/files/FOMCpresconf{date}.pdf"

DATE_RE = re.compile(r"(monetary|fomcminutes)(\d{8})")


@dataclass
class FomcDocument:
    doc_type: str          # "statement" or "minutes"
    meeting_date: str      # YYYY-MM-DD, date of the FOMC meeting itself
    release_date: str      # YYYY-MM-DD, date the document was actually released
    release_time: str      # e.g. "2:00 p.m." (best effort; "" if unknown)
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


def _extract_meeting_dates_from_html(html: str) -> set[str]:
    soup = BeautifulSoup(html, "html.parser")
    dates = set()
    for a in soup.find_all("a", href=True):
        m = DATE_RE.search(a["href"])
        if m and m.group(1) == "monetary":
            yyyymmdd = m.group(2)
            d = f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"
            dates.add(d)
    return dates


def discover_meeting_dates(years: list[int]) -> list[str]:
    """Return a sorted list of FOMC meeting dates (YYYY-MM-DD) for the given years.

    Older years (roughly >5-6 years old) live on a per-year historical page:
        fomchistorical{year}.htm
    Recent years (currently ~2021 onward) are NOT on that page yet — they
    only live on the main calendar page:
        fomccalendars.htm
    which lists several years at once. So: try the historical page for each
    year first; for any year that 404s, fall back to parsing the main
    calendar page (fetched once and reused) and filter dates to that year.
    """
    dates = set()
    years_needing_calendar_fallback = []

    for year in years:
        url = HISTORICAL_YEAR_URL.format(year=year)
        try:
            resp = _get(url)
        except Exception as e:
            print(f"  [info] {year} not on historical page ({e}); will use fomccalendars.htm instead")
            years_needing_calendar_fallback.append(year)
            continue
        dates |= _extract_meeting_dates_from_html(resp.text)
        time.sleep(0.5)  # be polite to the server

    if years_needing_calendar_fallback:
        try:
            resp = _get(CALENDAR_URL)
            calendar_dates = _extract_meeting_dates_from_html(resp.text)
            wanted_years = {str(y) for y in years_needing_calendar_fallback}
            dates |= {d for d in calendar_dates if d[:4] in wanted_years}
        except Exception as e:
            print(f"  [warn] could not fetch {CALENDAR_URL}: {e}")

    return sorted(dates)


def fetch_statement(meeting_date: str) -> FomcDocument:
    """meeting_date: 'YYYY-MM-DD'. Fetches the post-meeting press release."""
    yyyymmdd = meeting_date.replace("-", "")
    url = STATEMENT_URL.format(date=yyyymmdd)
    resp = _get(url)
    soup = BeautifulSoup(resp.text, "html.parser")

    text, release_time = _extract_statement_text_and_time(soup)

    return FomcDocument(
        doc_type="statement",
        meeting_date=meeting_date,
        release_date=meeting_date,  # statements are released same day
        release_time=release_time,
        url=url,
        text=text,
    )


def _find_release_date_near(html: str, position: int, window: int = 300) -> str:
    """Look for a 'Released Month DD, YYYY' string within `window` characters
    on either side of `position` in the raw HTML. Handles both page layouts
    (release date can appear before or after the minutes link)."""
    start = max(0, position - window)
    end = min(len(html), position + window)
    snippet = html[start:end]
    m = re.search(r"Released\s+([A-Z][a-z]+\s+\d{1,2},\s*\d{4})", snippet)
    if not m:
        return ""
    try:
        return dt.datetime.strptime(m.group(1), "%B %d, %Y").strftime("%Y-%m-%d")
    except ValueError:
        return ""


def discover_minutes_release_dates(years: list[int]) -> dict[str, str]:
    """Returns {meeting_date: minutes_release_date} by reading the actual
    release-date text printed next to each Minutes link on the listing
    pages — the minutes documents themselves don't reliably state their own
    release date, so we capture it here instead."""
    release_dates: dict[str, str] = {}

    def _scan(html: str):
        for m in re.finditer(r"fomcminutes(\d{8})\.htm", html):
            meeting = m.group(1)
            meeting_iso = f"{meeting[0:4]}-{meeting[4:6]}-{meeting[6:8]}"
            if meeting_iso in release_dates:
                continue
            rd = _find_release_date_near(html, m.start())
            if rd:
                release_dates[meeting_iso] = rd

    years_needing_calendar_fallback = []
    for year in years:
        url = HISTORICAL_YEAR_URL.format(year=year)
        try:
            resp = _get(url)
        except Exception:
            years_needing_calendar_fallback.append(year)
            continue
        _scan(resp.text)
        time.sleep(0.5)

    if years_needing_calendar_fallback:
        try:
            resp = _get(CALENDAR_URL)
            _scan(resp.text)
        except Exception as e:
            print(f"  [warn] could not fetch {CALENDAR_URL}: {e}")

    return release_dates


def fetch_minutes(meeting_date: str, release_date_override: str | None = None) -> FomcDocument:
    """meeting_date: 'YYYY-MM-DD'. Fetches the minutes (released ~3 weeks later).
    Pass release_date_override (from discover_minutes_release_dates()) to get
    the correct release date — the minutes page itself doesn't reliably state
    it, so leaving this unset will likely leave release_date blank."""
    yyyymmdd = meeting_date.replace("-", "")
    url = MINUTES_URL.format(date=yyyymmdd)
    resp = _get(url)
    soup = BeautifulSoup(resp.text, "html.parser")

    text = _extract_main_article_text(soup)
    release_date, release_time = _extract_release_datetime(resp.text)
    if not release_date and release_date_override:
        release_date = release_date_override

    return FomcDocument(
        doc_type="minutes",
        meeting_date=meeting_date,
        release_date=release_date or "",
        release_time=release_time or "",
        url=url,
        text=text,
    )


def _extract_statement_text_and_time(soup: BeautifulSoup) -> tuple[str, str]:
    """Statements have a 'For release at H:MM p.m. EST/EDT' line near the top."""
    text = _extract_main_article_text(soup)
    m = re.search(r"For (?:immediate )?release at ([\d:]+\s*[ap]\.m\.[^\n]*)", text)
    release_time = m.group(1).strip() if m else ""
    return text, release_time


def _extract_main_article_text(soup: BeautifulSoup) -> str:
    """Best-effort extraction of the main body text, dropping nav/boilerplate."""
    # The Fed's pages put the actual content inside <div id="article"> on most
    # press release / minutes pages. Fall back to <main> or full page text.
    container = soup.find("div", id="article") or soup.find("main") or soup
    # Drop known boilerplate blocks
    for tag in container.find_all(["nav", "header", "footer", "script", "style"]):
        tag.decompose()
    paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]
    text = "\n\n".join(p for p in paragraphs if p)
    return text.strip()


def _extract_release_datetime(raw_html: str) -> tuple[str, str]:
    """Try to find 'Released Month DD, YYYY at H:MM p.m.' pattern (used on
    fomcminutes pages / linked from fomchistorical pages)."""
    m = re.search(
        r"Released\s+([A-Z][a-z]+ \d{1,2}, \d{4})(?:\s+at\s+([\d:]+\s*[ap]\.m\.))?",
        raw_html,
    )
    if not m:
        return "", ""
    try:
        d = dt.datetime.strptime(m.group(1), "%B %d, %Y").strftime("%Y-%m-%d")
    except ValueError:
        d = ""
    t = m.group(2) or ""
    return d, t


def fetch_press_conference(meeting_date: str) -> FomcDocument:
    """meeting_date: 'YYYY-MM-DD'. Fetches and extracts text from the FOMC
    press conference transcript PDF.

    Note: not every meeting has a press conference (only the 8 regularly
    scheduled ones with SEP materials in recent years; historically only
    the March/June/Sept/Dec meetings had them before 2019, after which
    every meeting got one). A 404 here just means that meeting had no
    press conference — same as the minutes 404s you've already seen.
    """
    import pdfplumber  # local import so the rest of the module works
                        # even if pdfplumber isn't installed yet

    yyyymmdd = meeting_date.replace("-", "")
    url = PRESSCONF_PDF_URL.format(date=yyyymmdd)
    resp = _get(url)

    text_parts = []
    with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    text = "\n\n".join(text_parts).strip()

    return FomcDocument(
        doc_type="press_conference",
        meeting_date=meeting_date,
        release_date=meeting_date,  # press conferences happen same day as statement
        release_time="2:30 p.m.",   # standard time; not always exact, but close
        url=url,
        text=text,
    )


def collect_all_press_conferences(meeting_dates: list[str], sleep_between: float = 1.0) -> list[FomcDocument]:
    """Fetch press conference transcripts for a list of meeting dates
    (reuse the dates discovered by discover_meeting_dates() /
    already present in your fomc_docs list)."""
    docs = []
    for d in meeting_dates:
        try:
            docs.append(fetch_press_conference(d))
            print(f"  presconf  {d} ok")
        except Exception as e:
            print(f"  [info] presconf {d} skipped (likely no press conference that meeting): {e}")
        time.sleep(sleep_between)
    return docs


def collect_all(years: list[int], sleep_between: float = 1.0,
                 include_press_conferences: bool = True) -> tuple[list[FomcDocument], list[str]]:
    """Full pipeline: discover meeting dates, then fetch statement + minutes
    (+ press conference transcripts, by default) for each.
    Returns (docs, meeting_dates) — the dates are returned too so you don't
    have to call discover_meeting_dates() again elsewhere."""
    docs: list[FomcDocument] = []
    meeting_dates = discover_meeting_dates(years)
    print(f"Discovered {len(meeting_dates)} meeting dates across {years}")

    minutes_release_dates = discover_minutes_release_dates(years)
    print(f"Discovered {len(minutes_release_dates)} minutes release dates")

    for d in meeting_dates:
        try:
            docs.append(fetch_statement(d))
            print(f"  statement {d} ok")
        except Exception as e:
            print(f"  [warn] statement {d} failed: {e}")
        time.sleep(sleep_between)

        try:
            docs.append(fetch_minutes(d, release_date_override=minutes_release_dates.get(d)))
            print(f"  minutes   {d} ok")
        except Exception as e:
            print(f"  [warn] minutes {d} failed: {e}")
        time.sleep(sleep_between)

    if include_press_conferences:
        docs.extend(collect_all_press_conferences(meeting_dates, sleep_between=sleep_between))

    return docs, meeting_dates
