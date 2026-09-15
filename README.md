# FOMC Communications & Asset Prices — Assignment 2

FRE-GY 7871 A, NLP and the Investment Process, Fall 2026

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Optional: get a free FRED API key at
https://fred.stlouisfed.org/docs/api/api_key.html and paste it into the
`FRED_API_KEY` variable in the notebook. If you skip this, `market_data.py`
falls back to FRED's no-auth CSV endpoint, which works fine for this
assignment's series.

## Run

1. `python build_notebook.py` — (re)generates `notebooks/assignment2.ipynb`
   from `build_notebook.py` if you edit the cell definitions there. You can
   also just open and edit the `.ipynb` directly in Jupyter.
2. `jupyter notebook notebooks/assignment2.ipynb`
3. Run all cells top to bottom. Step 1 (scraping) and the FinBERT model
   downloads will take a while the first time — be patient and don't rerun
   from scratch unnecessarily (the scraper is polite/rate-limited on
   purpose so as not to hammer federalreserve.gov).

## Project structure

```
src/
  fomc_scraper.py      statements + minutes scraper
  speech_scraper.py     Chair speeches/testimony scraper
  market_data.py         FRED + Yahoo Finance indicator panel
  tone_scoring.py        word list + FinBERT tone scoring
  analysis.py             regressions + forecast scaffold
notebooks/
  assignment2.ipynb      main notebook (run this)
data/                    scraped/derived data (not committed — see .gitignore)
outputs/                 saved figures
build_notebook.py       generates the notebook from cell definitions
AI_USE.md                 AI use disclosure (edit before submitting)
```

## Notes

- Alternative FOMC statements (Alt. A/B/C/D), used by Doh, Kim, Yang (2021)
  and Doh, Song, Yang (2020/2023), are declassified only after a 5-year lag
  (latest available: Dec 2014 as of Nov 2020; check current status). They
  are **not usable** for this assignment's 2018–2026 window, so tone
  scoring here follows the "Parsing the Fed" methodology instead (word
  list + FinBERT), which needs no alternative statements.
- The scraper discovers meeting dates automatically from
  `federalreserve.gov/monetarypolicy/fomchistorical{year}.htm` rather than
  hardcoding dates, so it should keep working as new meetings are added.
- No data files are committed per the assignment instructions — `data/` is
  gitignored. Only the notebook (with output saved), code, and this
  disclosure go in the repo.
