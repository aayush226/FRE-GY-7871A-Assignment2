"""Builds notebooks/assignment2.ipynb from cell definitions below.
Run this once locally: python build_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ---------------------------------------------------------------------------
md("""# Assignment 2: Evaluating the Impact of FOMC Communications on Asset Prices
FRE-GY 7871 A — NLP and the Investment Process, Fall 2026

**Pipeline:** collect documents -> score tone (word list + FinBERT) -> pull market
data -> regress market reactions on tone -> forecast the September FOMC meeting.
""")

code("""import sys, os

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    # Edit this to your actual repo URL before running in Colab
    REPO_URL = 'https://github.com/YOUR_USERNAME/FRE-GY-7871A-Assignment2.git'
    REPO_DIR = '/content/FRE-GY-7871A-Assignment2'

    if not os.path.exists(REPO_DIR):
        !git clone {REPO_URL} {REPO_DIR}
    os.chdir(f'{REPO_DIR}/notebooks')

    # Colab already ships pandas/numpy/matplotlib/torch (with CUDA support) —
    # only install what's actually missing. Deliberately NOT reinstalling
    # torch here: pip installing it fresh can silently replace Colab's
    # GPU-enabled build with one that doesn't match the runtime's CUDA
    # version, which would push you back onto CPU without any error.
    !pip install -q -U pillow
    !pip install -q beautifulsoup4 yfinance fredapi pdfplumber tqdm transformers statsmodels

    import torch
    print('CUDA available:', torch.cuda.is_available())
    if not torch.cuda.is_available():
        print('WARNING: no GPU detected — check Runtime > Change runtime type > GPU')

sys.path.append('../src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import fomc_scraper
import speech_scraper
import market_data
import tone_scoring
import analysis

pd.set_option('display.max_columns', 50)
""")

# ---------------------------------------------------------------------------
md("""## Step 1: Collect documents
Statements + minutes (Feb 2018 - present) and Chair speeches/testimony
(Powell through May 21 2026, Warsh from May 22 2026).

This step needs normal internet access to federalreserve.gov — run it
outside any sandboxed environment.""")

code("""import os
import pickle

FORCE_RESCRAPE = False  # set True only if you want to re-hit federalreserve.gov

FOMC_CACHE = '../data/raw_fomc_docs.pkl'
SPEECH_CACHE = '../data/raw_speech_docs.pkl'

YEARS = list(range(2018, 2027))  # 2018..2026

if not FORCE_RESCRAPE and os.path.exists(FOMC_CACHE):
    with open(FOMC_CACHE, 'rb') as f:
        fomc_docs, meeting_dates = pickle.load(f)
    print(f"Loaded {len(fomc_docs)} statement/minutes/press-conference documents from cache")
else:
    # Now returns (docs, meeting_dates). include_press_conferences=True also
    # fetches and extracts text from every FOMC press conference PDF transcript
    # — a 'skipped' message for a given date is expected for meetings that had
    # no press conference, same as the minutes 404s.
    fomc_docs, meeting_dates = fomc_scraper.collect_all(YEARS, sleep_between=1.0, include_press_conferences=True)
    print(f"Collected {len(fomc_docs)} statement/minutes/press-conference documents")
    os.makedirs('../data', exist_ok=True)
    with open(FOMC_CACHE, 'wb') as f:
        pickle.dump((fomc_docs, meeting_dates), f)
""")

code("""if not FORCE_RESCRAPE and os.path.exists(SPEECH_CACHE):
    with open(SPEECH_CACHE, 'rb') as f:
        speech_docs = pickle.load(f)
    print(f"Loaded {len(speech_docs)} speech/testimony documents from cache")
else:
    speech_docs = speech_scraper.collect_all_speeches(YEARS, sleep_between=1.0)
    print(f"Collected {len(speech_docs)} speech/testimony documents")
    with open(SPEECH_CACHE, 'wb') as f:
        pickle.dump(speech_docs, f)
""")

code("""# Combine into one DataFrame with a common schema
rows = []
for d in fomc_docs:
    rows.append({
        'doc_type': d.doc_type,
        'speaker': None,
        'meeting_date': d.meeting_date,
        'release_date': d.release_date,
        'release_time': d.release_time,
        'url': d.url,
        'text': d.text,
    })
for d in speech_docs:
    rows.append({
        'doc_type': d.doc_type,
        'speaker': d.speaker,
        'meeting_date': None,
        'release_date': d.release_date,
        'release_time': d.release_time,
        'url': d.url,
        'text': d.text,
    })

docs_df = pd.DataFrame(rows)

def _infer_chair(row):
    # Prefer meeting_date (who chaired the meeting) for statements/minutes/
    # press conferences; fall back to release_date for speeches, which have
    # no meeting_date. Use pd.notna() rather than truthiness — a missing
    # meeting_date becomes NaN (a float) in the DataFrame, and bool(nan) is
    # True in Python, which would otherwise skip the fallback.
    meeting_date = row['meeting_date']
    release_date = row['release_date']
    d = meeting_date if pd.notna(meeting_date) and meeting_date else release_date
    if pd.isna(d) or not d:
        return 'powell'  # no usable date; default (shouldn't normally happen)
    return 'warsh' if str(d) >= '2026-05-22' else 'powell'

docs_df['chair'] = docs_df.apply(_infer_chair, axis=1)
docs_df.to_csv('../data/fomc_documents.csv', index=False)
docs_df.shape
""")

# ---------------------------------------------------------------------------
md("""### Table 1: Documents collected, by type and by Chair""")

code("""table1 = docs_df.groupby(['chair', 'doc_type']).size().unstack(fill_value=0)
table1
""")

# ---------------------------------------------------------------------------
md("""## Step 2: Trend analysis — score tone
Word list + FinBERT (sentiment) + FinBERT factor similarity, per the
"Parsing the Fed" methodology (no dependence on the 5-year-delayed
alternative statements, which aren't public past Dec 2014).""")

code("""# This is the slow step (FinBERT runs on ~290 documents, some quite long).
# A progress bar shows live status + ETA, and progress is checkpointed to
# CSV every 10 documents — if this gets interrupted (laptop sleep, kernel
# restart, etc.), re-running this cell picks up where it left off instead
# of rescoring everything from scratch.
SCORE_CHECKPOINT = '../data/fomc_documents_scored.csv'

scored_df = tone_scoring.score_documents(docs_df, text_col='text', checkpoint_path=SCORE_CHECKPOINT)
scored_df[['release_date', 'doc_type', 'chair', 'wordlist_score', 'finbert_score', 'rate_factor', 'inflation_factor']].head()
""")

md("""### Figure 1: Hawkish/dovish tone over time by document type
Warsh's term start (2026-05-22) marked with a vertical line.""")

code("""fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

for doc_type, grp in scored_df.dropna(subset=['release_date']).groupby('doc_type'):
    grp = grp.sort_values('release_date')
    axes[0].plot(pd.to_datetime(grp['release_date']), grp['finbert_score'], marker='o', label=doc_type, alpha=0.7)
    axes[1].plot(pd.to_datetime(grp['release_date']), grp['wordlist_score'], marker='o', label=doc_type, alpha=0.7)

for ax, title in zip(axes, ['FinBERT sentiment', 'Word list']):
    ax.axvline(pd.Timestamp('2026-05-22'), color='red', linestyle='--', label='Warsh sworn in')
    ax.set_title(f'FOMC Communication Tone Over Time ({title})')
    ax.set_ylabel('Tone score (+ hawkish / - dovish)')
    ax.legend()

plt.tight_layout()
plt.savefig('../outputs/figure1_tone_over_time.png', dpi=150)
plt.show()
""")

md("""### Powell vs. Warsh comparison""")

code("""comparison = scored_df.groupby('chair')[['wordlist_score', 'finbert_score', 'rate_factor', 'inflation_factor']].mean()
comparison
""")

# ---------------------------------------------------------------------------
md("""## Step 3: Validate market reactions
Pull the four indicators + 3-month bill control, compute one-day changes
around each release, and regress on tone scores.""")

code("""FRED_API_KEY = None  # optional: paste your free key from https://fred.stlouisfed.org/docs/api/api_key.html
# If left as None, market_data.py falls back to FRED's no-auth CSV endpoint.

panel = market_data.build_indicator_panel(start='2018-01-01', end='2026-09-14', fred_api_key=FRED_API_KEY)
panel.to_csv('../data/market_indicator_panel.csv')
panel.tail()
""")

code("""# Compute one-day changes for every scored document's release date. Passing
# release_time lets after-hours events (mainly speeches/testimony, which
# unlike statements/minutes/press conferences aren't always 2-2:30pm ET) get
# credited to the next trading day instead of the same day.
change_rows = []
for _, row in scored_df.dropna(subset=['release_date']).iterrows():
    changes = market_data.one_day_change(panel, row['release_date'], release_time=row.get('release_time'))
    if not changes:
        continue
    change_rows.append({**row.to_dict(), **changes})

market_df = pd.DataFrame(change_rows)
market_df.to_csv('../data/fomc_market_reactions.csv', index=False)
market_df.shape
""")

md("""### Table 2: One-day changes after each Warsh-era release, next to tone scores""")

code("""table2 = market_df[market_df['chair'] == 'warsh'][
    ['release_date', 'doc_type', 'wordlist_score', 'finbert_score', 'rate_factor', 'inflation_factor',
     'd_dxy', 'd_t10y2y', 'd_dgs1', 'd_growth_value']
].sort_values('release_date')
table2
""")

md("""### Table 3: Regressions — indicator change ~ tone score + 3mo bill control""")

code("""table3 = analysis.run_regressions(market_df, control_col='d_dgs3mo')
table3.sort_values(['target', 'tone_score'])
""")

# ---------------------------------------------------------------------------
md("""## Comparison with the readings
- Doh, Kim, Yang (2021) and Doh, Song, Yang (2020/2023) identify tone by
  comparing the released statement's Universal Sentence Encoder (USE)
  embedding to Board staff's alternative (Alt. A/B/C/D) statements. Those
  alternative statements are declassified only after a 5-year lag (latest
  available: Dec 2014), so they cannot be used for the 2018-2026 sample this
  assignment covers.
- Instead, this analysis follows the "Parsing the Fed" approach: a
  hand-built word list, plain FinBERT sentiment, and FinBERT-embedding
  cosine similarity to anchor sentences ("Interest rates will rise",
  "Inflation will rise") — none of which require alternative statements.
- [Fill in: how your R² / coefficient signs compare to the KC Fed papers'
  findings and to the "Parsing the Fed" slide deck's results table.]
""")

# ---------------------------------------------------------------------------
md("""## Step 4: Forecast — September 2026 FOMC meeting
Use the trend from Step 2 and the regression coefficients from Step 3 to
form the required forecast.""")

code("""forecast = analysis.forecast_probabilities(
    recent_tone_trend=scored_df.sort_values('release_date').tail(5)['finbert_score'],
    recent_market_reaction=market_df.tail(10),
)
# TODO: fill in forecast dict values based on your Step 2/3 results, then
# format as the four required forecast items:
#   Rate decision probabilities (cut/hold/hike, sum to 100%)
#   Statement tone probability (more hawkish than previous)
#   Market reaction probabilities + expected size, per indicator
#   Recommendation + what would prove you wrong
forecast
""")

md("""### Final write-up

**Rate decision:** P(cut) = ___%, P(hold) = ___%, P(hike) = ___%

**Statement tone:** P(more hawkish than previous statement) = ___%

**Market reaction:**
| Indicator | P(rises) | Expected change |
|---|---|---|
| DXY | | |
| 10s2s | | |
| 1yr Treasury | | |
| Growth-Value | | |

**Recommendation:** [one position, why the analysis supports it]

**What would prove this wrong:** [falsification condition]
""")

nb['cells'] = cells
with open('notebooks/assignment2.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Wrote notebooks/assignment2.ipynb")
