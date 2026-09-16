# AI Use Disclosure

Author: Aayush Shah

This project used Claude (Anthropic) as a coding and research assistant
throughout development, working interactively in a chat session.

## What AI was used for
- Helping me figure out the the full scraper (statements, minutes, press
  conference transcripts, speeches, and testimony) against
  federalreserve.gov, including discovering the correct URL patterns and
  listing-page structures through iterative debugging.
- Editing the tone-scoring code I wrote(word list lexicon, FinBERT sentiment,
  FinBERT factor similarity), based on the methodology described in the
  "Parsing the Fed" reference deck, since the Doh et al. papers' method
  (comparing to declassified alternative FOMC statements) isn't usable
  for this 2018-2026 sample (alternative statements are only public
  through Dec 2014).
- Writing the market-data pipeline (FRED/Yahoo Finance) and the
  regression code for Table 3.
- Debugging real bugs found during development, including: a yfinance
  API version change that broke market data collection; an off-by-one
  in yfinance's exclusive end-date handling; a chair-tagging bug caused
  by pandas converting missing values to NaN; a missing release-time
  field for FOMC minutes; and — the largest gap — that testimony
  documents live at a different URL path than regular speeches
  (/newsevents/testimony/ vs /newsevents/speech/), which an earlier
  version of the scraper silently missed entirely.
- Searching for and summarizing real-time market context for the
  September 2026 forecast (CME FedWatch probabilities, a Duke/WSJ survey
  of former Fed officials, and live market commentary), used as an input
  to the forecast alongside the model's own regression output.
- Drafting an initial version of the "Comparison with the readings"
  section and the Step 4 forecast, using the notebook's actual output
  numbers, which I then reviewed and adjusted.

## What was NOT AI-generated / required my own judgment
- Most of the code cells were initially written by me which the AI then
  edited bsaed on bugs found in the code. 
- Actually executing the notebook (in Google Colab, using a GPU runtime),
  verifying outputs at each step, and catching cases where results looked
  wrong or inconsistent.
- Expanding the word-list lexicon from the initial starter set.
- The specific probabilities and expected-magnitude estimates in the
  final forecast, and the recommendation and falsification condition —
  these are my own conclusions, informed by but not dictated by the
  regression output and the AI's suggestions.
- Requesting and directing the methodology audit against the assignment
  text before finalizing, which surfaced several of the fixes listed
  above.
- All final review, editing, and submission of this report and the
  accompanying notebook.

## Model
Claude Sonnet 4.5, via claude.ai, over multiple sessions during
September 2026.