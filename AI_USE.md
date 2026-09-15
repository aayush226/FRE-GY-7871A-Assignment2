# AI Use Disclosure

This project used Claude (Anthropic) as a coding assistant during development.
Fill in / edit this file honestly before submitting — it should reflect what
you actually used, not this template verbatim.

## What AI was used for
- Scaffolding the project structure (scraper modules, notebook skeleton).
- Identifying the correct federalreserve.gov URL patterns for FOMC
  statements, minutes, and speeches (verified against live pages during
  development).
- Drafting the word-list lexicon starting point and FinBERT scoring
  functions, based on the methodology in the "Parsing the Fed" reference
  deck and Doh, Kim, Yang (2021) / Doh, Song, Yang (2020, 2023).
- [Add: any help with regression setup, debugging, report drafting, etc.]

## What was NOT AI-generated / required human judgment
- Actual execution of the scraper and verification of collected data
  (the AI assistant's environment could not reach federalreserve.gov, FRED,
  or Yahoo Finance directly, so all live data collection was run and
  checked locally by [your name]).
- Expanding/curating the hawkish-dovish word list beyond the starter set.
- Interpreting regression results and forming the September forecast and
  recommendation.
- All final analysis, writing, and conclusions in the submitted report.

## Specific prompts / sessions
[Optional: link or summarize the conversation(s) used, per your instructor's
requirements.]
