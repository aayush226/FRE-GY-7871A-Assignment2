"""
Two tone-scoring methods for FOMC text, following the "Parsing the Fed"
approach (no dependence on the 5-year-delayed alternative statements):

  1. Word list / phrase lexicon  -> score_wordlist()
  2. FinBERT sentiment           -> score_finbert()
     (plus a FinBERT "factor similarity" variant -> score_factor_similarity())

All three return a score in roughly [-1, 1] where positive = hawkish.
"""

import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Word list / phrase lexicon
# ---------------------------------------------------------------------------
# Organized by category so it's auditable and easy to extend. Weights are on
# roughly a [-2, 2] scale (magnitude = how strongly hawkish/dovish); the
# score_wordlist() function further weights by phrase length (in words), so
# longer, more specific phrases count more than single common words.
#
# Sourced from: the standard vocabulary the FOMC has actually used across
# statements/minutes 2018-2026 (rate path, inflation outlook, labor market,
# balance sheet/QE, risk-balance language, forward-guidance certainty), plus
# the "Parsing the Fed" reference deck and Doh, Kim, Yang (2021).
#
# You should still spot-check this against a few of your own scraped
# statements — pull a clearly hawkish one (e.g. a 2022 hike statement) and a
# clearly dovish one (e.g. March 2020) from docs_df and skim for phrases
# this list is missing.

RATE_PATH_PHRASES = {
    # hawkish: tightening / more hikes
    "raise the target range": 1.5,
    "increase in the target range": 1.0,
    "further increases in the target range": 1.5,
    "additional policy firming": 1.5,
    "additional gradual increases": 1.0,
    "warrant additional increases": 1.5,
    "restrictive stance of monetary policy": 1.5,
    "higher for longer": 1.5,
    "hike": 0.5,
    # dovish: cutting / easing
    "lower the target range": -1.5,
    "decrease in the target range": -1.0,
    "cut the target range": -1.5,
    "ease the stance of monetary policy": -1.5,
    "gradual adjustments in the stance": -0.3,
    "will not hesitate to act": -0.5,
    "act as appropriate to sustain the expansion": -1.0,
    "exceptionally low levels": -1.5,
}

INFLATION_PHRASES = {
    # hawkish
    "elevated inflation": 1.0,
    "inflation remains elevated": 1.2,
    "inflation remains too high": 1.5,
    "upside risks to inflation": 1.2,
    "inflation has increased": 1.0,
    "persistently high inflation": 1.5,
    "vigilant to inflation risks": 1.2,
    "strongly committed to returning inflation": 1.0,
    # dovish
    "inflation has eased": -1.2,
    "inflation has declined": -1.0,
    "inflation is easing": -1.2,
    "moderating inflation": -1.0,
    "inflation is moving sustainably toward": -1.2,
    "inflation has come down": -1.0,
    "muted inflation pressures": -1.0,
    "below the committee's longer-run objective": -0.8,
}

LABOR_MARKET_PHRASES = {
    # hawkish (strong/tight labor market -> supports tightening)
    "strong job gains": 0.8,
    "robust pace": 0.5,
    "tight labor market": 1.0,
    "labor market remains tight": 1.0,
    "job gains have been robust": 0.8,
    # dovish (weakening labor market -> supports easing)
    "labor market conditions have eased": -1.0,
    "job gains have slowed": -1.0,
    "softening labor market": -1.2,
    "labor market has cooled": -1.2,
    "downside risks to employment": -1.0,
    "unemployment rate has moved up": -0.8,
    "layoffs": -0.5,
}

GROWTH_ACTIVITY_PHRASES = {
    "robust growth": 0.8,
    "economic activity has been expanding at a solid pace": 0.5,
    "growth of economic activity has moderated": -0.8,
    "economic activity has weakened": -1.2,
    "downside risks to growth": -1.0,
    "weigh heavily on economic activity": -1.5,
    "considerable risks to the economic outlook": -1.0,
}

BALANCE_SHEET_PHRASES = {
    # hawkish: shrinking balance sheet (QT)
    "reduce the size of the federal reserve's balance sheet": 1.2,
    "reducing its holdings": 1.0,
    "resumed reducing": 1.0,
    "continuing to reduce": 1.0,
    # dovish: growing balance sheet (QE)
    "increase its holdings": -1.2,
    "expand its holdings": -1.2,
    "large-scale asset purchases": -1.0,
    "purchase additional": -1.0,
}

RISK_AND_CERTAINTY_PHRASES = {
    # hawkish-leaning certainty / hawkish risk framing
    "committee judges that risks are weighted toward higher inflation": 1.2,
    "additional firming may be appropriate": 1.2,
    # dovish-leaning uncertainty / dovish risk framing
    "increased uncertainty about the economic outlook": -0.8,
    "highly accommodative": -1.2,
    "supportive of economic activity": -0.8,
    "patient in assessing": -0.8,
    "will act as appropriate": -0.5,
    "prepared to adjust the stance": -0.5,
}

WORD_LIST = {
    **RATE_PATH_PHRASES,
    **INFLATION_PHRASES,
    **LABOR_MARKET_PHRASES,
    **GROWTH_ACTIVITY_PHRASES,
    **BALANCE_SHEET_PHRASES,
    **RISK_AND_CERTAINTY_PHRASES,
}


def score_wordlist(text: str, lexicon: dict[str, float] = None) -> float:
    """Weighted-average sentiment by scanning for lexicon phrases.
    Score = sum(sign * len(phrase_in_words)) normalized by document length,
    following the "Parsing the Fed" weighting idea (longer/more specific
    phrase matches count more).
    """
    lexicon = lexicon or WORD_LIST
    text_low = text.lower()
    total = 0.0
    n_words = max(len(text_low.split()), 1)
    for phrase, sign in lexicon.items():
        n_words_in_phrase = len(phrase.split())
        count = len(re.findall(re.escape(phrase), text_low))
        total += sign * n_words_in_phrase * count
    # normalize so score is roughly bounded in [-1, 1] for typical statements
    return float(np.tanh(total / (n_words / 50)))


# ---------------------------------------------------------------------------
# Device selection (used by both FinBERT sentiment and the embedder below)
# ---------------------------------------------------------------------------
def _get_device():
    try:
        import torch
        return 0 if torch.cuda.is_available() else -1  # 0 = first GPU, -1 = CPU
    except ImportError:
        return -1


_DEVICE = _get_device()


# ---------------------------------------------------------------------------
# 2. FinBERT sentiment (requires transformers + torch)
# ---------------------------------------------------------------------------
_finbert_pipeline = None


def _get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        from transformers import pipeline
        _finbert_pipeline = pipeline(
            "sentiment-analysis", model="ProsusAI/finbert", truncation=True,
            device=_DEVICE, batch_size=32,
        )
        where = "GPU" if _DEVICE == 0 else "CPU"
        print(f"[tone_scoring] FinBERT sentiment pipeline loaded on {where}")
    return _finbert_pipeline


def score_finbert(text: str, max_sentences: int = 200) -> float:
    """Average FinBERT sentiment across sentences.
    FinBERT returns positive/negative/neutral with a confidence score.
    We map: positive -> +score, negative -> -score, neutral -> 0.
    NOTE: for FOMC text, "positive sentiment" empirically correlates with
    hawkish tone in the reference slide deck (stronger dollar, higher
    short rates) — but sanity-check this against your own regression signs
    rather than assuming positive=hawkish outright.
    """
    pipe = _get_finbert()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    sentences = sentences[:max_sentences]
    if not sentences:
        return 0.0
    results = pipe(sentences)
    signed = []
    for r in results:
        label = r["label"].lower()
        score = r["score"]
        if label == "positive":
            signed.append(score)
        elif label == "negative":
            signed.append(-score)
        else:
            signed.append(0.0)
    return float(np.mean(signed))


# ---------------------------------------------------------------------------
# 3. FinBERT factor similarity ("Parsing the Fed" Method 1)
# ---------------------------------------------------------------------------
FACTOR_SENTENCES = {
    "rate_factor": "Interest rates will rise",
    "inflation_factor": "Inflation will rise",
}

_finbert_embed_model = None


def _get_finbert_embedder():
    """Use FinBERT's underlying transformer to get sentence embeddings
    (mean-pooled last hidden state), rather than the classification head."""
    global _finbert_embed_model
    if _finbert_embed_model is None:
        from transformers import AutoTokenizer, AutoModel
        import torch
        tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModel.from_pretrained("ProsusAI/finbert")
        model.eval()
        if _DEVICE == 0:
            model = model.to("cuda")
        _finbert_embed_model = (tok, model)
        where = "GPU" if _DEVICE == 0 else "CPU"
        print(f"[tone_scoring] FinBERT embedder loaded on {where}")
    return _finbert_embed_model


def _embed(sentences: list[str]):
    import torch
    tok, model = _get_finbert_embedder()
    device = "cuda" if _DEVICE == 0 else "cpu"
    with torch.no_grad():
        enc = tok(sentences, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        # mean-pool over tokens, masking padding
        mask = enc["attention_mask"].unsqueeze(-1)
        summed = (out.last_hidden_state * mask).sum(1)
        counts = mask.sum(1).clamp(min=1)
        return (summed / counts).cpu().numpy()


def score_factor_similarity(text: str, max_sentences: int = 200) -> dict[str, float]:
    """Cosine similarity of each document sentence to anchor factor
    sentences, averaged. Returns one score per factor."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    sentences = sentences[:max_sentences]
    if not sentences:
        return {k: 0.0 for k in FACTOR_SENTENCES}

    doc_vecs = _embed(sentences)
    factor_names = list(FACTOR_SENTENCES.keys())
    factor_vecs = _embed([FACTOR_SENTENCES[k] for k in factor_names])

    def cos_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    scores = {}
    for fname, fvec in zip(factor_names, factor_vecs):
        sims = [cos_sim(dv, fvec) for dv in doc_vecs]
        scores[fname] = float(np.mean(sims))
    return scores


# ---------------------------------------------------------------------------
# Convenience: score a whole dataframe of documents, with a progress bar and
# incremental checkpointing so a long run is never fully lost if interrupted.
# ---------------------------------------------------------------------------
def score_documents(df, text_col: str = "text", checkpoint_path: str = None,
                     checkpoint_every: int = 10):
    """df: DataFrame with a text column (must also have a 'url' column,
    used as the unique key for checkpoint/resume).

    Adds columns: wordlist_score, finbert_score, rate_factor, inflation_factor

    If checkpoint_path is given:
      - progress is saved to that CSV every `checkpoint_every` documents
      - re-running this function with the same checkpoint_path picks up
        where it left off instead of rescoring everything from scratch
      - a tqdm progress bar shows live progress + estimated time remaining

    Returns the fully scored DataFrame (same row order as the input df).
    """
    from tqdm.auto import tqdm

    df = df.copy()
    score_cols = ["wordlist_score", "finbert_score", "rate_factor", "inflation_factor"]

    already_scored = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        prev = pd.read_csv(checkpoint_path)
        for _, r in prev.iterrows():
            already_scored[r["url"]] = {c: r[c] for c in score_cols}
        print(f"Resuming from checkpoint: {len(already_scored)} documents already scored")

    for c in score_cols:
        df[c] = np.nan

    rows_to_save = list(prev.to_dict("records")) if (checkpoint_path and os.path.exists(checkpoint_path)) else []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Scoring documents"):
        url = row.get("url")
        if url in already_scored:
            for c in score_cols:
                df.at[i, c] = already_scored[url][c]
            continue

        text = row[text_col] if isinstance(row[text_col], str) else ""
        wl = score_wordlist(text)
        fb = score_finbert(text)
        factors = score_factor_similarity(text)

        df.at[i, "wordlist_score"] = wl
        df.at[i, "finbert_score"] = fb
        df.at[i, "rate_factor"] = factors["rate_factor"]
        df.at[i, "inflation_factor"] = factors["inflation_factor"]

        if checkpoint_path:
            saved_row = row.to_dict()
            saved_row.update({
                "wordlist_score": wl, "finbert_score": fb,
                "rate_factor": factors["rate_factor"],
                "inflation_factor": factors["inflation_factor"],
            })
            rows_to_save.append(saved_row)
            if len(rows_to_save) % checkpoint_every == 0:
                pd.DataFrame(rows_to_save).to_csv(checkpoint_path, index=False)

    if checkpoint_path:
        pd.DataFrame(rows_to_save).to_csv(checkpoint_path, index=False)

    return df
