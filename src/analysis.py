"""
Regression helpers for Assignment Table 3:
  each indicator's one-day change regressed on each tone score,
  controlling for the 3-month bill change.
"""

import pandas as pd
import statsmodels.api as sm


TARGET_COLS = ["d_dxy", "d_t10y2y", "d_dgs1", "d_growth_value"]
TONE_COLS = ["wordlist_score", "finbert_score", "rate_factor", "inflation_factor"]


def run_regressions(df: pd.DataFrame, control_col: str = "d_dgs3mo") -> pd.DataFrame:
    """For every (target, tone_score) pair, run:
        target ~ const + tone_score + control
    Returns a tidy results table with coefficients, t-stats, R^2.
    """
    rows = []
    for target in TARGET_COLS:
        for tone in TONE_COLS:
            sub = df[[target, tone, control_col]].dropna()
            if len(sub) < 10:
                continue
            X = sm.add_constant(sub[[tone, control_col]])
            y = sub[target]
            model = sm.OLS(y, X).fit()
            rows.append({
                "target": target,
                "tone_score": tone,
                "n_obs": int(model.nobs),
                "coef_tone": model.params[tone],
                "t_stat_tone": model.tvalues[tone],
                "p_value_tone": model.pvalues[tone],
                "coef_control": model.params[control_col],
                "t_stat_control": model.tvalues[control_col],
                "r_squared": model.rsquared,
            })
    return pd.DataFrame(rows)


def forecast_probabilities(recent_tone_trend: pd.Series, recent_market_reaction: pd.DataFrame) -> dict:
    """Placeholder scaffold for Step 4 of the assignment (September forecast).

    This is intentionally left for you to fill in with judgment calls once
    you have the regression results — e.g.:
      - P(hawkish statement) from the recent trend/momentum in tone scores
      - P(indicator rises) from the sign+magnitude of significant regression
        coefficients applied to your expected tone surprise
      - P(cut/hold/hike) from Fed funds futures-implied probabilities
        (CME FedWatch) combined with your tone-based read

    Returns a dict you can fill in and print/format for the report.
    """
    return {
        "rate_decision": {"cut": None, "hold": None, "hike": None},
        "statement_more_hawkish_than_prev": None,
        "market_reaction": {
            "dxy": {"p_rise": None, "expected_change": None},
            "t10y2y": {"p_rise": None, "expected_change": None},
            "dgs1": {"p_rise": None, "expected_change": None},
            "growth_value": {"p_rise": None, "expected_change": None},
        },
        "recommendation": "",
        "invalidation": "",
    }
