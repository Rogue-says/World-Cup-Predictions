# World Cup Predictions

An experimental Python football outcome model using historical Elo, recent form, head-to-head results and rest days. It outputs win/draw/loss probabilities and a chart. Optional current bookmaker odds can be blended with the model for current or future matchups.

## Install and run

Requires Python 3.10+.

```bash
git clone https://github.com/Rogue-says/World-Cup-Predictions.git
cd World-Cup-Predictions
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python predict_today.py Portugal Uzbekistan --date 2026-06-23
```

The date above is an explicit prediction cutoff, not a claim that a verified fixture is bundled. The model assumes a neutral venue and World Cup tournament weighting for custom matchups.

On the first online run, historical results are downloaded from [martj42/international_results](https://github.com/martj42/international_results) into `data_cache/results.csv`. Internet access is required for that download. Subsequent runs reuse the cache.

```bash
# Refresh the cached results, validating before replacement:
python predict_today.py Brazil Japan --date 2026-09-10 --refresh-data
# No network access; requires an existing results CSV:
python predict_today.py Portugal Uzbekistan --date 2026-06-23 --offline
# Bash launcher selects .venv, venv, or system python3:
./wc Portugal Uzbekistan --date 2026-06-23 --offline
```

There is no bundled `fixtures.csv`. Provide `--date`, or supply `data_cache/fixtures.csv` with `teams` (for example `Portugal v Uzbekistan`) and `date_dt` (`YYYY-MM-DD`). Optional columns: `match_number`, `group`, `stadium`. Without a date or matching fixture, the program exits with an actionable error.

Charts are written beneath `predictions/<date>/`, relative to the repository rather than your shell's current directory.

## What changed

- Historical results are filtered **before all feature construction**, including final Elo. Later matches can no longer leak into a historical prediction's ratings.
- The model uses only features with historical training values. Live xG, injuries and odds are not fed into columns that contained only zeros during training.
- Current odds and squads are not requested for past dates. Historical evaluation cannot legitimately use today's API response as a past snapshot.
- Bookmaker matching supports reversed home/away order, requires complete valid decimal prices, avoids empty/substring matches, and averages available bookmaker prices.
- Removed interactive first-run key prompts and the launcher's stray example commands.
- Added clear CLI validation, offline behavior and a cache refresh option.

## Optional APIs

Copy `.env.example` to `.env` and populate only the services you need:

- `ODDS_API_KEY`: The Odds API for current three-way market prices.
- `API_FOOTBALL_KEY` or `RAPIDAPI_KEY`: API-Football squad lookup.

Other helper integrations remain in `api_config.py`; they are not all used by the main predictor. Missing keys or failed optional API requests fall back to model-only output. API quotas and coverage depend on your provider account; no free-tier availability is guaranteed here. Keys are loaded from the repository's `.env`, which must stay untracked.

## Model and evaluation

XGBoost trains on matches from 2006 up to the fixed validation boundary in 2023. Validation uses matches from that boundary up to the requested prediction cutoff and controls early stopping. The printed validation score is **not an untouched test-set estimate**, because early stopping used that window. Both partitions must have enough data, and the training partition must contain all three outcomes.

The model averages predictions in both team orientations. When eligible live odds exist, a heuristic blend gives the model 50–70% weight. A draw-threshold heuristic may choose a draw even when it is not the largest probability. Neither heuristic is proven calibration.

Do not treat old README accuracy claims or example percentages as verified performance for this revision. Run an independent chronological evaluation and retain its data snapshot before making performance claims. Backtesting code exists in `backtest.py`:

```bash
python backtest.py --help
```

Historical results may include extra-time scores; this dataset is not guaranteed to match regulation-time betting settlement. Predictions do not model injuries, tactics, lineups, player xG or all home-host advantages. Probability is not betting certainty.

## Tests

```bash
python -m unittest discover -s tests -v
```

The deterministic suite verifies reversed odds, malformed market data, invariance to future results, feature consistency, and a synthetic training/prediction cycle. CI runs these tests without API keys. Synthetic smoke tests verify operation, not predictive quality.

MIT license; see [LICENSE](LICENSE).
