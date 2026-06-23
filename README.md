# World Cup Predictions

Predicts the outcome of any 2026 World Cup match — win / draw / loss probabilities, full squad data, live betting odds, plus a chart you can post. You give it two teams, it does the rest.

**Two ways to use:**
1. **CSV Mode (no API keys)** — Run instantly with historical data only
2. **API Mode (free keys)** — Adds live odds + squad data for better predictions

## Quick Start

```bash
git clone https://github.com/Rogue-says/world_cup_predictions.git
cd world_cup_predictions
pip install -r requirements.txt
```

**Option 1: CSV Mode (works immediately, no API keys needed)**
```bash
python predict_today.py "Portugal" "Uzbekistan"
```

**Option 2: API Mode (better predictions, free API keys required)**
```bash
# Add your free API keys to .env file first
python predict_today.py "Portugal" "Uzbekistan"
```

Works on Windows, Mac, Linux. No Docker, no special setup.

## Sample Output

```
┌─── PORTUGAL SQUAD (26 players) ─────────────────────────────┐
│  Goalkeepers:
│    #1   Diogo Costa               Age: 26
│  Attackers:
│    #7   Cristiano Ronaldo         Age: 41
└──────────────────────────────────────────────────────────────┘

============================================================
  Portugal vs Uzbekistan
  2026-06-23  ·  Group K  ·  Houston Stadium
============================================================
  Portugal               win    78.6%
  Draw                          14.0%
  Uzbekistan             win     7.4%
------------------------------------------------------------
  PICK: Portugal  (78.6%)   [LOCK]
============================================================
```

## How It Works

The model combines historical CSV data with live API data:

### From CSV Data (always works, no API needed)
- **Elo ratings.** Computed from every international result since 2006. Each team starts at 1500 and trades points after every game.
- **Recent form.** Win rate and goal difference over last 5 and 10 matches.
- **Head-to-head.** Historical record between the two teams.
- **Rest days.** How long since each team last played.

### From Free APIs (optional, enhances prediction)
- **Live betting odds** from 25+ bookmakers (blended 65% with model)
- **Full squad data** — player names, positions, jersey numbers, age

### The Model
An **XGBoost** classifier trained on matches before the prediction date. On validation data it achieves **60% accuracy** (log-loss 0.86 vs 1.05 baseline).

## Free API Keys (Optional)

**Important: Each user must get their own API keys.** The prediction works without any API keys. Adding free keys gives you live odds and squad data.

### The Odds API (Recommended)
Get live betting odds from 25+ bookmakers.
1. Go to https://the-odds-api.com
2. Sign up (free, no credit card)
3. Copy your API key
4. Add to `.env`: `ODDS_API_KEY=your_key_here`
5. Free tier: 500 requests/month

### API-Football
Get squad data, lineups, injuries, and match stats.
1. Go to https://dashboard.api-football.com/register
2. Sign up (free, no credit card)
3. Go to Account → My Access
4. Copy your API key
5. Add to `.env`: `API_FOOTBALL_KEY=your_key_here`
6. Free tier: 100 requests/day

### The Rundown
Live scores and events.
1. Go to https://therundown.io/api
2. Sign up and get your API key
3. Add to `.env`: `THERUNDOWN_KEY=your_key_here`

## Rate Limits

If you hit API rate limits, the predictor automatically falls back to CSV-only mode:

| Mode | API Keys Needed | What You Get | Prediction |
|------|----------------|--------------|------------|
| **CSV Mode** | None | Historical data only | ~70% confidence |
| **API Mode** | Free keys | Squads + live odds + blended | ~78% confidence |

CSV mode never breaks, never errors out.

## API Configuration

Edit the `.env` file to add your keys:

```
ODDS_API_KEY=your_key_here
API_FOOTBALL_KEY=your_key_here
THERUNDOWN_KEY=your_key_here
```

Leave a key blank to disable that API.

## Project Structure

```
├── predict_today.py   # Main script
├── api_config.py      # API integrations
├── data_cache/        # Historical CSV data
├── .env               # Your API keys
├── requirements.txt   # Python dependencies
├── predictions/       # Output charts
└── venv/              # Virtual environment
```

## What It Doesn't Do

- No injuries or suspensions (API-Football paid plan needed)
- No expected goals (xG) — the stat that moves modern soccer models
- No lineups or tactics before match
- Draws are under-predicted (like most models)

## Data

Historical results from [martj42/international_results](https://github.com/martj42/international_results). Fixtures are the official 2026 schedule.

## License

MIT — do whatever you want with it.
