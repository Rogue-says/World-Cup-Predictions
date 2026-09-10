"""
predict_today.py — World Cup 2026 daily match predictor
========================================================

Trains the Elo + form + head-to-head XGBoost model ONCE, then predicts every
match on a given day and saves a branded win-probability chart for each.

HOW TO USE — one game at a time
-------------------------------
    python3 predict_today.py "Saudi Arabia" "Uruguay"
    python3 predict_today.py Spain "Cabo Verde"
    python3 predict_today.py            # then type the two teams when prompted

Team order doesn't matter. The match date, group and stadium are looked up
automatically from data_cache/fixtures.csv, so you only type the two teams.

It prints the win / draw / win probabilities, the pick, and a tag
(LOCK / LEAN / TOSS-UP, plus ⚠️ UPSET PICK), and saves one branded reel chart to:
    predictions/<date>/viz_<Home>_vs_<Away>.png
"""

import os
import argparse
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss, classification_report
from api_config import (
    get_odds, get_match_odds, odds_to_implied_probs,
    get_xg_api_football,
    get_injuries_api_football, get_team_id_api_football,
    print_api_status, API_FOOTBALL_KEY, RAPIDAPI_KEY
)



ROOT = Path(__file__).resolve().parent
CACHE_DIR = str(ROOT / "data_cache")
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
FIXTURES_PATH = os.path.join(CACHE_DIR, "fixtures.csv")

# normalizes the historical results.csv team names
NAME_MAP = {
    "USA": "United States", "Korea Republic": "South Korea",
    "Republic of Ireland": "Ireland", "Türkiye": "Turkey",
    "Cape Verde": "Cabo Verde", "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic", "Curaçao": "Curacao",
    "Congo DR": "DR Congo", "Congo": "Republic of the Congo",
}

# maps fixtures.csv team names -> the normalized results.csv names
FIXTURE_NAME_MAP = {
    "IR Iran": "Iran", "Korea Republic": "South Korea", "Türkiye": "Turkey",
    "Congo DR": "DR Congo", "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic", "Curaçao": "Curacao", "USA": "United States",
    "Cape Verde": "Cabo Verde",
}

FEATURES = [
    "neutral", "tournament_weight", "home_elo", "away_elo", "elo_diff",
    "home_win5", "away_win5", "home_gd5", "away_gd5",
    "home_win10", "away_win10", "home_rest_days", "away_rest_days",
    "h2h_n", "h2h_home_winrate", "h2h_home_gd",
]

TRAIN_START = "2006-01-01"
VAL_START = "2023-01-01"
MATCH_WEIGHT = 4          # FIFA World Cup
MATCH_NEUTRAL = True      # 2026 group games at neutral US/CA/MX venues for these teams

ELO_BASE = 1500.0
ELO_K = 32
ELO_HOME_BONUS = 60

# palette (matches your existing reel charts)
INK, MUTE, GRID = "#1a1a2e", "#8a8a9e", "#e8e8ee"
ORANGE, BLUE, GRAY = "#ff6b18", "#1f6feb", "#9aa0a6"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": MUTE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.titlecolor": INK,
    "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})


# ── data loading ────────────────────────────────────────────────────────────────
def fetch_results():
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "results.csv")
    if not os.path.exists(path):
        resp = requests.get(RESULTS_URL, timeout=120)
        resp.raise_for_status()
        with open(path, "wb") as fh:
            fh.write(resp.content)
    return pd.read_csv(path)


def normalize_country(name):
    return NAME_MAP.get(name, name) if isinstance(name, str) else name


def load_results():
    r = fetch_results()
    r["home_team"] = r["home_team"].map(normalize_country)
    r["away_team"] = r["away_team"].map(normalize_country)
    r["date"] = pd.to_datetime(r["date"])
    r = r.dropna(subset=["home_score", "away_score"]).copy()
    r["home_score"] = r["home_score"].astype(int)
    r["away_score"] = r["away_score"].astype(int)
    r["neutral"] = r["neutral"].astype(str).str.upper().eq("TRUE").astype(int)
    return r.sort_values("date").reset_index(drop=True)


# ── feature engineering ─────────────────────────────────────────────────────────
def tournament_weight(name):
    t = str(name).lower()
    if "fifa world cup" in t and "qualif" not in t:
        return 4
    if "qualif" in t:
        return 3
    big = ["uefa nations", "copa america", "afc asian cup", "africa cup",
           "concacaf", "uefa euro", "confederations"]
    if any(tok in t for tok in big):
        return 3
    if "friendly" in t:
        return 1
    return 2


def add_label_and_context(r):
    r = r.copy()
    r["label"] = np.where(r["home_score"] > r["away_score"], 0,
                          np.where(r["home_score"] == r["away_score"], 1, 2))
    r["tournament_weight"] = r["tournament"].map(tournament_weight)
    return r


def compute_elo(r):
    r = r.sort_values("date").reset_index(drop=True)
    rating, home_pre, away_pre = {}, np.zeros(len(r)), np.zeros(len(r))
    for i, row in r.iterrows():
        rh = rating.get(row.home_team, ELO_BASE)
        ra = rating.get(row.away_team, ELO_BASE)
        home_pre[i], away_pre[i] = rh, ra
        bonus = 0 if row.neutral == 1 else ELO_HOME_BONUS
        exp_home = 1 / (1 + 10 ** (-((rh + bonus) - ra) / 400))
        score_home = 1.0 if row.label == 0 else (0.5 if row.label == 1 else 0.0)
        margin = abs(int(row.home_score) - int(row.away_score))
        mult = np.log(max(margin, 1) + 1) * (2.2 / (abs(rh - ra) * 0.001 + 2.2))
        rating[row.home_team] = rh + ELO_K * mult * (score_home - exp_home)
        rating[row.away_team] = ra + ELO_K * mult * ((1 - score_home) - (1 - exp_home))
    r["home_elo"], r["away_elo"] = home_pre, away_pre
    r["elo_diff"] = home_pre - away_pre
    return r, rating


def per_team_long(r):
    home = pd.DataFrame({"date": r["date"].values, "team": r["home_team"].values,
                         "opp": r["away_team"].values, "gf": r["home_score"].values,
                         "ga": r["away_score"].values})
    away = pd.DataFrame({"date": r["date"].values, "team": r["away_team"].values,
                         "opp": r["home_team"].values, "gf": r["away_score"].values,
                         "ga": r["home_score"].values})
    long = pd.concat([home, away], ignore_index=True)
    long["result"] = np.where(long["gf"] > long["ga"], 1.0,
                              np.where(long["gf"] == long["ga"], 0.5, 0.0))
    long["gd"] = long["gf"] - long["ga"]
    return long


def add_form_features(r):
    long = per_team_long(r).sort_values(["team", "date"]).reset_index(drop=True)
    long["prev_date"] = long.groupby("team")["date"].shift(1)
    long["result_lag"] = long.groupby("team")["result"].shift(1)
    long["gd_lag"] = long.groupby("team")["gd"].shift(1)
    long["win5"] = long.groupby("team")["result_lag"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    long["gd5"] = long.groupby("team")["gd_lag"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    long["win10"] = long.groupby("team")["result_lag"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    long["rest_days"] = (long["date"] - long["prev_date"]).dt.days
    form = long[["date", "team", "win5", "gd5", "win10", "rest_days"]].drop_duplicates(["date", "team"])
    r = r.merge(form.rename(columns={"team": "home_team", "win5": "home_win5", "gd5": "home_gd5",
                                     "win10": "home_win10", "rest_days": "home_rest_days"}),
                on=["date", "home_team"], how="left")
    r = r.merge(form.rename(columns={"team": "away_team", "win5": "away_win5", "gd5": "away_gd5",
                                     "win10": "away_win10", "rest_days": "away_rest_days"}),
                on=["date", "away_team"], how="left")
    return r


def add_h2h_features(r):
    long = per_team_long(r).sort_values(["team", "opp", "date"]).reset_index(drop=True)
    g = long.groupby(["team", "opp"])
    long["h2h_n"] = g.cumcount()
    long["h2h_winrate"] = g["result"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    long["h2h_gd"] = g["gd"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    h2h = long[["date", "team", "opp", "h2h_n", "h2h_winrate", "h2h_gd"]].drop_duplicates(["date", "team", "opp"])
    r = r.merge(h2h.rename(columns={"team": "home_team", "opp": "away_team",
                                    "h2h_winrate": "h2h_home_winrate", "h2h_gd": "h2h_home_gd"}),
                on=["date", "home_team", "away_team"], how="left")
    return r


def build_dataset(r):
    r = add_label_and_context(r)
    r, final_elo = compute_elo(r)
    r = add_form_features(r)
    r = add_h2h_features(r)
    for col in ["odds_home_implied", "odds_draw_implied", "odds_away_implied",
                "home_xg_for", "away_xg_for", "home_xg_against", "away_xg_against",
                "home_injuries_count", "away_injuries_count"]:
        r[col] = 0.0
    return r, final_elo


# ── model ───────────────────────────────────────────────────────────────────────
def split_by_date(ds, train_start, val_start, cutoff):
    train = ds[(ds["date"] >= pd.Timestamp(train_start)) & (ds["date"] < pd.Timestamp(val_start))].copy()
    val = ds[(ds["date"] >= pd.Timestamp(val_start)) & (ds["date"] < pd.Timestamp(cutoff))].copy()
    return train, val


def class_sample_weights(labels):
    """Sqrt-inverse-frequency weights so draws are not drowned out by wins.

    Backtest v1 used full inverse-frequency (4x boost for draws) which over-
    corrected: draw recall went from 0.7% -> 66.6% but overall accuracy dropped
    from 60.5% -> 49.4% because the model became too draw-happy.

    Sqrt-based weights give a softer ~2x boost to draws, which is the standard
    compromise between correcting class imbalance and maintaining win accuracy.
    """
    counts = np.bincount(labels, minlength=3).astype(float)
    counts = np.where(counts == 0, 1, counts)  # avoid div-by-zero
    total = counts.sum()
    # sqrt of inverse frequency — softer than full inverse, harsher than none
    weight_per_class = np.sqrt(total / (3.0 * counts))
    return weight_per_class[labels]


def train_model(train, val):
    if train.empty or val.empty:
        raise ValueError('Not enough historical data for separate training and validation windows')
    if set(train['label'].astype(int)) != {0, 1, 2}:
        raise ValueError('Training data must contain home wins, draws and away wins')
    X_train, y_train = train[FEATURES].astype(float), train["label"].astype(int)
    X_val, y_val = val[FEATURES].astype(float), val["label"].astype(int)
    # Pass sample weights so draws get equal attention during training
    sw_train = class_sample_weights(y_train.values)
    model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3, n_estimators=600,
        learning_rate=0.05, max_depth=5, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=1.0, eval_metric="mlogloss", early_stopping_rounds=50,
        tree_method="hist", n_jobs=-1, random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sw_train,
              eval_set=[(X_val, y_val)], verbose=False)
    return model, X_val, y_val


def evaluate(model, X_val, y_val):
    proba = model.predict_proba(X_val)
    pred = proba.argmax(axis=1)
    base = np.tile(np.bincount(y_val, minlength=3) / len(y_val), (len(y_val), 1))
    print(f"  Validation accuracy : {accuracy_score(y_val, pred):.3f}")
    print(f"  Validation log-loss : {log_loss(y_val, proba, labels=[0,1,2]):.3f}  (baseline {log_loss(y_val, base, labels=[0,1,2]):.3f})")


# ── prediction helpers ──────────────────────────────────────────────────────────
def form_as_of(long, team, asof):
    sub = long[(long["team"] == team) & (long["date"] < pd.Timestamp(asof))].sort_values("date")
    if len(sub) == 0:
        return {"win5": 0.5, "gd5": 0.0, "win10": 0.5, "rest_days": 30.0}
    l5, l10 = sub.tail(5), sub.tail(10)
    return {"win5": float(l5["result"].mean()), "gd5": float((l5["gf"] - l5["ga"]).mean()),
            "win10": float(l10["result"].mean()),
            "rest_days": float((pd.Timestamp(asof) - sub["date"].max()).days)}


def h2h_as_of(long, team, opp, asof):
    sub = long[(long["team"] == team) & (long["opp"] == opp) & (long["date"] < pd.Timestamp(asof))]
    if len(sub) == 0:
        # Return neutral defaults instead of NaN — NaN was silently skewing XGBoost predictions.
        return 0.0, 0.5, 0.0
    return float(len(sub)), float(sub["result"].mean()), float(sub["gd"].mean())


def build_match_row(long, final_elo, home, away, neutral, weight, asof, match_odds=None):
    """Build a single feature row for the model.
    
    `match_odds` should be a pre-fetched odds dict (home/draw/away keys) so
    we don't refetch odds on every call (was fetched twice before — bug fix).
    """
    hf, af = form_as_of(long, home, asof), form_as_of(long, away, asof)
    he, ae = final_elo.get(home, ELO_BASE), final_elo.get(away, ELO_BASE)
    n, wr, gd = h2h_as_of(long, home, away, asof)

    row = {
        "neutral": int(neutral), "tournament_weight": weight,
        "home_elo": he, "away_elo": ae, "elo_diff": he - ae,
        "home_win5": hf["win5"], "away_win5": af["win5"],
        "home_gd5": hf["gd5"], "away_gd5": af["gd5"],
        "home_win10": hf["win10"], "away_win10": af["win10"],
        "home_rest_days": hf["rest_days"], "away_rest_days": af["rest_days"],
        "h2h_n": n, "h2h_home_winrate": wr, "h2h_home_gd": gd,

    }
    return pd.DataFrame([row])[FEATURES].astype(float)


# Draw probability threshold: if draw_prob >= this AND no team has a dominant
# win probability, promote the prediction to Draw.
#
# Tuning history (from backtest on 3622 matches, 2023-2026):
#   threshold=0.22, max_win=0.52 -> draw recall 66.6% but overall accuracy 49.4%  (too aggressive)
#   threshold=0.26, max_win=0.55 -> target: ~30-35% draw recall, ~58%+ overall
DRAW_THRESHOLD = 0.26
DRAW_MAX_WIN_PROB = 0.55  # don't call draw if one team clearly favoured


def calibrated_pick(p_home, p_draw, p_away):
    """Return the predicted label (0/1/2) with draw-threshold calibration.

    Standard argmax almost never picks draws because the raw draw probability
    sits around 0.20-0.28 and is always beaten by the win probabilities.
    This function promotes a draw prediction whenever:
      - draw probability >= DRAW_THRESHOLD, AND
      - neither team has a dominant win probability (< DRAW_MAX_WIN_PROB)
    """
    if (p_draw >= DRAW_THRESHOLD and
            p_home < DRAW_MAX_WIN_PROB and
            p_away < DRAW_MAX_WIN_PROB):
        return 1  # Draw
    return int(np.argmax([p_home, p_draw, p_away]))


def predict_symmetric(model, long, final_elo, a, b, asof, neutral, weight, odds_data=None):
    """Predict match outcome, blending model probabilities with market odds.

    Odds are fetched ONCE here and passed into build_match_row to avoid
    the previous bug where they were fetched separately on each call.

    Blending is now confidence-weighted: the stronger the model's top
    probability, the more we trust the model over the market. This replaces
    the previous arbitrary hardcoded 60% market / 40% model split.
    """
    # Fetch odds once and reuse for both orientations
    match_odds = get_match_odds(a, b, odds_data) if odds_data else None

    p_ab = model.predict_proba(build_match_row(long, final_elo, a, b, neutral, weight, asof, match_odds))[0]
    p_ba = model.predict_proba(build_match_row(long, final_elo, b, a, neutral, weight, asof, match_odds))[0]
    p_a = (p_ab[0] + p_ba[2]) / 2.0
    p_d = (p_ab[1] + p_ba[1]) / 2.0
    p_b = (p_ab[2] + p_ba[0]) / 2.0
    tot = p_a + p_d + p_b
    model_p = [p_a / tot, p_d / tot, p_b / tot]

    if match_odds:
        implied = odds_to_implied_probs(match_odds)
        if implied:
            odds_p = [implied["home"], implied["draw"], implied["away"]]
            print(f"  Model : {model_p[0]*100:.1f}% / {model_p[1]*100:.1f}% / {model_p[2]*100:.1f}%")
            print(f"  Market: {odds_p[0]*100:.1f}% / {odds_p[1]*100:.1f}% / {odds_p[2]*100:.1f}%")

            # Confidence-weighted blend: high model confidence -> trust model more.
            # Ranges from 0.5 (model=market) at top_prob=0.33 to 0.7 (model) at top_prob>=0.65.
            top_prob = max(model_p)
            model_weight = min(0.7, max(0.5, (top_prob - 0.33) / (0.65 - 0.33) * 0.2 + 0.5))
            market_weight = 1.0 - model_weight
            blended = [model_weight * m + market_weight * o for m, o in zip(model_p, odds_p)]
            tot = sum(blended)
            model_p = [p / tot for p in blended]

    return model_p


# ── fixtures ──────────────────────────────────────────────────────────────────
def map_fixture_name(name):
    name = name.strip()
    return FIXTURE_NAME_MAP.get(name, name)


def _side_matches(user_input, raw_name):
    """True if the user's typed team matches a fixture side (by raw or mapped name)."""
    u = user_input.strip().lower()
    return u in {raw_name.strip().lower(), map_fixture_name(raw_name).strip().lower()}


def find_fixture(team_a, team_b):
    """Find the single fixture for the two named teams (order doesn't matter)."""
    if not os.path.exists(FIXTURES_PATH):
        return None
    fx = pd.read_csv(FIXTURES_PATH)
    for _, row in fx.iterrows():
        if " v " not in str(row["teams"]):
            continue
        left, right = [p.strip() for p in str(row["teams"]).split(" v ")]
        forward = _side_matches(team_a, left) and _side_matches(team_b, right)
        reverse = _side_matches(team_a, right) and _side_matches(team_b, left)
        if forward or reverse:
            return {"match": row.get("match_number", ""), "group": row.get("group", ""),
                    "stadium": row.get("stadium", ""), "date": row.get("date_dt", ""),
                    "home_disp": left, "away_disp": right,
                    "home": map_fixture_name(left), "away": map_fixture_name(right)}
    return None


def list_team_names():
    if not os.path.exists(FIXTURES_PATH):
        return []
    fx = pd.read_csv(FIXTURES_PATH)
    names = set()
    for t in fx["teams"]:
        if " v " in str(t):
            for p in str(t).split(" v "):
                p = p.strip()
                if not any(w in p.lower() for w in ["winner", "runner", "third", "place", "group"]):
                    names.add(p)
    return sorted(names)


# ── chart ───────────────────────────────────────────────────────────────────────
def make_chart(m, p_home, p_draw, p_away, slate_date, out_dir, odds_data=None):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = [f"{m['home_disp']}\nwin", "Draw", f"{m['away_disp']}\nwin"]
    vals = [p_home, p_draw, p_away]
    colors = [ORANGE, GRAY, BLUE]
    bars = ax.bar(labels, [v * 100 for v in vals], color=colors, width=0.62, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1.2, f"{v*100:.1f}%",
                ha="center", va="bottom", fontsize=16, fontweight="bold")
    ax.set_ylim(0, max(vals) * 100 + 12)
    ax.set_ylabel("Win probability (%)")
    
    # Add odds info if available
    subtitle = f"{slate_date}  ·  {m['group']}  ·  {m['stadium']}"
    if odds_data:
        match_odds = get_match_odds(m["home"], m["away"], odds_data)
        if match_odds:
            subtitle += f"\nMarket odds: {m['home_disp']} {match_odds.get('home', 'N/A')} | Draw {match_odds.get('draw', 'N/A')} | {m['away_disp']} {match_odds.get('away', 'N/A')}"
    
    ax.set_title(f"{m['home_disp']} vs {m['away_disp']}\n{subtitle}", fontsize=13)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    safe = f"{m['home_disp']}_vs_{m['away_disp']}".replace(" ", "_").replace("/", "-")
    path = os.path.join(out_dir, f"viz_{safe}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def get_squad(team_name):
    """Fetch squad data from API-Football (free endpoint)."""
    if not API_FOOTBALL_KEY:
        return None

    # First get team ID by name
    url = "https://v3.football.api-sports.io/teams"
    params = {"search": team_name}
    headers = {"x-apisports-key": API_FOOTBALL_KEY}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        data = resp.json()
        if not data["response"]:
            print(f"  [api-football] No team found for '{team_name}'")
            return None
        team_id = data["response"][0]["team"]["id"]

        # Now get squad
        url2 = "https://v3.football.api-sports.io/players/squads"
        params2 = {"team": team_id}
        resp2 = requests.get(url2, params=params2, headers=headers, timeout=30)
        data2 = resp2.json()
        if data2["response"]:
            return data2["response"][0]["players"]
        else:
            print(f"  [api-football] No squad data returned for '{team_name}'")
    except Exception as e:
        print(f"  [api-football] Error fetching squad for '{team_name}': {e}")
    return None


def print_squad(name, players):
    """Print squad grouped by position."""
    if not players:
        return
    
    positions = {}
    for p in players:
        pos = p.get("position", "Unknown")
        if pos not in positions:
            positions[pos] = []
        positions[pos].append(p)
    
    print(f"┌─── {name.upper()} SQUAD ({len(players)} players) ─────────────────────────────┐")
    for pos in ["Goalkeeper", "Defender", "Midfielder", "Attacker"]:
        if pos in positions:
            print(f"│  {pos}s:")
            for p in positions[pos]:
                age = p.get("age", "?")
                number = p.get("number", "?")
                print(f"│    #{number:<3} {p['name']:<25} Age: {age}")
    print(f"└{'─' * 63}┘")


def tag_match(top_prob, p_home, p_away, home_elo, away_elo):
    favorite_is_home = p_home >= p_away
    fav_elo_is_home = home_elo >= away_elo
    upset = (favorite_is_home != fav_elo_is_home)
    if top_prob >= 0.60:
        strength = "MODEL_FAVORITE"
    elif top_prob >= 0.45:
        strength = "MODEL_LEAN"
    else:
        strength = "TOSS-UP"
    return strength + ("  ⚠️ UPSET PICK" if upset else "")


# ── main ────────────────────────────────────────────────────────────────────────
def history_before(results, cutoff):
    """Filter before feature construction, including final Elo, to avoid future leakage."""
    return results.loc[results['date'] < pd.Timestamp(cutoff)].copy()


def main():
    parser = argparse.ArgumentParser(description='Historical football outcome model (experimental).')
    parser.add_argument('home', help='First national team')
    parser.add_argument('away', help='Second national team')
    parser.add_argument('--date', help='Prediction cutoff YYYY-MM-DD; required when no fixture is available')
    parser.add_argument('--offline', action='store_true', help='Use cached results and no external APIs')
    parser.add_argument('--refresh-data', action='store_true', help='Download a fresh results CSV')
    args = parser.parse_args()
    if args.offline and args.refresh_data:
        parser.error('--offline and --refresh-data cannot be combined')
    team_a, team_b = normalize_country(args.home), normalize_country(args.away)
    if team_a.casefold() == team_b.casefold():
        parser.error('Choose two different teams')
    m = find_fixture(team_a, team_b) if not args.date else None
    if args.date:
        try:
            cutoff = pd.Timestamp(args.date)
            if str(cutoff.date()) != args.date: raise ValueError()
        except Exception:
            parser.error('--date must use YYYY-MM-DD')
        m = {'home': team_a, 'away': team_b, 'home_disp': team_a, 'away_disp': team_b,
             'date': args.date, 'group': 'Custom matchup', 'stadium': 'Neutral venue assumption', 'match': ''}
    if m is None:
        parser.error('No fixture found. Use --date YYYY-MM-DD or supply data_cache/fixtures.csv.')
    if args.offline and not (Path(CACHE_DIR) / 'results.csv').exists():
        parser.error('Offline mode requires data_cache/results.csv. Run once online or provide the CSV.')
    try:
        if args.refresh_data:
            # Validate before replacing an existing cache.
            import io
            response = requests.get(RESULTS_URL, timeout=60)
            response.raise_for_status()
            data = pd.read_csv(io.BytesIO(response.content))
            required = {'date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament', 'neutral'}
            if not required.issubset(data.columns): raise ValueError('Results CSV has missing columns')
            Path(CACHE_DIR).mkdir(exist_ok=True)
            temporary = Path(CACHE_DIR) / 'results.tmp'
            temporary.write_bytes(response.content)
            temporary.replace(Path(CACHE_DIR) / 'results.csv')
        results = history_before(load_results(), m['date'])
        valid = {name.casefold(): name for name in set(results.home_team) | set(results.away_team)}
        for side in ('home', 'away'):
            if m[side].casefold() not in valid: raise ValueError(f"Unknown team before cutoff: {m[side]}")
            m[side] = valid[m[side].casefold()]
        dataset, final_elo = build_dataset(results)
        long = per_team_long(results)
        train, val = split_by_date(dataset, TRAIN_START, VAL_START, m['date'])
        model, X_val, y_val = train_model(train, val)
        evaluate(model, X_val, y_val)
        # Current API data cannot reconstruct a historical pre-match snapshot.
        live = not args.offline and pd.Timestamp(m['date']).date() >= pd.Timestamp.now(tz='UTC').date()
        odds_data = get_odds() if live else None
        if live:
            for side in ('home', 'away'):
                squad = get_squad(m[side])
                if squad: print_squad(m[side], squad)
        p_home, p_draw, p_away = predict_symmetric(model, long, final_elo, m['home'], m['away'],
                                                  m['date'], MATCH_NEUTRAL, MATCH_WEIGHT, odds_data)
        outcomes = [(m['home'], p_home), ('Draw', p_draw), (m['away'], p_away)]
        pick, confidence = outcomes[calibrated_pick(p_home, p_draw, p_away)]
        out_dir = ROOT / 'predictions' / m['date']
        out_dir.mkdir(parents=True, exist_ok=True)
        chart = make_chart(m, p_home, p_draw, p_away, m['date'], str(out_dir), odds_data)
        for name, probability in outcomes: print(f'{name}: {probability:.1%}')
        print(f'Heuristic pick: {pick} ({confidence:.1%}); these probabilities are not a guarantee.')
        print(f'Chart saved: {chart}')
    except (ValueError, OSError, requests.RequestException) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
