"""
backtest.py — Walk-forward backtesting for the World Cup 2026 predictor
=======================================================================

Evaluates the XGBoost model by simulating real-world usage: for each match
in the validation window, the model is trained ONLY on data before that match,
then its prediction is compared against the actual result.

This prevents any look-ahead bias and gives a true out-of-sample accuracy.

Usage:
    python backtest.py                    # validate on 2023-01-01 to today
    python backtest.py --from 2022-01-01  # custom start date
    python backtest.py --tournament "FIFA World Cup" --from 2022-01-01

Results are saved to predictions/backtest_<date>.csv
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score, log_loss

warnings.filterwarnings("ignore")

# ── reuse core logic from predict_today ────────────────────────────────────────
from predict_today import (
    load_results, build_dataset, per_team_long,
    form_as_of, h2h_as_of, build_match_row,
    class_sample_weights, calibrated_pick,
    FEATURES, TRAIN_START, VAL_START, ELO_BASE, ELO_K, ELO_HOME_BONUS,
    INK, MUTE, GRID, ORANGE, BLUE, GRAY,
    tournament_weight, add_label_and_context, compute_elo,
    add_form_features, add_h2h_features
)

LABEL_MAP = {0: "Home Win", 1: "Draw", 2: "Away Win"}


# ── walk-forward training ──────────────────────────────────────────────────────
def train_on_cutoff(dataset, cutoff_date):
    """Train XGBoost on all data strictly before cutoff_date."""
    train = dataset[
        (dataset["date"] >= pd.Timestamp(TRAIN_START)) &
        (dataset["date"] < pd.Timestamp(cutoff_date))
    ].copy()

    if len(train) < 500:
        return None  # not enough data

    val_start = pd.Timestamp(TRAIN_START) + (pd.Timestamp(cutoff_date) - pd.Timestamp(TRAIN_START)) * 0.8
    val = train[train["date"] >= val_start].copy()
    train_inner = train[train["date"] < val_start].copy()

    if len(train_inner) < 200 or len(val) < 50:
        n = len(train)
        val = train.iloc[int(n * 0.8):].copy()
        train_inner = train.iloc[:int(n * 0.8)].copy()

    X_tr = train_inner[FEATURES].astype(float)
    y_tr = train_inner["label"].astype(int)
    X_v = val[FEATURES].astype(float)
    y_v = val["label"].astype(int)

    # Use inverse-frequency class weights to fix near-zero draw recall
    sw_train = class_sample_weights(y_tr.values)

    model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3, n_estimators=400,
        learning_rate=0.05, max_depth=5, subsample=0.85,
        colsample_bytree=0.85, reg_lambda=1.0, eval_metric="mlogloss",
        early_stopping_rounds=30, tree_method="hist",
        n_jobs=-1, random_state=42, verbosity=0,
    )
    model.fit(X_tr, y_tr, sample_weight=sw_train,
              eval_set=[(X_v, y_v)], verbose=False)
    return model


# ── backtest core ──────────────────────────────────────────────────────────────
def run_backtest(val_start, val_end, tournament_filter=None, retrain_every=50):
    """
    Walk-forward backtest.

    Parameters
    ----------
    val_start : str  – first date to evaluate (model trained on data before this)
    val_end   : str  – last date to evaluate
    tournament_filter : str or None  – only evaluate this tournament name
    retrain_every : int  – retrain the model every N matches (saves time vs retraining each match)
    """
    print(f"\nLoading & building full feature dataset ...")
    results = load_results()
    dataset, _ = build_dataset(results)
    long = per_team_long(results)

    val_mask = (dataset["date"] >= pd.Timestamp(val_start)) & \
               (dataset["date"] <= pd.Timestamp(val_end))

    if tournament_filter:
        val_mask &= dataset["tournament"].str.contains(tournament_filter, case=False, na=False)

    val_matches = dataset[val_mask].copy().sort_values("date").reset_index(drop=True)

    if len(val_matches) == 0:
        print(f"  No matches found between {val_start} and {val_end}")
        if tournament_filter:
            print(f"  (filter: '{tournament_filter}')")
        return None

    print(f"  Evaluating {len(val_matches)} matches from {val_start} to {val_end}")
    if tournament_filter:
        print(f"  Tournament filter: '{tournament_filter}'")
    print(f"  Retraining model every {retrain_every} matches ...")
    print()

    records = []
    model = None
    model_trained_at = -retrain_every  # force train on first match

    for idx, row in val_matches.iterrows():
        match_date = row["date"]

        # Retrain periodically (not every match — too slow for thousands of games)
        if idx - model_trained_at >= retrain_every:
            model = train_on_cutoff(dataset, match_date)
            model_trained_at = idx
            if model is None:
                continue

        home, away = row["home_team"], row["away_team"]
        actual_label = int(row["label"])

        # Compute Elo ratings as of this match date (using only data before it)
        hist = dataset[dataset["date"] < match_date].copy()
        if len(hist) < 10:
            continue

        # Get Elo from the full dataset's precomputed values for this row
        home_elo = float(row["home_elo"])
        away_elo = float(row["away_elo"])

        # Build feature row (no API calls in backtest — use historical data only)
        hf = form_as_of(long, home, match_date)
        af = form_as_of(long, away, match_date)
        n, wr, gd = h2h_as_of(long, home, away, match_date)

        feat_row = {
            "neutral": int(row["neutral"]),
            "tournament_weight": float(row["tournament_weight"]),
            "home_elo": home_elo, "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "home_win5": hf["win5"], "away_win5": af["win5"],
            "home_gd5": hf["gd5"], "away_gd5": af["gd5"],
            "home_win10": hf["win10"], "away_win10": af["win10"],
            "home_rest_days": hf["rest_days"], "away_rest_days": af["rest_days"],
            "h2h_n": n, "h2h_home_winrate": wr, "h2h_home_gd": gd,
            # Odds/xG/injuries not available in historical backtest — use neutral defaults
            "odds_home_implied": 0.33, "odds_draw_implied": 0.33, "odds_away_implied": 0.33,
            "home_xg_for": 1.5, "away_xg_for": 1.5,
            "home_xg_against": 1.0, "away_xg_against": 1.0,
            "home_injuries_count": 0, "away_injuries_count": 0,
        }

        X = pd.DataFrame([feat_row])[FEATURES].astype(float)

        # Symmetric prediction (average both orientations)
        proba_ab = model.predict_proba(X)[0]  # home=0, draw=1, away=2

        # Reverse orientation
        feat_rev = feat_row.copy()
        feat_rev.update({
            "home_elo": away_elo, "away_elo": home_elo,
            "elo_diff": away_elo - home_elo,
            "home_win5": af["win5"], "away_win5": hf["win5"],
            "home_gd5": af["gd5"], "away_gd5": hf["gd5"],
            "home_win10": af["win10"], "away_win10": hf["win10"],
            "home_rest_days": af["rest_days"], "away_rest_days": hf["rest_days"],
            "h2h_n": n,
            "h2h_home_winrate": 1.0 - wr if not np.isnan(wr) else 0.5,
            "h2h_home_gd": -gd if not np.isnan(gd) else 0.0,
        })
        X_rev = pd.DataFrame([feat_rev])[FEATURES].astype(float)
        proba_ba = model.predict_proba(X_rev)[0]

        p_home = (proba_ab[0] + proba_ba[2]) / 2.0
        p_draw = (proba_ab[1] + proba_ba[1]) / 2.0
        p_away = (proba_ab[2] + proba_ba[0]) / 2.0
        tot = p_home + p_draw + p_away
        p_home, p_draw, p_away = p_home / tot, p_draw / tot, p_away / tot

        pred_label = calibrated_pick(p_home, p_draw, p_away)
        correct = (pred_label == actual_label)

        records.append({
            "date": match_date.strftime("%Y-%m-%d"),
            "home_team": home,
            "away_team": away,
            "tournament": row["tournament"],
            "home_score": int(row["home_score"]),
            "away_score": int(row["away_score"]),
            "actual": LABEL_MAP[actual_label],
            "predicted": LABEL_MAP[pred_label],
            "correct": correct,
            "p_home": round(p_home, 4),
            "p_draw": round(p_draw, 4),
            "p_away": round(p_away, 4),
            "elo_diff": round(home_elo - away_elo, 1),
        })

        if (idx + 1) % 100 == 0:
            done = [r for r in records if r is not None]
            acc = sum(r["correct"] for r in done) / len(done) if done else 0
            print(f"  [{idx+1}/{len(val_matches)}]  running accuracy: {acc:.3f}")

    df = pd.DataFrame(records)
    return df


# ── metrics & chart ────────────────────────────────────────────────────────────
def compute_metrics(df):
    """Print detailed metrics and return summary dict."""
    actual_labels = df["actual"].map({v: k for k, v in LABEL_MAP.items()}).astype(int)
    pred_labels   = df["predicted"].map({v: k for k, v in LABEL_MAP.items()}).astype(int)
    proba_matrix  = df[["p_home", "p_draw", "p_away"]].values

    acc = accuracy_score(actual_labels, pred_labels)
    ll  = log_loss(actual_labels, proba_matrix, labels=[0, 1, 2])

    # Baseline: always predict the most frequent class
    counts = np.bincount(actual_labels, minlength=3)
    base_proba = np.tile(counts / counts.sum(), (len(actual_labels), 1))
    ll_base = log_loss(actual_labels, base_proba, labels=[0, 1, 2])

    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Matches evaluated : {len(df)}")
    print(f"  Accuracy          : {acc:.3f}  ({acc*100:.1f}%)")
    print(f"  Log-loss          : {ll:.3f}  (baseline: {ll_base:.3f})")
    print(f"  Edge over baseline: {ll_base - ll:+.3f}")
    print()

    # Per-outcome breakdown
    print("  Breakdown by actual outcome:")
    for label_int, label_str in LABEL_MAP.items():
        sub = df[df["actual"] == label_str]
        if len(sub) == 0:
            continue
        n_correct = sub["correct"].sum()
        print(f"    {label_str:<12}: {n_correct}/{len(sub)} correct  ({n_correct/len(sub)*100:.1f}%)")

    print()

    # Biggest upsets missed
    df["confidence"] = df[["p_home", "p_draw", "p_away"]].max(axis=1)
    missed_highs = df[~df["correct"]].nlargest(5, "confidence")
    if len(missed_highs):
        print("  Top 5 most confident wrong predictions:")
        for _, r in missed_highs.iterrows():
            print(f"    {r['date']}  {r['home_team']} vs {r['away_team']}  "
                  f"→ predicted {r['predicted']} ({r['confidence']*100:.0f}%) "
                  f"but was {r['actual']}")

    print("=" * 60)
    return {"accuracy": acc, "log_loss": ll, "log_loss_baseline": ll_base, "n": len(df)}


def make_backtest_chart(df, out_path):
    """Save a rolling-accuracy chart over time."""
    df = df.copy()
    df["date_parsed"] = pd.to_datetime(df["date"])
    df = df.sort_values("date_parsed")
    df["rolling_acc"] = df["correct"].astype(float).rolling(50, min_periods=10).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["date_parsed"], df["rolling_acc"] * 100, color=BLUE, lw=1.8, label="Rolling accuracy (50 matches)")
    ax.axhline(df["correct"].mean() * 100, color=ORANGE, ls="--", lw=1.2, label=f"Overall: {df['correct'].mean()*100:.1f}%")
    ax.set_ylim(20, 90)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Walk-Forward Backtest — Rolling Accuracy", fontweight="bold")
    ax.legend()
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  Chart saved → {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Walk-forward backtest for WC2026 predictor")
    p.add_argument("--from", dest="val_start", default="2023-01-01",
                   help="Start date for evaluation window (default: 2023-01-01)")
    p.add_argument("--to", dest="val_end", default=datetime.today().strftime("%Y-%m-%d"),
                   help="End date for evaluation window (default: today)")
    p.add_argument("--tournament", dest="tournament", default=None,
                   help="Filter to a specific tournament name (partial match, e.g. 'FIFA World Cup')")
    p.add_argument("--retrain-every", dest="retrain_every", type=int, default=50,
                   help="Retrain model every N matches (default: 50)")
    p.add_argument("--no-chart", dest="no_chart", action="store_true",
                   help="Skip saving the rolling accuracy chart")
    return p.parse_args()


def main():
    args = parse_args()

    df = run_backtest(
        val_start=args.val_start,
        val_end=args.val_end,
        tournament_filter=args.tournament,
        retrain_every=args.retrain_every,
    )

    if df is None or len(df) == 0:
        print("No results to evaluate.")
        sys.exit(1)

    metrics = compute_metrics(df)

    # Save CSV
    os.makedirs("predictions", exist_ok=True)
    today = datetime.today().strftime("%Y-%m-%d")
    csv_path = os.path.join("predictions", f"backtest_{today}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Full results saved → {csv_path}")

    # Save chart
    if not args.no_chart:
        chart_path = os.path.join("predictions", f"backtest_chart_{today}.png")
        make_backtest_chart(df, chart_path)


if __name__ == "__main__":
    main()
