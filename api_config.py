"""
api_config.py — API configuration and data fetching functions
============================================================

Loads API keys from .env and provides functions to fetch:
- Betting odds from The Odds API
- Live football data from RapidAPI
- Match data from API-Football
- Match data from football-data.org
- Live scores from The Rundown

API Status (tested):
- RapidAPI (free-api-live-football-data): Key valid but needs subscription
- API-Football via RapidAPI: Needs subscription
- API-Football direct: Needs API key from api-football.com
- football-data.org: Free tier only lists competitions, matches need paid plan
- The Rundown: Needs valid API key
- The Odds API: Needs valid API key from the-odds-api.com
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

# API Keys
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
FOOTBALL_DATA_ORG_KEY = os.getenv("FOOTBALL_DATA_ORG_KEY")
THERUNDOWN_KEY = os.getenv("THERUNDOWN_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")

# API Base URLs
RAPIDAPI_HOST = "free-api-live-football-data.p.rapidapi.com"
API_FOOTBALL_HOST = "api-football-v1.p.rapidapi.com"
FOOTBALL_DATA_ORG_BASE = "https://api.football-data.org/v4"
THERUNDOWN_BASE = "https://therundown.io/api/v2"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"


# ── RapidAPI Football Data ─────────────────────────────────────────────────────
def search_players_rapidapi(query):
    """Search football players via RapidAPI. Requires subscription."""
    if not RAPIDAPI_KEY:
        print("  [rapidapi] No RAPIDAPI_KEY set")
        return None
    
    url = f"https://{RAPIDAPI_HOST}/football-players-search"
    params = {"search": query}
    headers = {
        "x-rapidapi-host": RAPIDAPI_HOST,
        "x-rapidapi-key": RAPIDAPI_KEY
    }
    
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        data = resp.json()
        if "message" in data and "subscribed" in data["message"].lower():
            print(f"  [rapidapi] Needs subscription: {data['message']}")
            return None
        resp.raise_for_status()
        return data
    except Exception as e:
        print(f"  [rapidapi] Error: {e}")
        return None


def get_live_events_rapidapi(league_id=1):
    """Get top live events via RapidAPI. Requires subscription."""
    if not RAPIDAPI_KEY:
        return None
    
    url = f"https://{RAPIDAPI_HOST}/football-top-events"
    params = {"league_id": league_id}
    headers = {
        "x-rapidapi-host": RAPIDAPI_HOST,
        "x-rapidapi-key": RAPIDAPI_KEY
    }
    
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        data = resp.json()
        if "message" in data and "subscribed" in data["message"].lower():
            return None
        resp.raise_for_status()
        return data
    except Exception as e:
        print(f"  [rapidapi] Error: {e}")
        return None


# ── API-Football (via RapidAPI or direct) ──────────────────────────────────────
def get_team_id_api_football(team_name, use_rapidapi=True):
    """Get team ID from API-Football. Uses direct API first."""
    if API_FOOTBALL_KEY:
        url = "https://v3.football.api-sports.io/teams"
        params = {"search": team_name}
        headers = {"x-apisports-key": API_FOOTBALL_KEY}
    elif use_rapidapi and RAPIDAPI_KEY:
        url = f"https://{API_FOOTBALL_HOST}/v3/teams"
        params = {"search": team_name}
        headers = {
            "x-rapidapi-host": API_FOOTBALL_HOST,
            "x-rapidapi-key": RAPIDAPI_KEY
        }
    else:
        print("  [api-football] No API key available")
        return None
    
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        data = resp.json()
        
        if "errors" in data and data["errors"]:
            error_msg = str(data["errors"])
            if "token" in error_msg.lower() or "key" in error_msg.lower():
                return None
        
        resp.raise_for_status()
        teams = data.get("response", [])
        if teams:
            return teams[0]["team"]["id"]
    except Exception as e:
        pass
    
    return None


def get_injuries_api_football(team_id, season=2026):
    """Fetch injury data from API-Football.
    
    The /injuries endpoint requires either a fixture ID or a league+season
    combination. Passing only `team` returns empty results. We default to
    the current World Cup season (2026).
    """
    if API_FOOTBALL_KEY:
        url = "https://v3.football.api-sports.io/injuries"
        headers = {"x-apisports-key": API_FOOTBALL_KEY}
    elif RAPIDAPI_KEY:
        url = f"https://{API_FOOTBALL_HOST}/v3/injuries"
        headers = {
            "x-rapidapi-host": API_FOOTBALL_HOST,
            "x-rapidapi-key": RAPIDAPI_KEY
        }
    else:
        return None

    # Without league+season the endpoint returns nothing — this was the bug.
    params = {"team": team_id, "season": season, "league": 1}  # league 1 = FIFA World Cup

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        data = resp.json()

        if "errors" in data and data["errors"]:
            print(f"  [api-football] Injury fetch error: {data['errors']}")
            return None

        resp.raise_for_status()
        return data.get("response", [])
    except Exception as e:
        print(f"  [api-football] Error fetching injuries: {e}")
        return None


# ── football-data.org ──────────────────────────────────────────────────────────
def get_competitions_football_data_org():
    """List available competitions (free tier)."""
    url = f"{FOOTBALL_DATA_ORG_BASE}/competitions"
    headers = {}
    if FOOTBALL_DATA_ORG_KEY:
        headers["X-Auth-Token"] = FOOTBALL_DATA_ORG_KEY
    
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json().get("competitions", [])
    except Exception as e:
        print(f"  [football-data.org] Error: {e}")
        return []


def get_matches_football_data_org(competition_id=2000, status=None, limit=10):
    """Fetch matches from football-data.org. World Cup (2000) requires paid plan."""
    url = f"{FOOTBALL_DATA_ORG_BASE}/competitions/{competition_id}/matches"
    headers = {}
    if FOOTBALL_DATA_ORG_KEY:
        headers["X-Auth-Token"] = FOOTBALL_DATA_ORG_KEY
    
    params = {"limit": limit}
    if status:
        params["status"] = status
    
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        data = resp.json()
        
        if resp.status_code == 403:
            print(f"  [football-data.org] Requires paid subscription for competition {competition_id}")
            return None
        
        resp.raise_for_status()
        return data.get("matches", [])
    except Exception as e:
        print(f"  [football-data.org] Error: {e}")
        return None


# ── The Rundown ────────────────────────────────────────────────────────────────
def get_events_rundown(sport_id=4, date="2026-06-15"):
    """Fetch events from The Rundown. Requires valid API key."""
    if not THERUNDOWN_KEY or THERUNDOWN_KEY == "YOUR_KEY":
        print("  [rundown] No valid THERUNDOWN_KEY set")
        return None
    
    url = f"{THERUNDOWN_BASE}/sports/{sport_id}/events/{date}"
    params = {"key": THERUNDOWN_KEY}
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()
        
        if "message" in data and "unauthorized" in data["message"].lower():
            print("  [rundown] Invalid API key")
            return None
        
        resp.raise_for_status()
        return data.get("events", [])
    except Exception as e:
        print(f"  [rundown] Error: {e}")
        return None


# ── The Odds API ───────────────────────────────────────────────────────────────
def get_odds(sport="soccer_fifa_world_cup", region="eu", markets="h2h"):
    """Fetch betting odds from The Odds API. Requires valid API key."""
    if not ODDS_API_KEY or ODDS_API_KEY == "YOUR_ODDS_API_KEY_HERE":
        print("  [odds] No valid ODDS_API_KEY set - get free key at https://the-odds-api.com")
        return None
    
    url = f"{ODDS_API_BASE}/sports/{sport}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": region,
        "markets": markets,
        "oddsFormat": "decimal"
    }
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()
        
        if resp.status_code == 401:
            print(f"  [odds] Invalid API key - {data.get('message', '')}")
            return None
        
        if resp.status_code == 422:
            print(f"  [odds] No matches available for {sport}")
            return []
        
        resp.raise_for_status()
        return data
    except Exception as e:
        print(f"  [odds] Error: {e}")
        return None


def get_match_odds(home_team, away_team, odds_data):
    """Extract odds for a specific match from odds data."""
    if not odds_data:
        return None
    
    for game in odds_data:
        ht = game.get("home_team", "")
        at = game.get("away_team", "")
        if (home_team in ht or ht in home_team) and (away_team in at or at in away_team):
            bookmakers = game.get("bookmakers", [])
            if bookmakers:
                markets = bookmakers[0].get("markets", [])
                for market in markets:
                    if market.get("key") == "h2h":
                        outcomes = market.get("outcomes", [])
                        odds = {}
                        for outcome in outcomes:
                            name = outcome.get("name")
                            price = outcome.get("price")
                            if name == ht:
                                odds["home"] = price
                            elif name == at:
                                odds["away"] = price
                            elif name == "Draw":
                                odds["draw"] = price
                        if odds:
                            return odds
    return None


def odds_to_implied_probs(odds):
    """Convert decimal odds to implied probabilities."""
    if not odds or not all(k in odds for k in ["home", "draw", "away"]):
        return None
    
    total = (1/odds["home"]) + (1/odds["draw"]) + (1/odds["away"])
    return {
        "home": (1/odds["home"]) / total,
        "draw": (1/odds["draw"]) / total,
        "away": (1/odds["away"]) / total
    }


# ── xG via API-Football match statistics ─────────────────────────────────────
def get_xg_from_understat(team_slug):
    """DEPRECATED stub kept for import compatibility. Use get_xg_api_football() instead."""
    return None


def get_xg_api_football(team_id, season=2026, last_n=10):
    """Fetch average xG for/against from API-Football fixture statistics.
    
    Replaces the old Understat stub which never returned real data.
    Calls /fixtures to get the last N completed fixtures for the team,
    then /fixtures/statistics on each to extract 'Expected Goals' values.
    Returns a dict with xg_for and xg_against, or None if unavailable.
    
    Note: Each call costs API quota. Results are lightweight dicts so the
    caller should cache them if predicting multiple matches.
    """
    if not API_FOOTBALL_KEY:
        return None

    headers = {"x-apisports-key": API_FOOTBALL_KEY}

    # Step 1: get last N fixtures for this team in the given season
    try:
        fixtures_resp = requests.get(
            "https://v3.football.api-sports.io/fixtures",
            params={"team": team_id, "season": season, "last": last_n, "status": "FT"},
            headers=headers,
            timeout=30,
        )
        fixtures_data = fixtures_resp.json()
        if fixtures_data.get("errors"):
            print(f"  [api-football] xG fixtures error: {fixtures_data['errors']}")
            return None
        fixtures = fixtures_data.get("response", [])
    except Exception as e:
        print(f"  [api-football] Error fetching fixtures for xG: {e}")
        return None

    if not fixtures:
        return None

    xg_for_list, xg_against_list = [], []

    for fx in fixtures:
        fixture_id = fx["fixture"]["id"]
        home_team_id = fx["teams"]["home"]["id"]
        is_home = (home_team_id == team_id)

        try:
            stats_resp = requests.get(
                "https://v3.football.api-sports.io/fixtures/statistics",
                params={"fixture": fixture_id},
                headers=headers,
                timeout=30,
            )
            stats_data = stats_resp.json()
            teams_stats = stats_data.get("response", [])
        except Exception as e:
            print(f"  [api-football] Error fetching fixture stats ({fixture_id}): {e}")
            continue

        for team_stat in teams_stats:
            this_team_id = team_stat["team"]["id"]
            is_our_team = (this_team_id == team_id)
            for stat in team_stat.get("statistics", []):
                if stat["type"] == "Expected Goals" and stat["value"] is not None:
                    try:
                        val = float(stat["value"])
                        if is_our_team:
                            xg_for_list.append(val)
                        else:
                            xg_against_list.append(val)
                    except (ValueError, TypeError):
                        pass

    if not xg_for_list:
        return None

    return {
        "xg_for": round(sum(xg_for_list) / len(xg_for_list), 3),
        "xg_against": round(sum(xg_against_list) / len(xg_against_list), 3) if xg_against_list else 1.0,
    }


# ── Summary helper ─────────────────────────────────────────────────────────────
def print_api_status():
    """Print status of all configured APIs."""
    print("\n=== API Configuration Status ===")
    
    # RapidAPI
    if RAPIDAPI_KEY and RAPIDAPI_KEY != "YOUR_RAPIDAPI_KEY":
        print(f"  RapidAPI key: configured (needs subscription for football data)")
    else:
        print(f"  RapidAPI key: NOT SET")
    
    # API-Football
    if API_FOOTBALL_KEY and API_FOOTBALL_KEY != "YOUR_API_FOOTBALL_KEY_HERE":
        print(f"  API-Football key: configured")
    else:
        print(f"  API-Football key: NOT SET (get at https://www.api-football.com)")
    
    # football-data.org
    if FOOTBALL_DATA_ORG_KEY and FOOTBALL_DATA_ORG_KEY != "YOUR_FOOTBALL_DATA_ORG_KEY_HERE":
        print(f"  football-data.org key: configured (paid plan needed for matches)")
    else:
        print(f"  football-data.org key: NOT SET (free tier only lists competitions)")
    
    # The Rundown
    if THERUNDOWN_KEY and THERUNDOWN_KEY != "YOUR_KEY":
        print(f"  The Rundown key: configured")
    else:
        print(f"  The Rundown key: NOT SET (get at https://therundown.io)")
    
    # The Odds API
    if ODDS_API_KEY and ODDS_API_KEY != "YOUR_ODDS_API_KEY_HERE":
        print(f"  The Odds API key: configured")
    else:
        print(f"  The Odds API key: NOT SET (get FREE key at https://the-odds-api.com)")
    
    print("================================\n")
