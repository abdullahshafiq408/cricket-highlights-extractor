"""
utils/json_utils.py
───────────────────
Handles everything related to Cricsheet data:

1. load_cricsheet_json()   → loads and flattens a Cricsheet JSON file
2. fetch_by_espn_url()     → tries to find the Cricsheet match ID from an
                             ESPN Cricinfo URL, then downloads the JSON
3. extract_events()        → returns a list of "interesting" events from
                             the ball-by-ball data (boundaries, wickets, etc.)

About Cricsheet format:
───────────────────────
Cricsheet JSON looks like this (simplified):

{
  "info": { "teams": [...], "dates": [...] },
  "innings": [
    {
      "team": "India",
      "overs": [
        {
          "over": 0,
          "deliveries": [
            {
              "batter": "Rohit",
              "bowler": "Broad",
              "runs": {"batter": 4, "extras": 0, "total": 4},
              "wickets": []     ← empty if no wicket
            },
            ...
          ]
        }
      ]
    }
  ]
}

We flatten this into one row per delivery so it's easy to work with.
"""

import json
import re
import requests
from pathlib import Path


EVENT_BOUNDARY_4 = "boundary_4"
EVENT_BOUNDARY_6 = "boundary_6"
EVENT_WICKET = "wicket"
EVENT_DOT_BALL = "dot_ball"
EVENT_NORMAL = "normal"



def load_cricsheet_json(filepath: str) -> dict:
    """
    Load a Cricsheet JSON file and return it as a Python dict.

    Raises a clear error if the file doesn't look like Cricsheet format.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Cricsheet JSON not found: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Basic validation
    if "innings" not in data:
        raise ValueError(
            f"This doesn't look like a Cricsheet JSON file — "
            f"missing 'innings' key. Got keys: {list(data.keys())}"
        )

    return data


def flatten_deliveries(cricsheet_data: dict) -> list[dict]:
    """
    Flatten the nested Cricsheet structure into a flat list of deliveries.
    UPDATED: Now strictly tracks legal deliveries to prevent impossible 
    .7 or .8 over counts caused by wides and no-balls.
    """
    deliveries = []

    for innings_idx, innings in enumerate(cricsheet_data.get("innings", [])):
        team = innings.get("team", f"Team {innings_idx + 1}")

        for over_data in innings.get("overs", []):
            over_num = over_data.get("over", 0)

            for delivery in over_data.get("deliveries", []):
                
                # Check for illegal deliveries
                extras = delivery.get("extras", {})
                is_wide = "wides" in extras
                is_noball = "noballs" in extras

                if is_wide:
                    continue

                if not is_noball:
                    legal_ball_count += 1

                ball_num = max(1, legal_ball_count)
                if ball_num > 6:
                    ball_num = 6

                runs_batter = delivery.get("runs", {}).get("batter", 0)
                runs_total = delivery.get("runs", {}).get("total",  0)

                wickets = delivery.get("wickets", [])
                is_wicket = len(wickets) > 0
                wicket_kind = wickets[0].get("kind", None) if is_wicket else None
                player_out = wickets[0].get("player_out", None) if is_wicket else None

   
                event_type = _classify_event(runs_batter, is_wicket)

                deliveries.append({
                    "innings":           innings_idx + 1,
                    "team":              team,
                    "over":              over_num,
                    "ball":              ball_num,
                    "over_ball":         f"{over_num}.{ball_num}",
                    "batter":            delivery.get("batter", ""),
                    "bowler":            delivery.get("bowler", ""),
                    "runs_batter":       runs_batter,
                    "runs_total":        runs_total,
                    "is_wicket":         is_wicket,
                    "wicket_kind":       wicket_kind,
                    "wicket_player_out": player_out,
                    "event_type":        event_type,
                })

    return deliveries


def _classify_event(runs_batter: int, is_wicket: bool) -> str:
    """Simple rule-based classifier for delivery outcomes."""
    if is_wicket:
        return EVENT_WICKET
    if runs_batter == 6:
        return EVENT_BOUNDARY_6
    if runs_batter == 4:
        return EVENT_BOUNDARY_4
    if runs_batter == 0:
        return EVENT_DOT_BALL
    return EVENT_NORMAL


EVENT_PRIORITY = {
    EVENT_WICKET:     1.0,    # most exciting
    EVENT_BOUNDARY_6: 0.9,
    EVENT_BOUNDARY_4: 0.75,
    EVENT_DOT_BALL:   0.2,
    EVENT_NORMAL:     0.1,
}


def extract_events(
    deliveries:       list[dict],
    include_types:    list[str] | None = None,
) -> list[dict]:
    """
    Filter deliveries to only keep "interesting" events.

    include_types : list of event type strings to keep.
                    Defaults to wickets and boundaries.

    Returns the same dicts but with an added "base_priority" field.
    """
    if include_types is None:
        include_types = [EVENT_WICKET, EVENT_BOUNDARY_6, EVENT_BOUNDARY_4]

    events = []
    for d in deliveries:
        if d["event_type"] in include_types:
            d = dict(d)   # copy so we don't mutate the original
            d["base_priority"] = EVENT_PRIORITY.get(d["event_type"], 0.1)
            events.append(d)

    print(
        f"[json_utils] Found {len(events)} highlight events in Cricsheet data")
    return events

def fetch_by_espn_url(espn_url: str, save_to: str | None = None) -> dict | None:
    """
    Try to extract a match ID from an ESPN Cricinfo URL and fetch the
    corresponding Cricsheet JSON.

    ESPN URLs look like:
        https://www.espncricinfo.com/series/icc-wc-2023/match/1367712

    We extract the numeric match ID (1367712 here) and search Cricsheet.

    NOTE: Cricsheet doesn't have an official search-by-ESPN-ID API, so
    this is a best-effort approach using the Cricsheet API search endpoint.

    Returns the parsed JSON dict, or None if not found.
    """
    match = re.search(r"[-/](\d{6,8})(?:[/?#]|$)", espn_url)
    if not match:
        print(f"[json_utils] Could not extract match ID from URL: {espn_url}")
        return None

    espn_match_id = match.group(1)
    print(f"[json_utils] Extracted ESPN match ID: {espn_match_id}")

    search_url = f"https://cricsheet.org/api/matches/search/?espn_id={espn_match_id}"
    try:
        resp = requests.get(search_url, timeout=10)
        resp.raise_for_status()
        results = resp.json()
    except Exception as e:
        print(f"[json_utils] Cricsheet search failed: {e}")
        return None

    if not results:
        print(
            f"[json_utils] No Cricsheet match found for ESPN ID {espn_match_id}")
        return None

    cricsheet_id = results[0].get("id")
    download_url = f"https://cricsheet.org/api/matches/{cricsheet_id}/json"

    try:
        resp = requests.get(download_url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(
            f"[json_utils] Failed to download Cricsheet match {cricsheet_id}: {e}")
        return None

    if save_to:
        with open(save_to, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[json_utils] Saved Cricsheet JSON → {save_to}")

    return data
