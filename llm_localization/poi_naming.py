"""
poi_naming.py
Names GPS stay points using OpenStreetMap Overpass API (relations/ways/nodes)
with Nominatim as a fallback.
"""

import time
import requests


OVERPASS_URL = "https://overpass-api.de/api/interpreter"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
SEARCH_RADIUS_M = 100
HEADERS = {"User-Agent": "llm-location-prediction/1.0 (research project)"}


def _overpass_query(lat, lon, radius=SEARCH_RADIUS_M):
    """Query Overpass for named POIs near (lat, lon)."""
    query = f"""
    [out:json][timeout:10];
    (
      node["name"](around:{radius},{lat},{lon});
      way["name"](around:{radius},{lat},{lon});
      relation["name"](around:{radius},{lat},{lon});
    );
    out center tags;
    """
    try:
        resp = requests.post(OVERPASS_URL, data={"data": query}, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
        # Prefer amenity/building/leisure tags, then any named element
        for priority_key in ("amenity", "building", "leisure", "shop", "office"):
            for el in elements:
                if priority_key in el.get("tags", {}) and "name" in el["tags"]:
                    return el["tags"]["name"]
        # Fallback: first named element
        for el in elements:
            if "name" in el.get("tags", {}):
                return el["tags"]["name"]
    except Exception:
        pass
    return None


def _nominatim_query(lat, lon):
    #Reverse-geocode with Nominatim
    params = {
        "lat": lat,
        "lon": lon,
        "format": "jsonv2",
        "zoom": 18,
        "addressdetails": 1,
    }
    try:
        resp = requests.get(NOMINATIM_URL, params=params, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # Try display_name components in order of specificity
        addr = data.get("address", {})
        for key in ("amenity", "building", "house_name", "road", "neighbourhood"):
            if key in addr:
                return addr[key]
        return data.get("display_name", "Unknown Location").split(",")[0]
    except Exception:
        pass
    return "Unknown Location"


def name_stay(lat, lon, cache=None, delay=1.0):
    """
    Return a human-readable name for a stay point.

    Parameters:
    lat, lon : float
    cache : dict, optional
        Pass a dict to cache results by (lat, lon) key — avoids redundant API calls.
    delay : float
        Seconds to sleep between API calls (rate limiting).

    Returns:
    str
    """
    key = (round(lat, 5), round(lon, 5))
    if cache is not None and key in cache:
        return cache[key]

    name = _overpass_query(lat, lon)
    if name is None:
        time.sleep(delay)
        name = _nominatim_query(lat, lon)

    time.sleep(delay)

    if cache is not None:
        cache[key] = name
    return name


def name_stays(stays, cache=None, delay=1.0, verbose=True):
    """
    Add a 'name' field to each stay dict in-place.

    Parameters :
    stays : list[dict]
        Output of stay_detection.detect_stays().
    cache : dict, optional
    delay : float
    verbose : bool

    Returns:
    list[dict]  (same list, modified in-place)
    """
    if cache is None:
        cache = {}

    for i, stay in enumerate(stays):
        stay["name"] = name_stay(stay["lat"], stay["lon"], cache=cache, delay=delay)
        if verbose and (i + 1) % 10 == 0:
            print(f"  Named {i + 1}/{len(stays)} stays...")

    return stays
