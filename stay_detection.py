"""
stay_detection.py
Detects stay points from raw GPS traces using Haversine distance thresholding.

A stay point is a location where the user remained within RADIUS_M metres
for at least MIN_STAY_MINUTES minutes.
"""

import math
import pandas as pd
from datetime import timedelta


RADIUS_M = 100          # spatial threshold (metres)
MIN_STAY_MINUTES = 15   # minimum dwell time


def haversine(lat1, lon1, lat2, lon2):
    #Return great-circle distance in metres between two GPS coordinates.
    R = 6_371_000  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def detect_stays(df, radius_m=RADIUS_M, min_minutes=MIN_STAY_MINUTES):
    """
    Detect stay points from a GPS DataFrame.

    Parameters:
    df : pd.DataFrame
        Must have columns: latitude, longitude, timestamp (datetime).
    radius_m : float
        Spatial threshold in metres.
    min_minutes : int
        Minimum dwell time in minutes.

    Returns:
    list[dict]
        Each dict has: lat, lon, arrival, departure, duration_min, day_of_week, hour_slot.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    stays = []
    i = 0

    while i < len(df):
        j = i + 1
        while j < len(df):
            dist = haversine(df.loc[i, "latitude"], df.loc[i, "longitude"],
                             df.loc[j, "latitude"], df.loc[j, "longitude"])
            if dist > radius_m:
                break
            j += 1

        arrival = df.loc[i, "timestamp"]
        departure = df.loc[j - 1, "timestamp"]
        duration = (departure - arrival).total_seconds() / 60

        if duration >= min_minutes:
            # centroid of stay cluster
            cluster = df.iloc[i:j]
            lat_c = cluster["latitude"].mean()
            lon_c = cluster["longitude"].mean()
            hour = arrival.hour
            # time slot: 2-hour buckets
            slot = f"{(hour // 2) * 2:02d}-{(hour // 2) * 2 + 2:02d}"
            stays.append({
                "lat": lat_c,
                "lon": lon_c,
                "arrival": arrival,
                "departure": departure,
                "duration_min": round(duration, 1),
                "day_of_week": arrival.strftime("%A"),
                "hour_slot": slot,
            })
            i = j
        else:
            i += 1

    return stays


def load_studentlife_gps(filepath):
    """
    Load a Dartmouth StudentLife GPS CSV file.

    Expected columns (11 fields, no header):
        time, provider, network_type, latitude, longitude,
        altitude, bearing, speed, travelstate, confidence, index
    """
    col_names = [
        "time", "provider", "network_type",
        "latitude", "longitude", "altitude",
        "bearing", "speed", "travelstate", "confidence", "index"
    ]
    df = pd.read_csv(filepath, header=None, names=col_names, index_col=False)
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_localize(None)
    df = df[["timestamp", "latitude", "longitude"]].dropna()

    # Bounding box: Dartmouth / Hanover NH area
    df = df[
        (df["latitude"].between(43.5, 44.5)) &
        (df["longitude"].between(-72.5, -72.0))
    ]
    return df.reset_index(drop=True)
