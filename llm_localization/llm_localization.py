"""
LLM-Assisted Location Prediction
Dartmouth StudentLife Dataset — Groq API Inference

Pipeline:
1. Load GPS traces: detect stay points (15+ min within 100m)
2. Name each stay: Overpass API POIs + LLM (Dartmouth campus context)
3. Print full diary + user profile (top places, weekly timetable)
4. Predict location via LLM with business-hours constraints
5. Evaluate: LLM vs frequency baseline vs random
6. Uncertainty experiment: GPS noise vs naming accuracy

Setup:
    export GROQ_API_KEY="your-key-here"
    pip install groq requests pandas matplotlib numpy
    python llm_localization.py
"""

# Imports
import glob
import time
import math
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict
import requests
from groq import Groq

#  Config
DATA_ROOT       = '/content'
RESULTS_DIR     = '/content/results'
USERS_TO_RUN    = ['u01', 'u08', 'u13']
N_EVAL_SAMPLES  = 50
RUN_UNCERTAINTY = True
NOISE_LEVELS    = [0, 50, 100, 200, 400]

# Set GROQ_API_KEY as an environment variable
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise ValueError('GROQ_API_KEY environment variable not set. Run: export GROQ_API_KEY="your-key"')
GROQ_MODEL = 'llama-3.1-8b-instant'

os.makedirs(RESULTS_DIR, exist_ok=True)
client = Groq(api_key=GROQ_API_KEY)
print(f'Groq client ready -- model: {GROQ_MODEL}')


def generate(prompt, max_new_tokens=100):
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=max_new_tokens,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# GPS Loading 
def load_gps(data_root=DATA_ROOT):
    gps_files = glob.glob(f'{data_root}/**/sensing/gps/*.csv', recursive=True)
    if not gps_files:
        gps_files = glob.glob(f'{data_root}/dataset/sensing/gps/*.csv')
    print(f'Found {len(gps_files)} GPS files')
    dfs = []
    for f in gps_files:
        user = re.search(r'gps_(u\d+)\.csv', f)
        if not user:
            continue
        try:
            df = pd.read_csv(f, index_col=False)
            df['user'] = user.group(1)
            df['datetime'] = pd.to_datetime(df['time'], unit='s', errors='coerce')
            df = df.dropna(subset=['datetime', 'latitude', 'longitude'])
            df = df[(df['latitude'].between(40, 50)) & (df['longitude'].between(-80, -60))]
            dfs.append(df[['user', 'datetime', 'latitude', 'longitude', 'accuracy', 'travelstate']])
        except Exception as e:
            print(f'  Skip {f}: {e}')
    gps = pd.concat(dfs, ignore_index=True).sort_values(['user', 'datetime'])
    print(f'Total rows: {len(gps):,}  |  Users: {gps.user.nunique()}')
    return gps


# Stay Point Detection 
def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def detect_stay_points(user_df, dist_thresh=100, time_thresh=15):
    rows = user_df.sort_values('datetime').reset_index(drop=True)
    stays = []
    i = 0
    while i < len(rows):
        j = i + 1
        while j < len(rows):
            d = haversine(rows.loc[i, 'latitude'], rows.loc[i, 'longitude'],
                          rows.loc[j, 'latitude'], rows.loc[j, 'longitude'])
            if d > dist_thresh:
                break
            j += 1
        duration = (rows.loc[j - 1, 'datetime'] - rows.loc[i, 'datetime']).total_seconds() / 60
        if duration >= time_thresh:
            cluster = rows.iloc[i:j]
            stays.append({
                'arrive':       rows.loc[i, 'datetime'],
                'leave':        rows.loc[j - 1, 'datetime'],
                'duration_min': round(duration, 1),
                'lat':          cluster['latitude'].mean(),
                'lon':          cluster['longitude'].mean(),
                'avg_accuracy': cluster['accuracy'].mean(),
            })
        i = max(j, i + 1)
    return pd.DataFrame(stays)


#  POI Lookup and LLM Location Naming 
def get_nearby_pois(lat, lon, radius_m=500):
    """
    Query Overpass API for named places near a coordinate.
    Includes node + way + relation (university buildings are often relations).
    Falls back to Nominatim reverse geocoding if Overpass returns nothing.
    """
    radius_m = max(radius_m, 400)
    query = f"""
    [out:json][timeout:25];
    (
      node["name"](around:{radius_m},{lat},{lon});
      way["name"](around:{radius_m},{lat},{lon});
      relation["name"](around:{radius_m},{lat},{lon});
    );
    out tags;
    """
    try:
        r = requests.post('https://overpass-api.de/api/interpreter', data=query, timeout=30)
        elements = r.json().get('elements', [])
        names = list(dict.fromkeys(
            e['tags']['name'] for e in elements if 'name' in e.get('tags', {})
        ))
        if names:
            return names[:15]
    except Exception:
        pass
    try:
        r = requests.get(
            'https://nominatim.openstreetmap.org/reverse',
            params={'lat': lat, 'lon': lon, 'format': 'json', 'zoom': 18},
            headers={'User-Agent': 'llm-localization-research/1.0'},
            timeout=10
        )
        data = r.json()
        names = []
        addr = data.get('address', {})
        for key in ['amenity', 'building', 'leisure', 'tourism', 'shop', 'university']:
            if key in addr:
                names.append(addr[key])
        display = data.get('display_name', '').split(',')[0].strip()
        if display and display not in names:
            names.append(display)
        return names[:5] if names else []
    except Exception:
        return []


def duration_hint(duration_min):
    if duration_min > 300:
        return 'very long stay (5+ hours) - likely dorm/home, library, or all-day study'
    if duration_min > 120:
        return 'long stay (2-5 hours) - likely library, study room, or work'
    if duration_min > 45:
        return 'medium stay (45 min-2 hours) - class, meal, meeting'
    return 'short stay (under 45 min) - quick errand, meal pickup, or transit stop'


def time_of_day_label(hour):
    if hour < 6:  return 'late night / early morning (after midnight)'
    if hour < 9:  return 'morning'
    if hour < 12: return 'mid-morning'
    if hour < 14: return 'midday / lunchtime'
    if hour < 17: return 'afternoon'
    if hour < 20: return 'evening'
    if hour < 23: return 'night'
    return 'late night'


def name_location_llm(lat, lon, pois, arrive_dt, duration_min):
    hour  = arrive_dt.hour
    tod   = time_of_day_label(hour)
    dhint = duration_hint(duration_min)
    if pois:
        poi_str = ', '.join(pois[:12])
        prompt = (
            f"A Dartmouth College student (Hanover, NH) was at ({lat:.5f}, {lon:.5f}) "
            f"for {duration_min:.0f} minutes during {tod}.\n"
            f"Duration type: {dhint}\nNearby places: {poi_str}\n\n"
            f"Which of these nearby places did they most likely visit?\n"
            f"Pick the single most likely place. Do NOT say Unknown.\n\n"
            f"Reply exactly:\nLOCATION: <place name>\nCONFIDENCE: <High|Medium|Low>"
        )
    else:
        prompt = (
            f"A Dartmouth College student (Hanover, NH) was at ({lat:.5f}, {lon:.5f}) "
            f"for {duration_min:.0f} minutes during {tod}.\n"
            f"Duration type: {dhint}\nNo specific place names found nearby.\n\n"
            f"Based on time and duration, describe this location.\n"
            f"Examples: Residential Hall, Academic Building, Dining Hall, Library, Athletic Facility.\n"
            f"Do NOT say Unknown.\n\n"
            f"Reply exactly:\nLOCATION: <descriptive name>\nCONFIDENCE: <High|Medium|Low>"
        )
    try:
        text = generate(prompt, max_new_tokens=60)
        loc  = re.search(r'LOCATION:\s*(.+)', text)
        conf = re.search(r'CONFIDENCE:\s*(High|Medium|Low)', text, re.IGNORECASE)
        location = loc.group(1).strip() if loc else ''
        if not location or 'unknown' in location.lower() or len(location) < 3:
            location = 'Campus Building' if not pois else pois[0]
        return location, (conf.group(1) if conf else 'Low')
    except Exception as e:
        print(f'  LLM error: {e}')
        return 'Campus Building', 'Low'


def build_location_history(user_id, stay_df):
    records = []
    print(f'{user_id}: naming {len(stay_df)} stay points...')
    for idx, row in stay_df.iterrows():
        pois = get_nearby_pois(row['lat'], row['lon'])
        location, confidence = name_location_llm(
            row['lat'], row['lon'], pois, row['arrive'], row['duration_min'])
        records.append({
            'arrive': row['arrive'], 'leave': row['leave'],
            'duration_min': row['duration_min'], 'lat': row['lat'], 'lon': row['lon'],
            'avg_accuracy': row['avg_accuracy'], 'location': location,
            'confidence': confidence, 'n_pois': len(pois),
        })
        if (idx + 1) % 20 == 0:
            print(f'  {idx+1}/{len(stay_df)} named')
        time.sleep(0.3)
    return pd.DataFrame(records)


# Temporal Schedule and Profile Printing 
def time_slot(h):
    if h < 7:  return '00-07'
    if h < 9:  return '07-09'
    if h < 11: return '09-11'
    if h < 13: return '11-13'
    if h < 15: return '13-15'
    if h < 17: return '15-17'
    if h < 19: return '17-19'
    if h < 21: return '19-21'
    return '21-24'


def build_temporal_schedule(history):
    h = history.copy()
    h['arrive']  = pd.to_datetime(h['arrive'])
    h['hour']    = h['arrive'].dt.hour
    h['weekday'] = h['arrive'].dt.day_name()
    h['date']    = h['arrive'].dt.date
    h['slot']    = h['hour'].apply(time_slot)
    schedule = {}
    for (weekday, slot), group in h.groupby(['weekday', 'slot']):
        schedule[(weekday, slot)] = group['location'].value_counts().to_dict()
    overnight = h[h['hour'] < 7]
    home = overnight.groupby('location')['duration_min'].sum().idxmax() if len(overnight) > 0 \
           else h.groupby('location')['duration_min'].sum().idxmax()
    return {'schedule': schedule, 'home': home, 'history': h}


def print_stay_history(user_id, history):
    h = history.copy()
    h['arrive'] = pd.to_datetime(h['arrive'])
    h = h.sort_values('arrive')
    print(f'\n{"="*75}\n  STAY HISTORY -- {user_id}  ({len(h)} stays)\n{"="*75}')
    print(f'{"Date":<12} {"Time":<7} {"Location":<38} {"Duration":>9}  {"Conf":<6}  POIs')
    print(f'{"-"*75}')
    prev_date = None
    for _, row in h.iterrows():
        date_str = row['arrive'].strftime('%Y-%m-%d')
        if date_str != prev_date:
            if prev_date is not None: print()
            prev_date = date_str
        print(f'{date_str:<12} {row["arrive"].strftime("%H:%M"):<7} {str(row["location"])[:37]:<38} '
              f'{row["duration_min"]:.0f} min  {str(row.get("confidence", "?")):<6}  '
              f'{int(row["n_pois"]) if "n_pois" in row else "-"}')


def print_user_profile(user_id, patterns):
    h, schedule, home = patterns['history'], patterns['schedule'], patterns['home']
    print(f'\n{"="*55}\n  USER PROFILE -- {user_id}\n{"="*55}')
    print(f'  Inferred home   : {home}')
    print(f'  Total stays     : {len(h)}')
    print(f'  Unique locations: {h["location"].nunique()}')
    print(f'\n  Top 10 Most Visited Places:')
    for loc, cnt in h['location'].value_counts().head(10).items():
        print(f'    {loc:<40} {cnt:>3}x  |  {h[h["location"]==loc]["duration_min"].sum()/60:.1f} hrs')
    days  = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    slots = ['07-09', '09-11', '11-13', '13-15', '15-17', '17-19', '19-21', '21-24']
    header = '          ' + ''.join(f'{d[:3]:<13}' for d in days)
    print(f'\n  Weekly Timetable:\n{header}\n  {"-" * (len(header)-2)}')
    for slot in slots:
        row_str = f'  {slot}  '
        for day in days:
            locs = schedule.get((day, slot), {})
            cell = f'{max(locs,key=locs.get)[:9]}({locs[max(locs,key=locs.get)]})' if locs else '---'
            row_str += f'{cell:<13}'
        print(row_str)


# LLM Prediction and Evaluation 
def build_schedule_str(schedule, home):
    days  = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    slots = ['00-07', '07-09', '09-11', '11-13', '13-15', '15-17', '17-19', '19-21', '21-24']
    lines = [f'Home / usual overnight location: {home}', 'Weekly visit history:']
    for day in days:
        day_entries = []
        for slot in slots:
            locs = schedule.get((day, slot), {})
            if locs:
                top3 = sorted(locs.items(), key=lambda x: -x[1])[:3]
                day_entries.append(f'  {slot}: ' + ', '.join(f'{l}({c}x)' for l, c in top3))
        if day_entries:
            lines.append(f'\n{day}:')
            lines.extend(day_entries)
    return '\n'.join(lines)


def hours_open_context(hour):
    if hour < 6:
        return 'LATE NIGHT / EARLY MORNING (before 6am). Student almost certainly in dorm/apartment. Do not predict restaurant, cafe, library, or campus building.'
    if hour < 9:
        return 'EARLY MORNING (6-9am). Most campus buildings not open yet. Likely: dorm, gym, or early dining hall.'
    if hour < 22:
        return 'DAYTIME / EARLY EVENING (9am-10pm). Campus buildings, libraries, dining halls open. Predict based on usual schedule.'
    return 'LATE NIGHT (after 10pm). Most restaurants/buildings closed. Student likely in dorm or Baker Library. Do NOT predict restaurant or dining hall.'


def predict_location_llm(weekday, slot, hour, schedule, home):
    full_schedule = build_schedule_str(schedule, home)
    this_slot     = schedule.get((weekday, slot), {})
    this_slot_str = ', '.join(f'{l}({c}x)' for l, c in sorted(this_slot.items(), key=lambda x: -x[1])[:5]) \
                    if this_slot else 'No recorded visits at this day+time.'
    prompt = (
        f"You are predicting where a Dartmouth College student currently is.\n\n"
        f"{full_schedule}\n\n---\n"
        f"QUERY: {weekday}, time slot {slot}\n"
        f"Data for this exact slot: {this_slot_str}\n\n"
        f"IMPORTANT - Time: {hours_open_context(hour)}\n---\n\n"
        f"Give the single most likely location. If nighttime with no signal, prefer home ({home}).\n\n"
        f"Reply exactly:\nLOCATION: <name>\nCONFIDENCE: <High|Medium|Low>"
    )
    try:
        text = generate(prompt, max_new_tokens=60)
        loc  = re.search(r'LOCATION:\s*(.+)', text)
        conf = re.search(r'CONFIDENCE:\s*(High|Medium|Low)', text, re.IGNORECASE)
        location = loc.group(1).strip() if loc else home
        if not location or 'unknown' in location.lower():
            location = home
        return location, (conf.group(1) if conf else 'Low')
    except Exception as e:
        print(f'  LLM error: {e}')
        return home, 'Low'


def frequency_baseline(weekday, slot, schedule, home):
    past = schedule.get((weekday, slot), {})
    return max(past, key=past.get) if past else home


def evaluate(history_df, patterns, n_test_days=7, n_samples=50, seed=42):
    np.random.seed(seed)
    h = history_df.copy()
    h['arrive']  = pd.to_datetime(h['arrive'])
    h['date']    = h['arrive'].dt.date
    h['weekday'] = h['arrive'].dt.day_name()
    h['hour']    = h['arrive'].dt.hour
    h['slot']    = h['hour'].apply(time_slot)
    all_dates  = sorted(h['date'].unique())
    split_date = all_dates[-n_test_days] if len(all_dates) > n_test_days else all_dates[0]
    train = h[h['date'] < split_date]
    test  = h[h['date'] >= split_date]
    print(f'Train: {len(train["date"].unique())} days | Test: {len(test["date"].unique())} days | Test stays: {len(test)}')
    train_schedule = {}
    for (wd, sl), group in train.groupby(['weekday', 'slot']):
        train_schedule[(wd, sl)] = group['location'].value_counts().to_dict()
    home   = patterns['home']
    sample = test.sample(min(n_samples, len(test)), random_state=seed)
    results = []
    print(f'Evaluating {len(sample)} predictions...')
    for _, row in sample.iterrows():
        pred_loc, pred_conf = predict_location_llm(row['weekday'], row['slot'], row['hour'], train_schedule, home)
        base_loc = frequency_baseline(row['weekday'], row['slot'], train_schedule, home)
        actual   = row['location']
        results.append({
            'date': row['date'], 'weekday': row['weekday'], 'slot': row['slot'],
            'actual': actual, 'llm_pred': pred_loc, 'base_pred': base_loc,
            'confidence': pred_conf,
            'llm_correct':  actual.lower() == pred_loc.lower(),
            'base_correct': actual.lower() == base_loc.lower(),
        })
        time.sleep(0.2)
    df     = pd.DataFrame(results)
    n_locs = h['location'].nunique()
    print(f'\n{"="*50}\n  EVALUATION RESULTS\n{"="*50}')
    print(f'  Unique locations  : {n_locs}')
    print(f'  Random baseline   : {1/n_locs:.1%}  (1/{n_locs})')
    print(f'  Frequency baseline: {df.base_correct.mean():.1%}  ({df.base_correct.sum()}/{len(df)})')
    print(f'  LLM accuracy      : {df.llm_correct.mean():.1%}  ({df.llm_correct.sum()}/{len(df)})')
    hi = df[df['confidence'] == 'High']
    if len(hi):
        print(f'  LLM High-conf     : {hi.llm_correct.mean():.1%} over {len(hi)} predictions')
    print(f'\n  By time slot:\n{df.groupby("slot")[["llm_correct","base_correct"]].mean().round(3).to_string()}')
    print(f'\n  By weekday:\n{df.groupby("weekday")[["llm_correct","base_correct"]].mean().round(3).to_string()}')
    return df


#  Uncertainty Experiment 
def add_gps_noise(lat, lon, noise_m):
    if noise_m == 0:
        return lat, lon
    angle = np.random.uniform(0, 2 * np.pi)
    return (lat + (noise_m * np.cos(angle)) / 111_000,
            lon + (noise_m * np.sin(angle)) / (111_000 * np.cos(np.radians(lat))))


def uncertainty_experiment(stay_df, noise_levels_m=NOISE_LEVELS, n_samples=30):
    sample = stay_df.sample(min(n_samples, len(stay_df)), random_state=42).reset_index(drop=True)
    print('Building ground truth with clean GPS...')
    ground_truth = []
    for _, row in sample.iterrows():
        pois = get_nearby_pois(row['lat'], row['lon'])
        loc, _ = name_location_llm(row['lat'], row['lon'], pois, row['arrive'], row['duration_min'])
        ground_truth.append(loc)
        time.sleep(0.3)
    results = {'noise_m': [], 'accuracy': [], 'n_fallback': []}
    for noise in noise_levels_m:
        print(f'Noise level: {noise}m')
        correct, n_fallback = 0, 0
        for i, (_, row) in enumerate(sample.iterrows()):
            plat, plon = add_gps_noise(row['lat'], row['lon'], noise)
            pois = get_nearby_pois(plat, plon, radius_m=max(300, noise))
            pred, _ = name_location_llm(plat, plon, pois, row['arrive'], row['duration_min'])
            if pred == 'Campus Building': n_fallback += 1
            if pred.lower() == ground_truth[i].lower(): correct += 1
            time.sleep(0.3)
        acc = correct / len(sample)
        results['noise_m'].append(noise)
        results['accuracy'].append(acc)
        results['n_fallback'].append(n_fallback)
        print(f'  Accuracy: {acc:.1%}  Fallback: {n_fallback}/{len(sample)}')
    df = pd.DataFrame(results)
    plt.figure(figsize=(7, 4))
    plt.plot(df['noise_m'], df['accuracy'] * 100, 'o-', color='steelblue', linewidth=2)
    for _, row in df.iterrows():
        plt.annotate(f"{row['accuracy']:.0%}", (row['noise_m'], row['accuracy'] * 100),
                     textcoords='offset points', xytext=(0, 8), ha='center', fontsize=9)
    plt.xlabel('GPS noise (meters)')
    plt.ylabel('Location naming accuracy (%)')
    plt.title('LLM Location Naming Accuracy vs GPS Uncertainty\n(Dartmouth StudentLife Dataset)')
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/uncertainty_plot.png', dpi=150)
    plt.show()
    return df


# Main Pipeline 
def run_pipeline_for_user(user_id, gps_df):
    cache_path = f'{RESULTS_DIR}/history_{user_id}.csv'
    if os.path.exists(cache_path):
        print(f'{user_id}: loading from cache')
        history = pd.read_csv(cache_path, parse_dates=['arrive', 'leave'])
    else:
        user_gps = gps_df[gps_df.user == user_id]
        if len(user_gps) < 50:
            print(f'{user_id}: too few GPS rows ({len(user_gps)}), skipping')
            return None
        stays = detect_stay_points(user_gps)
        print(f'{user_id}: {len(stays)} stay points detected')
        if len(stays) == 0:
            return None
        history = build_location_history(user_id, stays)
        history.to_csv(cache_path, index=False)
    print_stay_history(user_id, history)
    patterns = build_temporal_schedule(history)
    print_user_profile(user_id, patterns)
    eval_df = evaluate(history, patterns, n_test_days=7, n_samples=N_EVAL_SAMPLES)
    eval_df.to_csv(f'{RESULTS_DIR}/eval_{user_id}.csv', index=False)
    return {'user': user_id, 'history': history, 'patterns': patterns, 'eval': eval_df}


if __name__ == '__main__':
    gps = load_gps()

    all_results = {}
    for uid in USERS_TO_RUN:
        print(f'\n{"="*45}\n  USER: {uid}\n{"="*45}')
        res = run_pipeline_for_user(uid, gps)
        if res:
            all_results[uid] = res

    # Multi-user summary
    summary = []
    for uid, res in all_results.items():
        df    = res['eval']
        n_loc = res['history']['location'].nunique()
        summary.append({
            'user':             uid,
            'n_stays':          len(res['history']),
            'n_locations':      n_loc,
            'random_baseline':  f'{1/n_loc:.1%}',
            'freq_baseline':    f"{df.base_correct.mean():.1%}",
            'llm_accuracy':     f"{df.llm_correct.mean():.1%}",
        })
    summary_df = pd.DataFrame(summary)
    print(f'\n{"="*55}\n  MULTI-USER SUMMARY\n{"="*55}')
    print(summary_df.to_string(index=False))
    summary_df.to_csv(f'{RESULTS_DIR}/summary.csv', index=False)

    # Uncertainty experiment
    if RUN_UNCERTAINTY and all_results:
        first_uid   = USERS_TO_RUN[0]
        stays_first = detect_stay_points(gps[gps.user == first_uid])
        print(f'\nRunning uncertainty experiment on {first_uid} ({len(stays_first)} stays)...')
        unc_df = uncertainty_experiment(stays_first)
        unc_df.to_csv(f'{RESULTS_DIR}/uncertainty_{first_uid}.csv', index=False)
        print('Uncertainty experiment complete.')
