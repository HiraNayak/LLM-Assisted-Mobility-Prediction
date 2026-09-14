# LLM-Assisted Wireless Localization

Predicting a user's next location using a Large Language Model (LLM) conditioned on their historical stay patterns that outperforms a frequency baseline by up to **+11.6 percentage points**.

## Overview

Traditional location prediction relies on frequency heuristics ie predict wherever the user has spent the most time. This project replaces that heuristic with an LLM that reasons over a user's named stay history (e.g., "Baker-Berry Library", "Phi Tau", "Ripley Hall") and predicts the next location given the current time slot and day of week.

**Pipeline:**
1. **Stay Point Detection** - GPS traces : stay points using Haversine distance (100m radius, 15 min minimum dwell)
2. **POI Naming** - Stay coordinates : human-readable location names via OSM Overpass API and Nominatim fallback
3. **LLM Prediction** - Groq-hosted LLM predicts next stay given history and time context
4. **Evaluation** - Compared against frequency baseline and random baseline on last-7-days test split

---

## Dataset

[Dartmouth StudentLife Spring 2013](http://studentlife.cs.dartmouth.edu/) : GPS traces from university students over a full semester.

- 13 users processed (u01-u13)
- 63-454 stay points per user
- 7-62 unique locations per user
- Train/test split: last 7 days held out as test set

**User profile:** University students tracked over a full academic semester. Stay history spans academic locations (libraries, lecture halls), residential (dorms, fraternities), and social/dining spots giving the LLM rich semantic patterns to reason over.


## Results

| User | Test Stays | Unique Locs | LLM Acc | Freq Acc | Random | **LLM vs Freq** |
|------|-----------|-------------|---------|----------|--------|----------------|
| u01  | 26        | 32          | **38.5%** | 26.9% | 3.1%   | **+11.6pp (+43% relative)** |
| u02  | 39        | 50          | **28.2%** | 25.6% | 2.0%   | **+2.6pp**  |
| u03  | 9         | 7           | **77.8%** | 77.8% | 14.3%  | 0pp(matches baseline - low diversity) |

**Key finding:** LLM achieves up to **43% relative improvement** over the frequency baseline (u01: 38.5% vs 26.9%). Performance is strongest at moderate location diversity (30-50 unique locations).


## Visualizations

### GPS Trajectory with Stay Points
GPS traces for a single user across all days, with detected pause/stay locations marked.

![Trajectory with Pauses](figures/trajectory_with_pauses.png)

### Stay Detection with Uncertainty Radius
Each detected stay point shown with a 100m uncertainty radius which is the threshold used in Haversine-based clustering.

![Uncertainty 100m](figures/uncertainty_100m.png)

---

## Tech Stack

- **LLM:** Groq API - results reported using `llama-3.1-8b-instant` (now discontinued; code defaults to `groq/compound-mini`)
- **GPS Processing:** Custom Haversine stay-point detector (Python)
- **POI Naming:** OpenStreetMap Overpass API + Nominatim fallback
- **Mapping:** Folium / GPS coordinate plotting
- **Dataset:** Dartmouth StudentLife 2013

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/llm-mobility-prediction
cd llm-mobility-prediction
pip install -r requirements.txt
```

Set your Groq API key:
```bash
export GROQ_API_KEY=your_key_here
```

Run the pipeline:
```bash
python llm_localization.py
```

---

## Project Structure

```
llm-location-prediction/
├── llm_localization.py       # Main pipeline  runs end-to-end
├── stay_detection.py         # Haversine-based stay point detector
├── poi_naming.py             # OSM Overpass + Nominatim POI lookup
├── evaluate.py               # LLM vs freq vs random baseline eval
├── requirements.txt
├── data/                     # StudentLife GPS data (not included - see dataset link)
├── cache/                    # Named stay cache, auto-created (JSON per user)
├── results/                  # Per-user accuracy results (JSON)
└── figures/                  # Trajectory and stay visualizations
```

---

## Author

Hira - MS ECE, Carnegie Mellon University  
Research in LLM-based mobility prediction and wireless localization.
