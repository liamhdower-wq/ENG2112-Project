"""
fire_map.py — Interactive fire probability map
Uses GeoJson rectangles so cells align exactly with the grid.
Time-series uses a custom HTML/JS slider (no HeatMapWithTime).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import folium
import json
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_CSV        = '/Users/liam/Desktop/ENGG2112/Final/fire_probability_predictions.csv'
OUTPUT_HTML      = '/Users/liam/Desktop/ENGG2112/Final/fire_map_interactive.html'
OUTPUT_TIME_HTML = '/Users/liam/Desktop/ENGG2112/Final/fire_map_timeseries.html'
OUTPUT_PNG       = '/Users/liam/Desktop/ENGG2112/Final/fire_map_static.png'

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading predictions ...")
df = pd.read_csv(INPUT_CSV)
print(f"  {len(df):,} rows | prob range: {df['fire_probability'].min():.4f} – {df['fire_probability'].max():.4f}")

map_lat = df['lat_centre'].mean()
map_lon = df['lon_centre'].mean()

lats     = np.sort(df['lat_centre'].unique())
lons     = np.sort(df['lon_centre'].unique())
half_lat = abs(float(np.median(np.diff(lats)))) / 2
half_lon = float(np.median(np.diff(lons))) / 2
print(f"  Grid cell size: {half_lat*2:.4f}° lat × {half_lon*2:.4f}° lon")

sw = [df['lat_centre'].min() - half_lat, df['lon_centre'].min() - half_lon]
ne = [df['lat_centre'].max() + half_lat, df['lon_centre'].max() + half_lon]

# ── Helper: probability → hex colour (white → yellow → orange → red) ─────────
def prob_to_hex(p, alpha=True):
    cmap = plt.cm.YlOrRd
    r, g, b, a = cmap(float(p))
    if alpha:
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}{int(a*0.75*255):02x}"
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

# ── 2. Average-probability GeoJSON (all months) ───────────────────────────────
print("\nBuilding average probability GeoJSON ...")
avg = (df.groupby(['lat_centre', 'lon_centre'])['fire_probability']
         .mean().reset_index())

def make_geojson(data_df):
    features = []
    for r in data_df.itertuples():
        lat, lon, prob = r.lat_centre, r.lon_centre, r.fire_probability
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lon - half_lon, lat - half_lat],
                    [lon + half_lon, lat - half_lat],
                    [lon + half_lon, lat + half_lat],
                    [lon - half_lon, lat + half_lat],
                    [lon - half_lon, lat - half_lat],
                ]]
            },
            "properties": {
                "prob": round(prob, 4),
                "color": prob_to_hex(prob)
            }
        })
    return {"type": "FeatureCollection", "features": features}

avg_geojson = make_geojson(avg)

# ── 3. Interactive map ────────────────────────────────────────────────────────
print("Building interactive map ...")
m = folium.Map(location=[map_lat, map_lon], zoom_start=7,
               tiles='CartoDB positron')
m.fit_bounds([sw, ne])

folium.GeoJson(
    avg_geojson,
    name='Fire probability grid',
    style_function=lambda f: {
        'fillColor'  : f['properties']['color'],
        'fillOpacity': 0.75,
        'color'      : 'none',
        'weight'     : 0
    },
    tooltip=folium.GeoJsonTooltip(
        fields=['prob'],
        aliases=['Fire probability:'],
        localize=True
    )
).add_to(m)

folium.LayerControl().add_to(m)
m.save(OUTPUT_HTML)
print(f"  ✓ Saved → {OUTPUT_HTML}")

# ── 4. Time-series map (custom HTML slider) ───────────────────────────────────
print("\nBuilding time-series map ...")
df['date_label'] = (df['year'].astype(str) + '-' +
                    df['month'].astype(str).str.zfill(2))
months_sorted = sorted(df['date_label'].unique())
print(f"  {len(months_sorted)} months: {months_sorted[0]} → {months_sorted[-1]}")

# Build one GeoJSON per month, store as JS object
months_js = {}
for month in months_sorted:
    month_df = df[df['date_label'] == month][['lat_centre','lon_centre','fire_probability']].copy()
    month_df.columns = ['lat_centre', 'lon_centre', 'fire_probability']
    months_js[month] = make_geojson(month_df)

# Base map HTML
m3 = folium.Map(location=[map_lat, map_lon], zoom_start=7,
                tiles='CartoDB positron')
m3.fit_bounds([sw, ne])
base_html = m3.get_root().render()

# Inject geojson data + custom slider UI into the page
months_json_str = json.dumps(months_js)
map_var = m3.get_name()

inject = f"""
<script>
var allMonths = {months_json_str};
var months = {json.dumps(months_sorted)};
var currentLayer = null;

function loadMonth(idx) {{
    var map = {map_var};
    if (currentLayer) {{ map.removeLayer(currentLayer); }}
    var month = months[idx];
    var geojson = allMonths[month];
    currentLayer = L.geoJson(geojson, {{
        style: function(f) {{
            return {{
                fillColor: f.properties.color,
                fillOpacity: 0.75,
                color: 'transparent',
                weight: 0
            }};
        }},
        onEachFeature: function(f, layer) {{
            layer.bindTooltip('Fire probability: ' + f.properties.prob);
        }}
    }}).addTo(map);
    document.getElementById('month-label').innerText = month;
    document.getElementById('month-slider').value = idx;
}}

window.addEventListener('load', function() {{ loadMonth(0); }});
</script>

<div style="
    position: fixed;
    bottom: 30px;
    left: 50%;
    transform: translateX(-50%);
    background: white;
    padding: 12px 20px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    z-index: 9999;
    font-family: Arial, sans-serif;
    text-align: center;
    min-width: 400px;
">
    <div style="font-size:15px;font-weight:bold;margin-bottom:8px">
        Fire Probability — <span id="month-label">--</span>
    </div>
    <input
        id="month-slider"
        type="range"
        min="0"
        max="{len(months_sorted)-1}"
        value="0"
        step="1"
        style="width:100%;cursor:pointer"
        oninput="loadMonth(parseInt(this.value))"
    >
    <div style="display:flex;justify-content:space-between;font-size:11px;color:#666;margin-top:4px">
        <span>{months_sorted[0]}</span>
        <span>{months_sorted[-1]}</span>
    </div>
    <div style="margin-top:8px;display:flex;justify-content:center;gap:8px">
        <button onclick="step(-1)"
            style="padding:4px 14px;border-radius:6px;border:1px solid #ccc;cursor:pointer">
            ◀ Prev
        </button>
        <button onclick="togglePlay()"
            id="play-btn"
            style="padding:4px 14px;border-radius:6px;border:1px solid #ccc;cursor:pointer">
            ▶ Play
        </button>
        <button onclick="step(1)"
            style="padding:4px 14px;border-radius:6px;border:1px solid #ccc;cursor:pointer">
            Next ▶
        </button>
    </div>
</div>

<script>
var playing = false;
var playInterval = null;
function step(dir) {{
    var s = document.getElementById('month-slider');
    var next = Math.min(Math.max(parseInt(s.value) + dir, 0), {len(months_sorted)-1});
    loadMonth(next);
}}
function togglePlay() {{
    playing = !playing;
    document.getElementById('play-btn').innerText = playing ? '⏸ Pause' : '▶ Play';
    if (playing) {{
        playInterval = setInterval(function() {{
            var s = document.getElementById('month-slider');
            var next = parseInt(s.value) + 1;
            if (next > {len(months_sorted)-1}) next = 0;
            loadMonth(next);
        }}, 800);
    }} else {{
        clearInterval(playInterval);
    }}
}}
</script>
"""

# Write final HTML
final_html = base_html.replace('</body>', inject + '</body>')
with open(OUTPUT_TIME_HTML, 'w') as f:
    f.write(final_html)
print(f"  ✓ Saved → {OUTPUT_TIME_HTML}")

# ── 5. Static PNG ─────────────────────────────────────────────────────────────
print("\nBuilding static PNG ...")
avg_dict  = avg.set_index(['lat_centre', 'lon_centre'])['fire_probability'].to_dict()
peak_dict = (df.groupby(['lat_centre', 'lon_centre'])['fire_probability']
               .max().reset_index()
               .set_index(['lat_centre', 'lon_centre'])['fire_probability'].to_dict())

fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor='#F8F9FA')
cmap = plt.cm.YlOrRd

for ax, prob_dict, title in [
    (axes[0], avg_dict,  'Average Fire Probability\n(all months)'),
    (axes[1], peak_dict, 'Peak Fire Probability\n(worst month per cell)')
]:
    ax.set_facecolor('#D6EAF8')
    vals = list(prob_dict.values())
    norm = mcolors.Normalize(vmin=0, vmax=max(vals))
    for (lat, lon), prob in prob_dict.items():
        ax.add_patch(Rectangle(
            (lon - half_lon, lat - half_lat),
            half_lon * 2, half_lat * 2,
            linewidth=0, facecolor=cmap(norm(prob))
        ))
    ax.set_xlim(df['lon_centre'].min() - half_lon, df['lon_centre'].max() + half_lon)
    ax.set_ylim(df['lat_centre'].min() - half_lat, df['lat_centre'].max() + half_lat)
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                 ax=ax, shrink=0.7, label='Fire Probability')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
print(f"  ✓ Saved → {OUTPUT_PNG}")
plt.show()

print(f"\n✓ All done:")
print(f"  Interactive map  → {OUTPUT_HTML}")
print(f"  Time-series map  → {OUTPUT_TIME_HTML}")
print(f"  Static PNG       → {OUTPUT_PNG}")