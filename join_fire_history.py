"""
join_fire_history.py
────────────────────
Joins fire_history_grid.csv to the original RF dataset, replacing the
old fire columns (fire_hotspot_count, fire_total_frp, fire_mean_confidence,
fire_max_brightness, fire_occurred, fire_data_available) with the new
ground-truth columns from the fire history shapefile.
"""

import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
ORIGINAL_CSV    = '/Users/liam/Desktop/ENGG2112/Final/rainfall_vegetation_fire_solar_temp_joined.csv'
FIRE_HISTORY_CSV= '/Users/liam/Desktop/ENGG2112/Final/fire_history_grid.csv'
OUTPUT_CSV      = '/Users/liam/Desktop/ENGG2112/Final/rainfall_vegetation_fire_solar_temp_final.csv'

# ── 1. Load both datasets ─────────────────────────────────────────────────────
print("Loading datasets ...")
orig = pd.read_csv(ORIGINAL_CSV)
fire = pd.read_csv(FIRE_HISTORY_CSV)
print(f"  Original : {orig.shape[0]:,} rows × {orig.shape[1]} columns")
print(f"  Fire hist: {fire.shape[0]:,} rows × {fire.shape[1]} columns")

# ── 2. Drop old fire columns from original ────────────────────────────────────
old_fire_cols = [
    'fire_hotspot_count',
    'fire_total_frp',
    'fire_mean_confidence',
    'fire_max_brightness',
    'fire_occurred',
    'fire_data_available',
]
old_fire_cols = [c for c in old_fire_cols if c in orig.columns]
print(f"\nDropping old fire columns: {old_fire_cols}")
orig = orig.drop(columns=old_fire_cols)

# ── 3. Merge on grid_id + year + month ────────────────────────────────────────
print("\nMerging ...")
# Drop duplicate spatial columns from fire_history (already in original)
fire = fire.drop(columns=['lat_centre', 'lon_centre'], errors='ignore')

merged = orig.merge(fire, on=['grid_id', 'year', 'month'], how='left')
print(f"  Merged shape: {merged.shape[0]:,} rows × {merged.shape[1]} columns")

# ── 4. Fill any unmatched rows (grid/months outside fire history range) ────────
merged['fire_occurred'] = merged['fire_occurred'].fillna(0).astype(int)
merged['num_fires']     = merged['num_fires'].fillna(0).astype(int)
if 'total_area_ha' in merged.columns:
    merged['total_area_ha'] = merged['total_area_ha'].fillna(0)

# ── 5. Sort to match original order ──────────────────────────────────────────
merged = merged.sort_values(['lon_centre', 'lat_centre', 'year', 'month']).reset_index(drop=True)

# ── 6. Save ───────────────────────────────────────────────────────────────────
merged.to_csv(OUTPUT_CSV, index=False)
print(f"\n✓ Saved → {OUTPUT_CSV}")
print(f"  Rows    : {len(merged):,}")
print(f"  Columns : {merged.columns.tolist()}")
print(f"\n  fire_occurred=1: {merged['fire_occurred'].sum():,}")
print(f"  fire_occurred=0: {(merged['fire_occurred']==0).sum():,}")
print(f"\nSample:")
print(merged[merged['fire_occurred']==1].head(3).to_string(index=False))
