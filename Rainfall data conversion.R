# ============================================================
# BOM AGCD Rainfall .nc → Victoria 25 km gridded CSV
# ------------------------------------------------------------
# Filters output to only the lat/lon grid points defined in
# Left_tile_2019_25km.csv and Right_tile_2019_25km.csv
#
# Output columns: grid_id | lat_centre | lon_centre | year | month | rainfall_mm
#
# Install required packages (run once):
#   install.packages(c("tidyverse", "ncdf4", "stringr"))
# ============================================================

library(ncdf4)
library(tidyverse)
library(stringr)

# ── CONFIGURATION ────────────────────────────────────────────
INPUT_DIR   <- "~/Desktop/ENGG2112/Data/Rainfall"  # folder with your 24 .nc files
OUTPUT_CSV  <- "~/Desktop/ENGG2112/Data/Rainfall/rainfall_25km_2019_2020.csv"

# Your two Victoria tile files — update paths if needed
LEFT_TILE   <- "~/Desktop/ENGG2112/Data/Left_tile_2019_25km.csv"
RIGHT_TILE  <- "~/Desktop/ENGG2112/Data/Right_tile_2019_25km.csv"

GRID_STEP   <- 0.225   # degrees ≈ 25 km over Australia

# Rainfall variable name — set to NULL to auto-detect
RAIN_VAR <- NULL

# Tolerance for coordinate matching (degrees)
MATCH_TOL <- 0.13   # half a grid step — catches any native cell within the tile cell
# ─────────────────────────────────────────────────────────────


# ── LOAD VICTORIA GRID POINTS ────────────────────────────────
cat("Loading Victoria grid points from tile files...\n")
vic_points <- bind_rows(
  read_csv(LEFT_TILE,  show_col_types = FALSE),
  read_csv(RIGHT_TILE, show_col_types = FALSE)
) |>
  select(latitude, longitude) |>
  distinct()

cat(sprintf("  Victoria grid points: %s unique lat/lon pairs\n",
            format(nrow(vic_points), big.mark = ",")))
cat(sprintf("  Lat range: %.4f to %.4f\n",
            min(vic_points$latitude), max(vic_points$latitude)))
cat(sprintf("  Lon range: %.4f to %.4f\n\n",
            min(vic_points$longitude), max(vic_points$longitude)))

# ── FIX: derive grid vectors directly from tile files ────────
# Previously these were built with seq() from the bounding box,
# producing only ~13 synthetic points. Now we use the exact
# unique lat/lon values present in the tile files so every
# tile coordinate gets a rainfall value.
grid_lats <- sort(unique(vic_points$latitude))
grid_lons <- sort(unique(vic_points$longitude))

cat(sprintf("  Grid lats: %d unique values (%.4f to %.4f)\n",
            length(grid_lats), min(grid_lats), max(grid_lats)))
cat(sprintf("  Grid lons: %d unique values (%.4f to %.4f)\n\n",
            length(grid_lons), min(grid_lons), max(grid_lons)))

# Bounding box with buffer to crop .nc files before binning
LAT_MIN <- min(vic_points$latitude)  - MATCH_TOL
LAT_MAX <- max(vic_points$latitude)  + MATCH_TOL
LON_MIN <- min(vic_points$longitude) - MATCH_TOL
LON_MAX <- max(vic_points$longitude) + MATCH_TOL


# ── HELPER: detect rainfall variable name ────────────────────
detect_rain_var <- function(var_names) {
  candidates <- c("rain_day", "rain", "precip", "precipitation",
                  "monthly_rainfall", "total_rain", "rainfall")
  for (c in candidates) {
    match <- var_names[str_detect(tolower(var_names), c)]
    if (length(match) > 0) return(match[1])
  }
  skip <- c("lat", "latitude", "lon", "longitude", "time", "crs")
  fallback <- var_names[!tolower(var_names) %in% skip]
  if (length(fallback) > 0) return(fallback[1])
  stop("Cannot identify rainfall variable from: ", paste(var_names, collapse = ", "))
}


# ── HELPER: extract year & month from filename ───────────────
extract_year_month <- function(filepath) {
  name <- tools::file_path_sans_ext(basename(filepath))
  
  m <- str_match(name, "(20[12]\\d)(0[1-9]|1[0-2])")
  if (!is.na(m[1, 1])) return(list(year  = as.integer(m[1, 2]),
                                   month = as.integer(m[1, 3])))
  
  m <- str_match(name, "(20[12]\\d)[_-](0[1-9]|1[0-2])")
  if (!is.na(m[1, 1])) return(list(year  = as.integer(m[1, 2]),
                                   month = as.integer(m[1, 3])))
  
  stop("Cannot parse year/month from: ", filepath)
}


# ── HELPER: bin native cells into 25 km grid ─────────────────
# grid_lats / grid_lons are now the EXACT tile coordinates,
# so every tile point is guaranteed a result row.
bin_to_25km <- function(rain_mat, src_lats, src_lons,
                        grid_lats, grid_lons, step) {
  half <- step / 2
  
  if (src_lats[1] > src_lats[length(src_lats)]) {
    src_lats <- rev(src_lats)
    rain_mat <- rain_mat[nrow(rain_mat):1, ]
  }
  
  records <- vector("list", length(grid_lats) * length(grid_lons))
  idx <- 1L
  
  for (glat in grid_lats) {
    lat_idx <- which(src_lats >= glat - half & src_lats < glat + half)
    if (length(lat_idx) == 0) next
    
    for (glon in grid_lons) {
      lon_idx <- which(src_lons >= glon - half & src_lons < glon + half)
      if (length(lon_idx) == 0) next
      
      block     <- rain_mat[lat_idx, lon_idx]
      mean_rain <- mean(block, na.rm = TRUE)
      
      records[[idx]] <- list(
        lat_centre  = round(glat, 4),
        lon_centre  = round(glon, 4),
        rainfall_mm = round(mean_rain, 3)
      )
      idx <- idx + 1L
    }
  }
  
  bind_rows(records[1:(idx - 1)])
}


# ── HELPER: process a single .nc file ────────────────────────
process_nc_file <- function(filepath, rain_var,
                            grid_lats, grid_lons, step) {
  nc  <- nc_open(filepath)
  on.exit(nc_close(nc))
  
  var_names <- names(nc$var)
  rv <- if (!is.null(rain_var)) rain_var else detect_rain_var(var_names)
  
  rain_raw <- ncvar_get(nc, rv)
  
  lat_name <- intersect(c("lat", "latitude"),  names(nc$dim))[1]
  lon_name <- intersect(c("lon", "longitude"), names(nc$dim))[1]
  src_lats <- ncvar_get(nc, lat_name)
  src_lons <- ncvar_get(nc, lon_name)
  
  fill_val <- ncatt_get(nc, rv, "_FillValue")$value
  miss_val <- ncatt_get(nc, rv, "missing_value")$value
  if (!is.null(fill_val)) rain_raw[rain_raw == fill_val] <- NA
  if (!is.null(miss_val)) rain_raw[rain_raw == miss_val] <- NA
  
  if (length(dim(rain_raw)) == 3) rain_raw <- rain_raw[, , 1]
  rain_mat <- t(rain_raw)
  
  # Crop to Victoria bounding box before binning (faster)
  lat_keep <- which(src_lats >= LAT_MIN & src_lats <= LAT_MAX)
  lon_keep <- which(src_lons >= LON_MIN & src_lons <= LON_MAX)
  src_lats <- src_lats[lat_keep]
  src_lons <- src_lons[lon_keep]
  rain_mat <- rain_mat[lat_keep, lon_keep]
  
  bin_to_25km(rain_mat, src_lats, src_lons, grid_lats, grid_lons, step)
}


# ── HELPER: match rainfall grid to Victoria tile points ──────
# bin_to_25km now uses the exact tile coordinates as grid centres,
# so this is a clean direct join rather than a nearest-neighbour
# approximation. The tolerance filter is kept as a safety net.
match_to_vic_points <- function(df_grid, vic_points, tol = MATCH_TOL) {
  vic_points |>
    rowwise() |>
    mutate(
      lat_match = df_grid$lat_centre[which.min(abs(df_grid$lat_centre - latitude))],
      lon_match = df_grid$lon_centre[which.min(abs(df_grid$lon_centre - longitude))]
    ) |>
    ungroup() |>
    left_join(df_grid, by = c("lat_match" = "lat_centre",
                              "lon_match" = "lon_centre")) |>
    filter(abs(lat_match - latitude) <= tol,
           abs(lon_match - longitude) <= tol) |>
    select(lat_centre = latitude, lon_centre = longitude, rainfall_mm)
}


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

nc_files <- list.files(INPUT_DIR, pattern = "\\.nc$", full.names = TRUE)

if (length(nc_files) == 0) {
  stop("No .nc files found in '", INPUT_DIR, "'.")
}
cat(sprintf("Found %d .nc files\n\n", length(nc_files)))

all_records <- vector("list", length(nc_files))

for (i in seq_along(nc_files)) {
  fp <- nc_files[i]
  ym <- extract_year_month(fp)
  cat(sprintf("  [%2d/%d]  %-50s  →  %d-%02d\n",
              i, length(nc_files), basename(fp), ym$year, ym$month))
  
  df_grid <- process_nc_file(fp, RAIN_VAR, grid_lats, grid_lons, GRID_STEP)
  
  # Filter to Victoria tile points only
  df_vic <- match_to_vic_points(df_grid, vic_points)
  
  df_vic$year  <- ym$year
  df_vic$month <- ym$month
  
  all_records[[i]] <- df_vic
}

# Combine and tidy
result <- bind_rows(all_records) |>
  mutate(grid_id = sprintf("lat%.4f_lon%.4f", lat_centre, lon_centre)) |>
  select(grid_id, lat_centre, lon_centre, year, month, rainfall_mm) |>
  arrange(year, month, lat_centre, lon_centre)

write_csv(result, OUTPUT_CSV)

cat(sprintf("\n✓  Saved %s rows → %s\n",
            format(nrow(result), big.mark = ","), OUTPUT_CSV))
cat(sprintf("   Unique grid cells : %s  (expected ~%d)\n",
            format(n_distinct(result$grid_id), big.mark = ","),
            nrow(vic_points)))
cat(sprintf("   Months covered    : %d\n",
            nrow(distinct(result, year, month))))
cat(sprintf("   Rainfall range    : %.1f – %.1f mm\n",
            min(result$rainfall_mm, na.rm = TRUE),
            max(result$rainfall_mm, na.rm = TRUE)))
cat("\nSample rows:\n")
print(head(result, 5))