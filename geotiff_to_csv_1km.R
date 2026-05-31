# GeoTIFF to CSV Converter with Resampling
# Resamples from 10m to 1km resolution before converting to CSV
#
# Install required packages (run once):
# install.packages(c("terra", "dplyr"))

library(terra)
library(dplyr)

geotiff_to_csv_resampled <- function(tif_path, output_csv = NULL,
                                     target_res_m = 25000,
                                     skip_nodata = TRUE) {
  
  if (is.null(output_csv)) {
    output_csv <- sub("\\.tif$", "_25km.csv", tif_path, ignore.case = TRUE)
  }
  
  cat("\nProcessing:", basename(tif_path), "\n")
  
  # Load raster
  r <- rast(tif_path)
  
  cat("  Original dimensions:", ncol(r), "x", nrow(r), "pixels\n")
  cat("  Original resolution:", res(r)[1], "m\n")
  
  # Reproject to a metre-based CRS for accurate resampling (GDA2020 / MGA for Australia)
  # This ensures "1km" is actually 1km on the ground
  cat("  Reprojecting to GDA2020 (EPSG:7855) for resampling...\n")
  r_projected <- project(r, "EPSG:7855", method = "mode")
  # method = "mode" is best for categorical land cover data —
  # it picks the most common class in each new cell rather than averaging codes
  
  # Resample to target resolution
  cat("  Resampling to", target_res_m, "m resolution...\n")
  r_resampled <- aggregate(r_projected, fact = round(target_res_m / res(r_projected)[1]),
                           fun = "modal")  # modal = most frequent class per cell
  
  cat("  New dimensions:", ncol(r_resampled), "x", nrow(r_resampled), "pixels\n")
  
  # Reproject back to WGS84 (EPSG:4326) for lat/lon output
  cat("  Reprojecting to WGS84 for lat/lon output...\n")
  r_wgs84 <- project(r_resampled, "EPSG:4326", method = "mode")
  
  # Convert to data frame
  df <- as.data.frame(r_wgs84, xy = TRUE, na.rm = skip_nodata)
  names(df)[1:2] <- c("longitude", "latitude")
  if (nlyr(r_wgs84) == 1) names(df)[3] <- "value"
  
  cat("  Rows in CSV:", format(nrow(df), big.mark = ","), "\n")
  
  write.csv(df, output_csv, row.names = FALSE)
  cat("  Saved ->", output_csv, "\n")
  
  return(output_csv)
}

# ── Edit these paths ──────────────────────────────────────────────────────────

tif_file <- file.path(path.expand("~"), "Desktop/ENGG2112/Data/Left_tile_2019.tif")


# ─────────────────────────────────────────────────────────────────────────────

geotiff_to_csv_resampled(tif_file, target_res_m = 25000)

cat("\nDone! CSV files saved with '_25km' suffix alongside your TIF files.\n")
