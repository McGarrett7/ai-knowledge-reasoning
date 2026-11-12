"""SolarKG-VN construction (Sec. 4 and 7.1-7.2).

raster  --sampling-->  solar_dataset.csv  --triples-->  SolarKG-VN.csv

The raster can be either
  * the Global Solar Atlas GIS GeoTIFF (single-band float32 PVOUT, georeferenced),
    read with rasterio; nodata cells are skipped and lon/lat are stored; or
  * the bundled poster map PVOUT.tif (RGBA, 8-bit, no georeference). Band 1 of
    that file is the red colour channel of the rendered map, so its values are
    8-bit colour indices, not kWh/kWp. It is kept only so the published
    SolarKG-VN statistics can be reproduced. A warning is printed when it is used.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from config import KG_CSV, RASTER_PATH, REL_POTENTIAL, REL_PRODUCES, SOLAR_CSV, SOLAR_ENERGY, DataConfig


def _read_band(path):
    """Return (band as 2-D array, nodata mask, transform or None, is_gis)."""
    try:
        import rasterio
        with rasterio.open(path) as ds:
            band = ds.read(1, masked=True)
            is_gis = ds.crs is not None and ds.count == 1
            return np.asarray(band.filled(0)), np.asarray(band.mask), (ds.transform if is_gis else None), is_gis
    except ImportError:
        pass
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        band = np.asarray(im.getchannel(0))
        is_gis = im.mode in ("F", "I", "I;16") and len(im.getbands()) == 1
    return band, np.zeros(band.shape, dtype=bool), None, is_gis


def raster_to_samples(path=RASTER_PATH, n_samples: int = 20000, seed: int = 42, out=SOLAR_CSV) -> pd.DataFrame:
    band, nodata, transform, is_gis = _read_band(path)
    if not is_gis:
        warnings.warn(
            f"{path} is not a single-band georeferenced PVOUT GeoTIFF (poster map?). "
            "Sampled values are colour-channel intensities, not kWh/kWp. Download the GIS "
            "data from https://globalsolaratlas.info/download/vietnam for physical units.")
    rows, cols = band.shape
    rng = np.random.default_rng(seed)
    i = rng.integers(0, rows, size=n_samples)
    j = rng.integers(0, cols, size=n_samples)
    v = band[i, j].astype(float)
    keep = (v > 0) & ~nodata[i, j] & np.isfinite(v)
    df = pd.DataFrame({"x": i[keep], "y": j[keep], "pv_potential": v[keep]})
    if transform is not None:
        lon, lat = transform * (df["y"].values + 0.5, df["x"].values + 0.5)
        df["lon"], df["lat"] = lon, lat
    df.to_csv(out, index=False)
    print(f"Raster {rows}x{cols}; sampled {len(df)} valid points -> {out}")
    return df


def samples_to_triples(samples: pd.DataFrame) -> pd.DataFrame:
    """Template mapping used for the published SolarKG-VN (one location -> 2 triples)."""
    triples = []
    for x, y, v in samples[["x", "y", "pv_potential"]].itertuples(index=False):
        loc = f"location_{int(x)}_{int(y)}"
        triples.append((loc, REL_POTENTIAL, f"{float(v):.1f}"))
        triples.append((loc, REL_PRODUCES, SOLAR_ENERGY))
    return pd.DataFrame(triples, columns=["head", "relation", "tail"])


def build_kg(samples: pd.DataFrame | None = None, cfg: DataConfig = DataConfig(), seed: int = 42,
             out=KG_CSV) -> pd.DataFrame:
    if samples is None:
        samples = pd.read_csv(SOLAR_CSV)
    kg = samples_to_triples(samples).drop_duplicates()
    kg = kg.sample(n=min(cfg.kg_size, len(kg)), random_state=seed).reset_index(drop=True)
    kg.to_csv(out, index=False)
    print(f"SolarKG-VN: {len(kg)} triples, {kg['head'].nunique()} locations -> {out}")
    return kg


def synthetic_upscale(samples: pd.DataFrame, n_locations: int, seed: int = 0) -> pd.DataFrame:
    """Larger KGs for the scalability study: resample real points with spatial
    jitter. These graphs are used for timing only and are labelled synthetic."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(samples), size=n_locations)
    base = samples.iloc[idx]
    x = base["x"].values * 4 + rng.integers(0, 4, size=n_locations)
    y = base["y"].values * 4 + rng.integers(0, 4, size=n_locations)
    df = pd.DataFrame({"x": x, "y": y, "pv_potential": base["pv_potential"].values})
    return df.drop_duplicates(subset=["x", "y"]).reset_index(drop=True)
