"""GeoDatum converter: Nigerian Minna-datum grid coordinates -> WGS84 latitude/longitude.

Built on pyproj (PROJ). Grids and EPSG codes (all present in pyproj's database):
  UTM 31N = 26331, UTM 32N = 26332, West Belt = 26391, Mid Belt = 26392, East Belt = 26393.
Note: there is no 'Minna / UTM zone 33N' in the registry.

The 'inside Nigeria' test is a coarse bounding box, NOT a border polygon.
Run `python geodatum_converter.py` to self-test against known values.
"""
import math
from pyproj import Transformer

GRIDS = {
    "UTM 31N": 26331,
    "UTM 32N": 26332,
    "West Belt": 26391,
    "Mid Belt": 26392,
    "East Belt": 26393,
}
BBOX = (4.0, 14.0, 2.5, 14.8)  # lat_min, lat_max, lon_min, lon_max

_TO_WGS84 = {g: Transformer.from_crs(c, 4326, always_xy=True) for g, c in GRIDS.items()}
_FROM_WGS84 = {g: Transformer.from_crs(4326, c, always_xy=True) for g, c in GRIDS.items()}


def inside_nigeria_bbox(lat, lon):
    lat_min, lat_max, lon_min, lon_max = BBOX
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def convert_to_latlon(easting, northing, grid, datum="Minna"):
    """The tool the model calls. Returns a dict (JSON-serialisable)."""
    if datum != "Minna":
        return {"error": f"Unsupported datum '{datum}'. Only Minna is supported."}
    if grid not in GRIDS:
        return {"error": f"Unsupported grid '{grid}'. Supported: {', '.join(GRIDS)}."}
    lon, lat = _TO_WGS84[grid].transform(easting, northing)
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return {"error": "Conversion failed for these coordinates."}
    return {
        "latitude": round(lat, 5),
        "longitude": round(lon, 5),
        "grid": grid,
        "datum_in": "Minna",
        "datum_out": "WGS84",
        "inside_nigeria_bbox": inside_nigeria_bbox(lat, lon),
    }


def latlon_to_grid(lat, lon, grid):
    """Used only to make realistic example inputs."""
    x, y = _FROM_WGS84[grid].transform(lon, lat)
    return x, y


def _selftest():
    # Values printed by pyproj in Colab on 10 Nov 2026 for E=600000, N=892340
    expected = {
        "UTM 31N": (8.07282, 3.90677),
        "UTM 32N": (8.07283, 9.90687),
        "West Belt": (12.05042, 7.88961),
        "Mid Belt": (12.07030, 7.85110),
        "East Belt": (12.03168, 7.81597),
    }
    ok = True
    for grid, (lat, lon) in expected.items():
        r = convert_to_latlon(600000, 892340, grid)
        good = abs(r["latitude"] - lat) < 2e-5 and abs(r["longitude"] - lon) < 2e-5
        ok &= good
        print(f"{grid:10s} {r['latitude']:.5f} {r['longitude']:.5f}  {'OK' if good else 'MISMATCH'}")
    t = _TO_WGS84["UTM 31N"]
    print("Transformation used:", t.description, "| accuracy:", t.accuracy)
    print("ALL OK" if ok else "SOME VALUES DIFFER")


if __name__ == "__main__":
    _selftest()
