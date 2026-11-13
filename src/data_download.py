"""
[COMPONENT 1a: DATA -- acquisition]

Downloads the raw GHCN-Daily record for the chosen station from NOAA NCEI and
saves it, unmodified, to data/raw/. The raw file is never edited; every
cleaning step happens in data_clean.py so it can be re-run and audited.

Source: NOAA NCEI Global Historical Climatology Network Daily (GHCN-Daily)
https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily

Run alone:  python -m src.data_download
"""
import urllib.request
from datetime import datetime, timezone

from src import config


def download(force: bool = False) -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if config.RAW_FILE.exists() and not force:
        print(f"[data] raw file already present: {config.RAW_FILE.name} (use --force to re-download)")
        return
    print(f"[data] downloading {config.NOAA_URL}")
    urllib.request.urlretrieve(config.NOAA_URL, config.RAW_FILE)
    # provenance note kept next to the raw file
    (config.RAW_DIR / "DOWNLOAD_LOG.txt").write_text(
        f"source: {config.NOAA_URL}\n"
        f"station: {config.STATION_ID} {config.STATION_NAME}\n"
        f"downloaded_utc: {datetime.now(timezone.utc).isoformat()}\n"
        f"bytes: {config.RAW_FILE.stat().st_size}\n"
    )
    print(f"[data] saved {config.RAW_FILE} ({config.RAW_FILE.stat().st_size:,} bytes)")


if __name__ == "__main__":
    import sys
    download(force="--force" in sys.argv)
