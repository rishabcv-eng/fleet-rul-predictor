"""
Download the real NASA C-MAPSS turbofan degradation dataset.

The project ships with a synthetic generator so it runs anywhere, but the
real benchmark is better for a portfolio: your numbers become comparable to
published results. Run this on your own machine (it needs internet access),
then re-run training -- no other changes required, the file format matches.

    python scripts/download_data.py

If the automatic download fails (NASA moves these files around fairly
often), grab the data manually from:

    https://www.nasa.gov/intelligent-systems-division/  -> Prognostics Data Repository
    or Kaggle: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

and unzip it so that data/raw/train_FD001.txt exists.
"""

import io
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Mirrors, tried in order. The NASA repository URL changes periodically.
SOURCES = [
    "https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip",
    "https://ti.arc.nasa.gov/c/6/",
]

WANTED = ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")


def try_download(url: str) -> bytes | None:
    print(f"  trying {url} ...", end=" ", flush=True)
    try:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=60) as response:
            payload = response.read()
        print(f"ok ({len(payload) / 1_048_576:.1f} MB)")
        return payload
    except Exception as error:  # noqa: BLE001 - we genuinely want any failure
        print(f"failed ({type(error).__name__})")
        return None


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading NASA C-MAPSS dataset...")

    payload = next((data for url in SOURCES if (data := try_download(url))), None)

    if payload is None:
        print(
            "\nCould not download automatically. This is common -- NASA "
            "reorganises\nthis repository from time to time. Two options:\n"
            "\n  1. Download manually (see the docstring at the top of this "
            "file)\n  2. Keep using the synthetic data: python "
            "scripts/generate_data.py\n"
        )
        return 1

    extracted = 0
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.namelist():
            name = Path(member).name
            if name in WANTED:
                (RAW_DIR / name).write_bytes(archive.read(member))
                print(f"  extracted {name}")
                extracted += 1

    if extracted == 0:
        print("Archive downloaded but the expected FD001 files were not inside.")
        return 1

    print(f"\nDone. {extracted} files in {RAW_DIR}")
    print("Now run: python -m backend.train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
