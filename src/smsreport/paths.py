"""Central place for data/result locations. Override with environment variables if needed."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("SMSREPORT_DATA", ROOT / "data"))
FMC_DIR = Path(os.environ.get("FMC_DIR", DATA / "mwo2kg-and-echidna" / "mwo2kg" / "failure_mode_classification" / "input_data"))
THERMAL_DIR = Path(os.environ.get("THERMAL_DIR", DATA / "thermal-images-equip"))   # contains ex_motor/ and ex_trans/
CWRU_DIR = Path(os.environ.get("CWRU_DIR", DATA / "CWRU-dataset"))
RESULTS = Path(os.environ.get("SMSREPORT_RESULTS", ROOT / "results"))
CACHE = Path(os.environ.get("SMSREPORT_CACHE", ROOT / "cache"))
for d in (RESULTS, CACHE, RESULTS / "curves"):
    d.mkdir(parents=True, exist_ok=True)
