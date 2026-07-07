"""
btc_onchain.py
==============
Downloads Bitcoin on-chain indicators at daily frequency from the
BGeometrics free API (bitcoin-data.com/v1/).

Free-tier indicators (last 4 years of data)
--------------------------------------------
1. MVRV Z-Score   – GET /v1/mvrv-zscore
2. NUPL           – GET /v1/nupl
3. SOPR           – GET /v1/sopr
4. NVT Signal     – GET /v1/nvts   (NVT ratio; closest free equivalent)
5. Exchange Netflow – PAID TIER ONLY (Advanced plan, $8/mo)
   → Replaced with Puell Multiple (GET /v1/puell-multiple), a free
     miner-flow indicator that also signals market cycle extremes.

NOTE on Exchange Netflow
------------------------
Exchange flow metrics (inflow, outflow, netflow, reserves) require the
Advanced plan at portal.bgeometrics.com/pricing. The free tier returns
404 for all exchange-flow endpoints regardless of slug. If you upgrade,
re-add the entry:
    ("exchange-netflow", "Exchange Netflow", "bitcoin-data.com", "netflow"),

Output
------
Each indicator is saved as a CSV file in ./data/ and a snapshot of the
latest value for each is printed to the console.

API key
-------
No token is required for the free tier on bitcoin-data.com. The .env
key (BGEOMETRICS_API_KEY) is still used if present (bearer header),
but the endpoints work without it on the free plan.

Free-tier limits: 8 requests/hour · 15 requests/day.
This script makes exactly 5 requests per run.
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
from dotenv import load_dotenv

# ── Configuration ─────────────────────────────────────────────────────────────

load_dotenv()
API_KEY    = os.getenv("BGEOMETRICS_API_KEY", "")
BASE_URL   = "https://bitcoin-data.com/v1"
DATA_DIR   = Path("data")
PAUSE_SEC  = 12     # polite gap — free tier allows ~8 req/hr
RETRY_MAX  = 3      # retries on 429
RETRY_BASE = 30     # seconds before first retry; doubles each time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Endpoint definitions ───────────────────────────────────────────────────────
# Tuple: (slug, friendly_name, value_column_hint)
# All endpoints confirmed on the Free plan at bitcoin-data.com/v1/

INDICATORS = [
    ("mvrv-zscore",    "MVRV Z-Score",    "mvrv"),
    ("nupl",           "NUPL",            "nupl"),
    ("sopr",           "SOPR",            "sopr"),
    ("nvts",           "NVT Signal",      "nvts"),
    ("puell-multiple", "Puell Multiple",  "puellMultiple"),
]


# ── Core helpers ───────────────────────────────────────────────────────────────

def fetch_indicator(slug: str) -> list[dict]:
    """
    GET /v1/{slug} — returns the JSON list of daily records.
    Sends the API key as a Bearer header when available; works without
    it on the free tier. Retries on 429 with exponential backoff.
    """
    url = f"{BASE_URL}/{slug}"
    log.info("Fetching %-22s  →  %s", slug, url)

    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

    for attempt in range(RETRY_MAX + 1):
        resp = requests.get(url, headers=headers, timeout=30)

        if resp.status_code == 429:
            if attempt < RETRY_MAX:
                wait = RETRY_BASE * (2 ** attempt)
                log.warning(
                    "429 rate-limited on '%s' — waiting %ds before retry %d/%d",
                    slug, wait, attempt + 1, RETRY_MAX,
                )
                time.sleep(wait)
                continue
            else:
                resp.raise_for_status()

        resp.raise_for_status()
        break

    data = resp.json()
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    raise ValueError(f"Unexpected response shape for {slug}: {type(data)}")


def to_dataframe(records: list[dict], name: str) -> pd.DataFrame:
    """
    Convert API records to a date-indexed DataFrame.
    bitcoin-data.com uses 'd' (ISO string) + 't' (unix seconds).
    """
    df = pd.DataFrame(records)

    DATE_COLS_ISO  = ("d", "date", "dt")
    DATE_COLS_UNIX = ("t", "unixts", "timestamp", "time")

    col_lower = {c.lower(): c for c in df.columns}
    iso_col   = next((col_lower[k] for k in DATE_COLS_ISO  if k in col_lower), None)
    unix_col  = next((col_lower[k] for k in DATE_COLS_UNIX if k in col_lower), None)

    if iso_col is not None:
        date_col = iso_col
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    elif unix_col is not None:
        date_col = unix_col
        df["date"] = pd.to_datetime(df[date_col], unit="s", errors="coerce")
        if df["date"].isna().all():
            df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        raise ValueError(
            f"No date column found in {name} response. Columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["date"]).sort_values("date").set_index("date")

    cols_to_drop = {date_col}
    if iso_col:  cols_to_drop.add(iso_col)
    if unix_col: cols_to_drop.add(unix_col)
    df.drop(columns=list(cols_to_drop), errors="ignore", inplace=True)
    return df


def save_csv(df: pd.DataFrame, name: str) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    slug = name.lower().replace(" ", "_")
    path = DATA_DIR / f"{slug}.csv"
    df.to_csv(path)
    log.info("Saved %d rows  →  %s", len(df), path)
    return path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log.info(
        "Starting BTC on-chain download  (%d indicators, ~%ds between calls)",
        len(INDICATORS), PAUSE_SEC,
    )
    summary_rows = []

    for i, (slug, name, _hint) in enumerate(INDICATORS):
        try:
            records = fetch_indicator(slug)
            df      = to_dataframe(records, name)
            save_csv(df, name)

            latest      = df.iloc[-1]
            latest_date = df.index[-1].date()
            vals = "  ".join(
                f"{col}={latest[col]:.4f}"
                for col in df.columns
                if pd.api.types.is_numeric_dtype(df[col])
            )
            summary_rows.append({"Indicator": name, "Latest date": latest_date,
                                  "Values": vals})

        except requests.HTTPError as exc:
            log.error("%s  HTTP %s", name, exc.response.status_code)
            summary_rows.append({"Indicator": name, "Latest date": "ERROR",
                                  "Values": str(exc)})
        except Exception as exc:
            log.error("%s  %s", name, exc)
            summary_rows.append({"Indicator": name, "Latest date": "ERROR",
                                  "Values": str(exc)})

        if i < len(INDICATORS) - 1:
            log.info("Waiting %ds before next request ...", PAUSE_SEC)
            time.sleep(PAUSE_SEC)

    print("\n" + "=" * 70)
    print(f"  BTC On-Chain Snapshot   (downloaded {datetime.now():%Y-%m-%d %H:%M})")
    print("=" * 70)
    for row in summary_rows:
        print(f"\n  {row['Indicator']}  [{row['Latest date']}]")
        print(f"    {row['Values']}")
    print("\n" + "=" * 70)
    print(f"  CSV files saved to: {DATA_DIR.resolve()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
