# MIT License
# No express warranty
import requests
import pandas as pd

TOKEN = '90145b70881158a4c3d7f6f328b624381186f08f'


def download_cot_reports(start_year=2015, end_year=2025):
    """Download COT reports for Bitcoin from start_year to end_year inclusive."""
    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }

    for year in range(start_year, end_year + 1):
        api_url = f"https://api.cryptodatadownload.com/v1/data/cftc/cot/?contract_name=BITCOIN&year={year}&return=JSON"
        response = requests.get(api_url, headers=headers)

        if response.status_code != 200:
            print(f"Failed to fetch data for {year}: HTTP {response.status_code}")
            continue

        try:
            data = response.json()
            df = pd.json_normalize(data, 'result')
            df.to_csv(f"BTC_COT_Report_{year}.csv", index=False)
            print(f"Saved: BTC_COT_Report_{year}.csv")
        except Exception as e:
            print(f"Error processing data for {year}: {e}")
            
            
def download_cot_reports2(start_year=2015, end_year=2025):
    """Download COT reports for Ethereum from start_year to end_year inclusive."""
    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }

    for year in range(start_year, end_year + 1):
        api_url = f"https://api.cryptodatadownload.com/v1/data/cftc/cot/?contract_name=ETHER%20CASH%20SETTLED&year={year}&return=JSON"
        response = requests.get(api_url, headers=headers)

        if response.status_code != 200:
            print(f"Failed to fetch data for {year}: HTTP {response.status_code}")
            continue

        try:
            data = response.json()
            df = pd.json_normalize(data, 'result')
            df.to_csv(f"ETH_COT_Report_{year}.csv", index=False)
            print(f"Saved: ETH_COT_Report_{year}.csv")
        except Exception as e:
            print(f"Error processing data for {year}: {e}")            


if __name__ == "__main__":
    download_cot_reports()
    download_cot_reports2()