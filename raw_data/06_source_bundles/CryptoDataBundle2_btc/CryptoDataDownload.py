# -*- coding: utf-8 -*-
"""
CryptoDataDownload API TOKEN is obtainable here (costs around $50/mo): https://www.cryptodatadownload.com/
The api is here: https://www.cryptodatadownload.com/api/
"""


# MIT License
# No express warranty
import requests
import pandas as pd
TOKEN = '90145b70881158a4c3d7f6f328b624381186f08f'


def example_1():
    """This example will pull BTC blockchain block data and load into a Pandas Dataframe """
    api_url = f"https://api.cryptodatadownload.com/v1/data/summary/blockchain/blocks/?symbol=btc&limit=100000&return=JSON&auth_token={TOKEN}"
    response = requests.get(api_url)
    data = response.json()

    # Load the data into a pandas DataFrame
    df = pd.json_normalize(data, 'result')
    print(df.head())  # print sample rows
    df.to_csv("BTC_BlockData.csv")


def example_2():
    """This example will pull BTC blockchain block data and load into a Pandas Dataframe using Auth Header"""
    api_url = f"https://api.cryptodatadownload.com/v1/data/summary/blockchain/blocks/?symbol=btc&limit=100000&return=JSON"

    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }
    response = requests.get(api_url, headers=headers)  # include token in header
    data = response.json()

    # Load the data into a pandas DataFrame
    df = pd.json_normalize(data, 'result')
    print(df.head())  # print sample rows
    df.to_csv("BTC_BlockData.csv")
    

def example_3():
    """This example will pull data and load into a Pandas Dataframe using Auth Header"""
    """" The BTC Funding rates -- there are actually 2 endpoints to retrieve this from. The Binance is more suited for strategy development or analysis whereas DeriBit is more "live". """
    """"DeriBit endpoint - This interval is very granular and funding rates are updated hourly. """
    api_url = f"https://api.cryptodatadownload.com/v1/data/summary/binance/futures/funding/?symbol=BTCUSDT&limit=100000&return=JSON"

    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }
    response = requests.get(api_url, headers=headers)  # include token in header
    data = response.json()

    # Load the data into a pandas DataFrame
    df = pd.json_normalize(data, 'result')
    print(df.head())  # print sample rows
    df.to_csv("BTC_Deribit_FundingRates.csv")
    
def example_3b():
    """This example will pull data and load into a Pandas Dataframe using Auth Header"""
    """" The BTC Funding rates -- there are actually 2 endpoints to retrieve this from. The Binance is more suited for strategy development or analysis whereas DeriBit is more "live". """
    """"DeriBit endpoint - This interval is very granular and funding rates are updated hourly. """
    api_url = f"https://api.cryptodatadownload.com/v1/data/summary/binance/futures/funding/?symbol=ETHUSDT&limit=100000&return=JSON"

    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }
    response = requests.get(api_url, headers=headers)  # include token in header
    data = response.json()

    # Load the data into a pandas DataFrame
    df = pd.json_normalize(data, 'result')
    print(df.head())  # print sample rows
    df.to_csv("ETH_Deribit_FundingRates.csv")
    

def example_4():
    """This example will pull data and load into a Pandas Dataframe using Auth Header"""
    """"Binance endpoint - This funding rate is for 8hour intervals, and is on a slight lag. Meaning, we have it through end of June 2025."""
    api_url = f"https://api.cryptodatadownload.com/v1/data/summary/binance/futures/funding/?symbol=BTCUSDT&limit=100000&return=JSON"

    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }
    response = requests.get(api_url, headers=headers)  # include token in header
    data = response.json()

    # Load the data into a pandas DataFrame
    df = pd.json_normalize(data, 'result')
    print(df.head())  # print sample rows  
    df.to_csv("BTC_Binance_FundingRates.csv")
    
    
def example_4b():
    """This example will pull data and load into a Pandas Dataframe using Auth Header"""
    """"Binance endpoint - This funding rate is for 8hour intervals, and is on a slight lag. Meaning, we have it through end of June 2025."""
    api_url = f"https://api.cryptodatadownload.com/v1/data/summary/binance/futures/funding/?symbol=ETHUSDT&limit=100000&return=JSON"

    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }
    response = requests.get(api_url, headers=headers)  # include token in header
    data = response.json()

    # Load the data into a pandas DataFrame
    df = pd.json_normalize(data, 'result')
    print(df.head())  # print sample rows  
    df.to_csv("ETH_Binance_FundingRates.csv")    
    
    
def example_5():
    """This example will pull data and load into a Pandas Dataframe using Auth Header"""
    """" Binance Futures Metrics (Open Interest) """
    api_url = f"https://api.cryptodatadownload.com/v1/data/summary/binance/futures/metrics/?symbol=BTCUSDT&limit=100000&return=JSON"

    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }
    response = requests.get(api_url, headers=headers)  # include token in header
    data = response.json()

    # Load the data into a pandas DataFrame
    df = pd.json_normalize(data, 'result')
    print(df.head())  # print sample rows 
    df.to_csv("BTC_Binance_FuturesMetrics.csv")
    

def example_5b():
    """This example will pull data and load into a Pandas Dataframe using Auth Header"""
    """" Binance Futures Metrics (Open Interest) """
    api_url = f"https://api.cryptodatadownload.com/v1/data/summary/binance/futures/metrics/?symbol=ETHUSDT&limit=100000&return=JSON"

    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }
    response = requests.get(api_url, headers=headers)  # include token in header
    data = response.json()

    # Load the data into a pandas DataFrame
    df = pd.json_normalize(data, 'result')
    print(df.head())  # print sample rows 
    df.to_csv("ETH_Binance_FuturesMetrics.csv")   
    
    
def example_6():
    """This example will pull data and load into a Pandas Dataframe using Auth Header"""
    """" Committment of Traders report  BTC"""
    api_url = f"https://api.cryptodatadownload.com/v1/data/cftc/cot/?contract_name=BITCOIN&year=2025&&return=JSON"

    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }
    response = requests.get(api_url, headers=headers)  # include token in header
    data = response.json()

    # Load the data into a pandas DataFrame
    df = pd.json_normalize(data, 'result')
    print(df.head())  # print sample rows  
    df.to_csv("BTC_COT_Report_2025.csv")
    

def example_7():
    """This example will pull data and load into a Pandas Dataframe using Auth Header"""
    """" Committment of Traders report  ETH"""
    api_url = f"https://api.cryptodatadownload.com/v1/data/cftc/cot/?contract_name=ETHER%20CASH%20SETTLED&year=2025&return=JSON"

    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }
    response = requests.get(api_url, headers=headers)  # include token in header
    data = response.json()

    # Load the data into a pandas DataFrame
    df = pd.json_normalize(data, 'result')
    print(df.head())  # print sample rows  
    df.to_csv("ETH_COT_Report_2025.csv")
    
    
def example_8():
    """This example will pull data and load into a Pandas Dataframe using Auth Header"""
    """" Bitcoin DVOL index """
    api_url = f"https://api.cryptodatadownload.com/v1/data/ohlc/deribit/volatility/?symbol=BTC&limit=100000&return=JSON"

    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }
    response = requests.get(api_url, headers=headers)  # include token in header
    data = response.json()

    # Load the data into a pandas DataFrame
    df = pd.json_normalize(data, 'result')
    print(df.head())  # print sample rows  
    df.to_csv("BTC_Deribit_DVOL.csv")   
    
 
def example_9():
    """This example will pull data and load into a Pandas Dataframe using Auth Header"""
    """" Ethereum DVOL index """
    api_url = f"https://api.cryptodatadownload.com/v1/data/ohlc/deribit/volatility/?symbol=ETH&limit=100000&return=JSON"

    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }
    response = requests.get(api_url, headers=headers)  # include token in header
    data = response.json()

    # Load the data into a pandas DataFrame
    df = pd.json_normalize(data, 'result')
    print(df.head())  # print sample rows  
    df.to_csv("ETH_Deribit_DVOL.csv")     
    
    
    
def example_10():
    """This example will pull data and load into a Pandas Dataframe using Auth Header"""
    """"MarketBreadth: 52-week highs and lows for each trading date"""
    api_url = f"https://api.cryptodatadownload.com/v1/data/breadth/52wk-highs-lows/?date=2025-07-24&limit=100000&return=JSON"

    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }
    response = requests.get(api_url, headers=headers)  # include token in header
    data = response.json()

    # Load the data into a pandas DataFrame
    df = pd.json_normalize(data, 'result')
    print(df.head())  # print sample rows  
    df.to_csv("MarketBreadth_52WeekHiLow.csv")    
    
    
def example_11():
    """This example will pull data and load into a Pandas Dataframe using Auth Header"""
    """"MarketBreadth: Moving Average Tracking"""
    api_url = f"https://api.cryptodatadownload.com/v1/data/breadth/moving-average-tracking/?date=2025-07-24&limit=100000&return=JSON"

    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }
    response = requests.get(api_url, headers=headers)  # include token in header
    data = response.json()

    # Load the data into a pandas DataFrame
    df = pd.json_normalize(data, 'result')
    print(df.head())  # print sample rows  
    df.to_csv("MarketBreadth_MATracking.csv") 


def example_12():
    """This example will pull data and load into a Pandas Dataframe using Auth Header"""
    """"Binance: all available"""
    api_url = f"https://api.cryptodatadownload.com/v1/data/ohlc/binance/all/available/?return=JSON"

    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }
    response = requests.get(api_url, headers=headers)  # include token in header
    data = response.json()

    # Load the data into a pandas DataFrame
    df = pd.json_normalize(data, 'result')
    print(df.head())  # print sample rows  
    df.to_csv("Binance_AllAvailable.csv")  

def example_13():
    """This example will pull data and load into a Pandas Dataframe using Auth Header"""
    """"Binance: option chain by date"""
    api_url = f"https://api.cryptodatadownload.com/v1/?underlying=BTCUSDT&symbol=BTC-240628-70000-C&type=C&maturity=251229&limit=100000&return=JSON"

    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }
    response = requests.get(api_url, headers=headers)  # include token in header
    data = response.json()

    # Load the data into a pandas DataFrame
    df = pd.json_normalize(data, 'result')
    print(df.head())  # print sample rows  
    df.to_csv("BTC_Binance_OptionChainByDate.csv")   


def example_14():
    """This example will pull data and load into a Pandas Dataframe using Auth Header"""
    """"Binance: basis trade difference (“basis” column = (spot_close − futures_close))"""
    api_url = f"https://api.cryptodatadownload.com/v1/data/summary/binance/basis/?symbol=BTCUSDT&timeframe=daily&return=JSON"

    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }
    response = requests.get(api_url, headers=headers)  # include token in header
    data = response.json()

    # Load the data into a pandas DataFrame
    df = pd.json_normalize(data, 'result')
    print(df.head())  # print sample rows  
    df.to_csv("BTC_Binance_BasisSpotFutDifference.csv") 
    
    
def example_15():
    """This example will pull data and load into a Pandas Dataframe using Auth Header"""
    """"Binance: basis trade difference (“basis” column = (spot_close − futures_close))"""
    api_url = f"https://api.cryptodatadownload.com/v1/data/summary/binance/basis/?symbol=ETHUSDT&timeframe=daily&return=JSON"

    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }
    response = requests.get(api_url, headers=headers)  # include token in header
    data = response.json()

    # Load the data into a pandas DataFrame
    df = pd.json_normalize(data, 'result')
    print(df.head())  # print sample rows  
    df.to_csv("ETH_Binance_BasisSpotFutDifference.csv")    



def example_16():
    """This example will pull data and load into a Pandas Dataframe using Auth Header"""
    """"Binance: Open Interest Totals by Call or Put"""
    api_url = f"https://api.cryptodatadownload.com/v1/data/summary/binance/options/oi/?underlying=BTCUSDT&limit=100000&return=JSON"

    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }
    response = requests.get(api_url, headers=headers)  # include token in header
    data = response.json()

    # Load the data into a pandas DataFrame
    df = pd.json_normalize(data, 'result')
    print(df.head())  # print sample rows  
    df.to_csv("BTC_Binance_OITotalsByCallOrPut.csv") 

def example_17():
    """This example will pull data and load into a Pandas Dataframe using Auth Header"""
    """"Binance: Open Interest Totals by Call or Put"""
    api_url = f"https://api.cryptodatadownload.com/v1/data/summary/binance/options/oi/?underlying=ETHUSDT&limit=100000&return=JSON"

    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }
    response = requests.get(api_url, headers=headers)  # include token in header
    data = response.json()

    # Load the data into a pandas DataFrame
    df = pd.json_normalize(data, 'result')
    print(df.head())  # print sample rows  
    df.to_csv("ETH_Binance_OITotalsByCallOrPut.csv")  


def example_18():
    """This example will pull data and load into a Pandas Dataframe using Auth Header"""
    """"Binance: Volume and statistical summary meta data using raw trade print activity from each day."""
    api_url = f"https://api.cryptodatadownload.com/v1/data/summary/binance/spot/transactional/?symbol=BTCUSDT&enddate=2025-07-24&limit=100000&return=JSON"

    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }
    response = requests.get(api_url, headers=headers)  # include token in header
    data = response.json()

    # Load the data into a pandas DataFrame
    df = pd.json_normalize(data, 'result')
    print(df.head())  # print sample rows  
    df.to_csv("BTC_Binance_VolumeStatistics.csv")   

def example_19():
    """This example will pull data and load into a Pandas Dataframe using Auth Header"""
    """"Binance: Volume and statistical summary meta data using raw trade print activity from each day."""
    api_url = f"https://api.cryptodatadownload.com/v1/data/summary/binance/spot/transactional/?symbol=ETHUSDT&enddate=2025-07-24&limit=100000&return=JSON"

    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }
    response = requests.get(api_url, headers=headers)  # include token in header
    data = response.json()

    # Load the data into a pandas DataFrame
    df = pd.json_normalize(data, 'result')
    print(df.head())  # print sample rows  
    df.to_csv("ETH_Binance_VolumeStatistics.csv") 
    
    
def example_20():
    """This example will pull data and load into a Pandas Dataframe using Auth Header"""
    """"Binance: Hourly data."""
    api_url = f"https://api.cryptodatadownload.com/v1/data/ohlc/binance/spot/?symbol=BTCUSDT&interval=1h&enddate=2025-07-24&limit=100000&return=JSON"

    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }
    response = requests.get(api_url, headers=headers)  # include token in header
    data = response.json()

    # Load the data into a pandas DataFrame
    df = pd.json_normalize(data, 'result')
    print(df.head())  # print sample rows  
    df.to_csv("BTC_Binance_OHLC_hourly.csv")     

def example_20b():
    """This example will pull data and load into a Pandas Dataframe using Auth Header"""
    """"Binance: Hourly data."""
    api_url = f"https://api.cryptodatadownload.com/v1/data/ohlc/binance/spot/?symbol=ETHUSDT&interval=1h&enddate=2025-07-24&limit=100000&return=JSON"

    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }
    response = requests.get(api_url, headers=headers)  # include token in header
    data = response.json()

    # Load the data into a pandas DataFrame
    df = pd.json_normalize(data, 'result')
    print(df.head())  # print sample rows  
    df.to_csv("ETH_Binance_OHLC_hourly.csv")   
    
def example_21():
    """This example will pull data and load into a Pandas Dataframe using Auth Header"""
    """"Binance: Daily data."""
    api_url = f"https://api.cryptodatadownload.com/v1/data/ohlc/binance/spot/?symbol=BTCUSDT&interval=1d&enddate=2025-07-24&limit=100000&return=JSON"

    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }
    response = requests.get(api_url, headers=headers)  # include token in header
    data = response.json()

    # Load the data into a pandas DataFrame
    df = pd.json_normalize(data, 'result')
    print(df.head())  # print sample rows  
    df.to_csv("BTC_Binance_OHLC_daily.csv")     

def example_21b():
    """This example will pull data and load into a Pandas Dataframe using Auth Header"""
    """"Binance: Daily data."""
    api_url = f"https://api.cryptodatadownload.com/v1/data/ohlc/binance/spot/?symbol=ETHUSDT&interval=1d&enddate=2025-07-24&limit=100000&return=JSON"

    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }
    response = requests.get(api_url, headers=headers)  # include token in header
    data = response.json()

    # Load the data into a pandas DataFrame
    df = pd.json_normalize(data, 'result')
    print(df.head())  # print sample rows  
    df.to_csv("ETH_Binance_OHLC_daily.csv")   


def example_22():
    """This example will pull data and load into a Pandas Dataframe using Auth Header"""
    """"Binance: Par Yields Real."""
    api_url = f"https://api.cryptodatadownload.com/v1/data/treasury/par-yields-real/?year=2025&limit=100000&return=JSON"

    headers = {
        'Authorization': f'TOKEN {TOKEN}'
    }
    response = requests.get(api_url, headers=headers)  # include token in header
    data = response.json()

    # Load the data into a pandas DataFrame
    df = pd.json_normalize(data, 'result')
    print(df.head())  # print sample rows  
    df.to_csv("Par_Yields_Real.csv")      
       


if __name__ == "__main__":
    #example_1()
    #example_2()
    #example_3()
    #example_3b()
    #example_4()
    #example_4b()
    #example_5()
    #example_5b()
    #example_6()
    #example_7()
    #example_8() 
    #example_9()
    #example_10()
    #example_11() #not working
    #example_12() #not downloading but working
    #example_13() #not working
    #example_14()
    #example_15()
    #example_16()
    #example_17()
    #example_18()
    #example_19()
    #example_20()
    #example_20b()
    #example_21()
    #example_21b()
    example_22()