import pandas as pd
import datetime
import time

import yfinance as yf

#NOTE: YOU NEED TO pip install yfinance=0.2.61

start_date=datetime.datetime(2024, 6,30)
end_date= datetime.datetime.now()
stock_list = ["BTC-USD"]

for stock in stock_list:
    
    stock_data = yf.download(stock, start=start_date, end=end_date, interval='1wk')
    sd = pd.DataFrame(stock_data.values, index=stock_data.index, columns=["Close","High","Low","Open","Volume"])
    
    if stock_data.empty:
        print(f"Warning: No data for {stock}")
        continue
    
    sd.to_csv(f"{stock}.csv")
    print(f"Saved {stock}.csv")


