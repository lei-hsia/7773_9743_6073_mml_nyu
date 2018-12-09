import pandas_datareader as pdr

start_date = "2013-01-01"
end_date = "2017-12-31"
symbols = ["AMZN","AAPL","GOOG","FB","NFLX","V",
"NVDA","AMD",
"MSFT","IBM","INTC","TSM","CSCO",
"BA","LMT","NOC","RTN","UTX",
"GS","JPM","C","BAC","AXP","DIS",
"JNJ","MCD","UNH","WMT","WBA",
"CVX","XOM","BP",
"KO","VZ","T"]

for symbol in symbols:
    print('Fetching quote history for %r' % symbol)
    stock = pdr.get_data_yahoo(symbol,start=start_date,end=end_date)
    stock.to_csv(symbol+".csv")
