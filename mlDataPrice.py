from tvDatafeed import TvDatafeed, Interval

username = 'shahram4m@gmail.com'
password = 'myskymoon'

tv = TvDatafeed(username, password)


# index
nifty_index_data = tv.get_hist(symbol='BTCUSD',exchange='BITSTAMP',interval=Interval.in_daily,n_bars=1000000)

print(type(nifty_index_data))

nifty_index_data.to_csv('my_file_price.csv')