import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Define the symbol for Abans PLC
symbol = 'ABANS.LK'  # Use the correct ticker for Abans PLC in the relevant exchange

# Get today's date and the first date of the current year
end_date = datetime.now()
start_date = datetime(end_date.year, 1, 1)

# Fetch historical market data
data = yf.download(symbol, start=start_date, end=end_date)

# Plotting the closing prices
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label=symbol, color='green')

# Adding titles and labels
plt.title('YTD Closing Prices of Abans PLC')
plt.xlabel('Date')
plt.ylabel('Closing Price (LKR)')
plt.legend(loc='upper left')
plt.grid()

# Show the plot
plt.show()