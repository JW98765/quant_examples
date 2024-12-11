from pathlib import Path
from time import perf_counter
import datetime
import heapq
from tqdm.auto import tqdm
from tqdm.contrib import tenumerate

import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from pykalman import KalmanFilter


"""
(VERY IMPORTANT, we will attempt to address this later) One of the biggest limitations of the model is that it requires the regression parameters (alpha and beta) which create the stationary spread to be similar between the modeled session (morning/afternoon) and the session we are performing inference on, otherwise we may have offsets and linear trends in the spread signal which make it unreliable. As we can see from the backtest plot, this is not the case and alpha and beta vary significantly between sessions. Is there a way we can reliably estimate the current alpha and beta from past data to ensure the spread is stationary?
"""

"""
Plotting
"""
def plot_data_general(data, day_num=0, n_samples_day=1802):
    # Select the data for the given day
    data_day = data.iloc[n_samples_day*day_num:n_samples_day*day_num+n_samples_day]

    fig, ax = plt.subplots(5, 1, figsize=(14, 15))

    # Plot for X
    ax[0].plot(data_day['Seconds'], data_day['X_BID'], label='X Bid Price', color='blue', linestyle='--')
    ax[0].plot(data_day['Seconds'], data_day['X_ASK'], label='X Ask Price', color='red', linestyle='--')
    ax[0].plot(data_day['Seconds'], data_day['X_MID'], label='X Mid Price', color='green')
    ax[0].set_title('Prices of X')
    ax[0].set_xlabel('Time (seconds)')
    ax[0].set_ylabel('Price (AUD)')
    ax[0].legend()

    # Plot for Y
    ax[1].plot(data_day['Seconds'], data_day['Y_BID'], label='Y Bid Price', color='blue', linestyle='--')
    ax[1].plot(data_day['Seconds'], data_day['Y_ASK'], label='Y Ask Price', color='red', linestyle='--')
    ax[1].plot(data_day['Seconds'], data_day['Y_MID'], label='Y Mid Price', color='green')
    ax[1].set_title('Prices of Y')
    ax[1].set_xlabel('Time (seconds)')
    ax[1].set_ylabel('Price (AUD)')
    ax[1].legend()

    # Plot for X and Y comparison
    ax[2].plot(data_day['Seconds'], data_day['X_MID'], label='X Mid Price', color='blue', linestyle='--')
    ax[2].plot(data_day['Seconds'], data_day['Y_MID'], label='Y Mid Price', color='red', linestyle='--')
    ax[2].set_title('Prices of X and Y compared')
    ax[2].set_xlabel('Time (seconds)')
    ax[2].set_ylabel('Price (AUD)')
    ax[2].legend()

    # Plot volumes
    ax[3].plot(data_day['Seconds'], data_day['X_BID_VOL'], label='X Bid Volume', color='blue', alpha=0.5)
    ax[3].plot(data_day['Seconds'], data_day['X_ASK_VOL'], label='X Ask Volume', color='red', alpha=0.5)
    ax[3].plot(data_day['Seconds'], data_day['Y_BID_VOL'], label='Y Bid Volume', color='cyan', alpha=0.5)
    ax[3].plot(data_day['Seconds'], data_day['Y_ASK_VOL'], label='Y Ask Volume', color='magenta', alpha=0.5)
    ax[3].set_title('Bid and Ask Volumes of X and Y')
    ax[3].set_xlabel('Time (seconds)')
    ax[3].set_ylabel('Volume (Units)')
    ax[3].legend()

    # Plotting where there is data (market open) and what is a time skip
    ax[4].plot(data_day['Seconds'], data_day['Open'], label='Market open', color='purple')
    ax[4].set_title('Market open and close periods')
    ax[4].set_xlabel('Time (seconds)')
    ax[4].set_ylabel('Market open (1) or close (0)')
    ax[4].legend()

    plt.tight_layout()
    plt.show()


def plot_residuals(data, model):
    # Compute residuals (spread) using linear regression coefficients
    alpha, beta = model.params
    y_pred = beta*data['LOG_X_MID'] + alpha
    residuals = data['LOG_Y_MID'] - y_pred
    # residuals = model.resid
    # Compute the FFT of the residuals
    fft_resid = np.fft.fft(residuals)
    fft_freq = np.fft.fftfreq(len(fft_resid), d=10)
    # Create a figure
    fig, ax = plt.subplots(4, 1, figsize=(14, 15))
    # Create a plot of the residuals over time
    ax[0].plot(residuals, label='Residuals over time')
    ax[0].set_title('Residuals Analysis')
    ax[0].set_xlabel('Time (10s)')
    ax[0].set_ylabel('Residual (log(price)')
    # Create an autocorrelation plot of the residuals
    pd.plotting.autocorrelation_plot(residuals, ax=ax[1], label='Autocorrelation of residuals')
    ax[1].set_title('Autocorrelation of residuals')
    # Create a plot of the FFT of the residuals
    ax[2].plot(fft_freq[:50], np.abs(fft_resid)[:50], label='FFT of residuals')
    ax[2].set_title('FFT of residuals')
    ax[2].set_xlabel('Frequency (Hz)')
    ax[2].set_ylabel('Amplitude')
    # Create a scatter plot of the log prices
    cm = plt.get_cmap('jet')
    sc = ax[3].scatter(data['LOG_X_MID'], data['LOG_Y_MID'], s=50, c=data['LOG_X_MID'], cmap=cm, marker='o', alpha=0.6, label='Log Price', edgecolor='k')
    ax[3].plot(data['LOG_X_MID'], y_pred, '-', c='black', linewidth=3, label='OLS Fit')
    ax[3].set_title('Scatter plot of log prices')
    ax[3].set_xlabel('LOG_X_MID')
    ax[3].set_ylabel('LOG_Y_MID')
    ax[3].legend()
    cb = plt.colorbar(sc, ax=ax[3])
    # Generate 10 evenly spaced indices, set the locations of yticks
    indices = np.linspace(0, len(data.index) - 1, 10, dtype=int)
    cb.ax.set_yticklabels([str(data.index[i]) for i in indices])
    # Show the plot
    plt.show()


def plot_backtest(capital_gains, unrealized_gains, cash_balances, portfolio_values, coints, alphas, betas):
    fig, ax = plt.subplots(4, 1, figsize=(12, 15))
    # Plot capital gains, unrealized gains, portfolio values and cash balances over days
    ax[0].plot(capital_gains, label='Capital Gains', color='green', linestyle='--')
    ax[0].plot(unrealized_gains, label='Unrealized Gains', color='orange', linestyle='--')
    ax[0].plot(portfolio_values, label='Portfolio Value', color='blue')
    ax[0].set_title('Portfolio value and gains over time')
    ax[0].set_xlabel('Time (day/week)')
    ax[0].set_ylabel('Gains ($)')
    ax[0].legend()

    """
    ax[1].plot(unrealized_gains, label='Unrealized Gains')
    ax[1].set_title('Unrealized gains over time')
    ax[1].set_xlabel('Time (day/week)')
    ax[1].set_ylabel('Unrealized gains ($)')
    ax[1].legend()

    ax[2].plot(portfolio_values, label='Portfolio Value')
    ax[2].axhline(0, color='black', linestyle='--', label='Zero profit line')
    ax[2].set_title('Portfolio value over time')
    ax[2].set_xlabel('Time (day/week)')
    ax[2].set_ylabel('Gains and portfolio value ($)')
    ax[2].legend()        
    """

    # Plot cash balances (investment required to maintain positions)
    ax[1].plot(cash_balances, label='Cash balance')
    ax[1].set_title('Change in cash balance over time')
    ax[1].set_xlabel('Time (day/week)')
    ax[1].set_ylabel('Cash balance ($)')
    ax[1].legend()

    # Plot cointegrations and changes in alphas and betas over time (the hedging ratio)
    ax[2].plot(coints, label='p-value')
    ax[2].set_title('Cointegration p-values in a day')
    ax[2].set_xlabel('Time (day/week)')
    ax[2].set_ylabel('p-value')
    ax[2].legend()

    ax[3].plot(alphas, label='Alpha')
    ax[3].plot(betas, label='Beta')
    ax[3].set_title('Coefficients from regression')
    ax[3].set_xlabel('Time (day/week)')
    ax[3].set_ylabel('Coefficients')
    ax[3].legend()

    plt.show()


def plot_coint_vs_portfolio(coints, values):
    # Difference the cumulative sum of the portfolio values to get the daily change
    values_diff = np.diff(values)
    values_diff = np.insert(values_diff, -1, values[-1] - values[-2])
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(coints, values_diff)
    ax.set_xlabel("Cointegration p-value")
    ax.set_ylabel("Change in portfolio value")
    ax.set_title("Cointegration p-value vs. Change in portfolio value")
    plt.show()


""" 
Data processing
"""
def concat_days(data_grouped, start_day, n_days):
    # Concatenate first n_days of data (no adjustments)
    data_concat = pd.DataFrame()
    for i, date in enumerate(data_grouped.groups):
        if start_day <= i < start_day+n_days:
            data_day = data_grouped.get_group(date)
            data_concat = pd.concat([data_concat, data_day])
    data_concat.reset_index(drop=True, inplace=True)
    return data_concat


def check_volumes(group):
    return not (group['X_ASK_VOL'].nunique() == 1 and group['X_BID_VOL'].nunique() == 1 and group['Y_ASK_VOL'].nunique() == 1 and group['Y_BID_VOL'].nunique() == 1)


"""
Modeling
"""
def linear_regression(X, Y, print_stats=True):
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    if print_stats:
        print(model.summary())
    return model


def coint_test(data, print_stats=True):
    # Perform linear regression and get residuals
    lr = linear_regression(data['LOG_X_MID'], data['LOG_Y_MID'], print_stats=print_stats)
    residuals = lr.resid
    # Test for stationarity of residuals using ADF test
    adf_test = adfuller(residuals)
    # Check for cointegration using the Engle-Granger two-step method
    score, pvalue, _ = coint(data['LOG_X_MID'], data['LOG_Y_MID'])
    # Print if specified
    if print_stats:
        print(f"ADF Test p-value: {adf_test[1]}")
        print(f"Cointegration Test p-value: {pvalue}")
    # Return model and test results
    return lr, adf_test, pvalue


def calculate_thresholds_coint(data, model, std_thresh_enter=1.96, std_thresh_exit=0.5, plot=True):
    # Compute residuals (spread) using linear regression coefficients
    alpha, beta = model.params
    y_pred = beta*data['LOG_X_MID'] + alpha
    residuals = data['LOG_Y_MID'] - y_pred
    # Z-standardize the residuals
    """ Important because when we perform inference on a different time period using the alpha and beta from the pre-fitted model, the residuals will have a different mean and std"""
    residuals = (residuals - residuals.mean()) / residuals.std()
    # Compute the thresholds for entering and exiting trades
    short_entry = std_thresh_enter
    long_entry = -std_thresh_enter
    short_exit = std_thresh_exit
    long_exit = -std_thresh_exit
    # Plot if specified
    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        ax[0].plot(residuals, label='Residuals')
        ax[0].axhline(short_entry, color='red', linestyle='--', label='Short Entry')
        ax[0].axhline(long_entry, color='green', linestyle='--', label='Long Entry')
        ax[0].axhline(0.0, color='black', linestyle='-', label='Mean')
        ax[0].axhline(short_exit, color='red', linestyle='-', label='Short Exit')
        ax[0].axhline(long_exit, color='green', linestyle='-', label='Long Exit')
        ax[0].set_title('Residuals and Trading Signals of Model')
        ax[0].set_xlabel('Time (10s)')
        ax[0].set_ylabel('Residual (log(price))')
        ax[0].legend()

        ax[1].plot(data['X_BID_VOL'], label='X Bid Volume', color='blue', alpha=0.5)
        ax[1].plot(data['X_ASK_VOL'], label='X Ask Volume', color='red', alpha=0.5)
        ax[1].plot(data['Y_BID_VOL'], label='Y Bid Volume', color='cyan', alpha=0.5)
        ax[1].plot(data['Y_ASK_VOL'], label='Y Ask Volume', color='magenta', alpha=0.5)
        ax[1].set_title('Bid and Ask Volumes of X and Y')
        ax[1].set_xlabel('Time (10s)')
        ax[1].set_ylabel('Volume (Units)')
        ax[1].legend()

        plt.show()
    # Return parameters required for trade decisions
    return short_entry, long_entry, short_exit, long_exit


class Stock:
    def __init__(self):
        self.n_longs = 0
        self.n_shorts = 0
        self.longs = []  # Min-heap for long positions
        self.shorts = []  # Max-heap for short positions (using negative prices)
        self.long_positions = {}  # Dictionary to keep track of quantities at each price
        self.short_positions = {}  # Dictionary to keep track of quantities at each price

    def buy_long(self, price, quantity):
        """ Buy long position at a given price and quantity, returns the cash investment and number of successful trades """
        # Cleaner indexing with ints
        quantity = int(quantity)
        price = int(round(price))

        cash_diff = - price * quantity
        n_successful_trades = quantity
        self.n_longs += quantity
        if price in self.long_positions:
            self.long_positions[price] += quantity
        else:
            self.long_positions[price] = quantity
            # heapq.heappush(self.longs, price)

        return cash_diff, n_successful_trades

    def sell_long(self, current_price, quantity):
        """ Sell long position at a given price and quantity, returns the cash acquired, number of trades made, and capital gains/losses """
        """ We close the long position with the lowest buy price currently in our portfolio, hence the heap """
        quantity = int(quantity)
        current_price = int(round(current_price))

        capital_gains = 0
        cash_diff = 0
        n_successful_trades = 0
        while quantity > 0 and self.long_positions:
            # lowest_price = self.longs[0]
            lowest_price = list(self.long_positions.keys())[0]
            if self.long_positions[lowest_price] > quantity:
                self.n_longs -= quantity
                self.long_positions[lowest_price] -= quantity
                capital_gains += quantity * (current_price - lowest_price)
                cash_diff += quantity * current_price
                n_successful_trades += quantity
                quantity = 0
            else:
                self.n_longs -= self.long_positions[lowest_price]
                quantity -= self.long_positions[lowest_price]
                capital_gains += self.long_positions[lowest_price] * (current_price - lowest_price)
                cash_diff += self.long_positions[lowest_price] * current_price
                n_successful_trades += self.long_positions[lowest_price]
                del self.long_positions[lowest_price]
                # heapq.heappop(self.longs)
        return cash_diff, n_successful_trades, capital_gains

    def short_sell(self, price, quantity):
        """ Short sell position at a given price and quantity, returns the cash acquired and number of successful trades """
        quantity = int(quantity)
        price = int(round(price))

        cash_diff = price * quantity
        n_successful_trade = quantity
        self.n_shorts += quantity
        if price in self.short_positions:
            self.short_positions[price] += quantity
        else:
            self.short_positions[price] = quantity
            # heapq.heappush(self.shorts, -price)
        return cash_diff, n_successful_trade

    def buy_to_cover(self, current_price, quantity):
        """ Buy to cover short position at a given price and quantity, returns the cash investment, number of trades made, and capital gains/losses """
        """ Close the short position with the highest sell price """
        quantity = int(quantity)
        current_price = int(round(current_price))

        capital_gains = 0
        cash_diff = 0
        n_successful_trades = 0
        while quantity > 0 and self.short_positions:
            # highest_price = -self.shorts[0]
            highest_price = list(self.short_positions.keys())[0]
            if self.short_positions[highest_price] > quantity:
                self.n_shorts -= quantity
                self.short_positions[highest_price] -= quantity
                capital_gains += quantity * (highest_price - current_price)
                cash_diff -= quantity * current_price
                n_successful_trades += quantity
                quantity = 0
            else:
                self.n_shorts -= self.short_positions[highest_price]
                quantity -= self.short_positions[highest_price]
                capital_gains += self.short_positions[highest_price] * (highest_price - current_price)
                cash_diff -= self.short_positions[highest_price] * current_price
                n_successful_trades += self.short_positions[highest_price]
                del self.short_positions[highest_price]
                # heapq.heappop(self.shorts)
        return cash_diff, n_successful_trades, capital_gains

    def position_value(self, curr_sell_price, curr_buy_price):
        long_value = sum(curr_sell_price * qty for qty in self.long_positions.values())
        short_value = sum(curr_buy_price * qty for qty in self.short_positions.values())
        # Value of the position is the long value - short value
        return long_value - short_value

    def unrealized_gains(self, curr_sell_price, curr_buy_price):
        long_gains_unrealized = sum(
            (curr_sell_price - buy_price) * qty for buy_price, qty in self.long_positions.items())
        short_gains_unrealized = sum(
            (sell_price - curr_buy_price) * qty for sell_price, qty in self.short_positions.items())
        return long_gains_unrealized + short_gains_unrealized

    def __str__(self):
        return f"Longs: {self.long_positions}, Shorts: {self.short_positions}"


""" This class covers the portfolio of cash, stocks, and the trading history, 
the total_portfolio_value method tracks capital gains/losses and unrealized gains/losses from holdings 
at a specific time step. We will call this for every time step in the simulation """
class Portfolio:
    def __init__(self, initial_cash=0.0):
        self.initial_cash = initial_cash  # Initial cash balance
        self.cash = initial_cash  # Track cash balance
        self.stocks = {"X": Stock(), "Y": Stock()}
        self.curr_time_step = 0
        self.trade_history = {
            "X": {"buys": [], "sales": []},
            "Y": {"buys": [], "sales": []}
        }

    def buy_long(self, stock, price, quantity, timestep):
        # Create new stock object if it doesn't exist in portfolio yet
        if stock not in self.stocks:
            self.stocks[stock] = Stock()
        cash_diff, n_trades = self.stocks[stock].buy_long(price, quantity)
        self.cash += cash_diff
        # Update trade history
        self.trade_history[stock]["buys"].append((timestep, price, n_trades))

    def sell_long(self, stock, current_price, quantity, timestep):
        if stock not in self.stocks:
            self.stocks[stock] = Stock()
        cash_diff, n_trades, capital_gains = self.stocks[stock].sell_long(current_price, quantity)
        self.cash += cash_diff
        self.trade_history[stock]["sales"].append((timestep, current_price, n_trades))
        return capital_gains

    def short_sell(self, stock, price, quantity, timestep):
        if stock not in self.stocks:
            self.stocks[stock] = Stock()
        cash_diff, n_trades = self.stocks[stock].short_sell(price, quantity)
        self.cash += cash_diff
        self.trade_history[stock]["sales"].append((timestep, price, n_trades))

    def buy_to_cover(self, stock, current_price, quantity, timestep):
        if stock not in self.stocks:
            self.stocks[stock] = Stock()
        cash_diff, n_trades, capital_gains = self.stocks[stock].buy_to_cover(current_price, quantity)
        self.cash += cash_diff
        self.trade_history[stock]["buys"].append((timestep, current_price, n_trades))
        return capital_gains

    def total_portfolio_value(self, curr_prices):
        total_value = self.cash
        for stock in self.stocks:
            total_value += self.stocks[stock].position_value(curr_prices[stock]['sell'], curr_prices[stock]['buy'])
        return total_value

    def stock_quantities(self):
        return {stock: {'long': self.stocks[stock].n_longs,
                        'short': self.stocks[stock].n_shorts} for stock in self.stocks}


# Function to generate the trading signal of a single time step
def generate_signal(residual, short_entry, long_entry, short_exit, long_exit):
    # Generate the relevant entry and exit signals at each data point
    # Default signal: hold
    entry_signal = 'hold'
    exit_signal = 'hold'
    # Entry points
    if residual > short_entry:
        entry_signal = 'short'
    elif residual < long_entry:
        entry_signal = 'long'
    # Exit points
    if residual < short_exit:
        exit_signal = 'short'
    elif residual > long_exit:
        exit_signal = 'long'
    return entry_signal, exit_signal


def calc_max_volumes(x_vol, y_vol, beta):
    if x_vol >= int(beta * y_vol):
        x_vol = int(beta * y_vol)
    else:
        y_vol = int(x_vol / beta)
    return x_vol, y_vol


# Execute the strategy on one segment of data, marking entry and exit positions and volumes, calculating profit, and plotting
def execute_coint_trades(data, model, short_entry, long_entry, short_exit, long_exit, portfolio, warm_up_period=360, trade_freq=1,
                         plot=True):
    """
    This should simulate the data points coming in one at a time as we receive them in market.
    Thus, the z-score should be calculated on a cumulative basis to generate the trading signals
    """
    # Get model coefficients
    alpha, beta = model.params

    # Compute residuals (spread) using linear regression coefficients
    y_pred = beta * data['LOG_X_MID'] + alpha
    residuals = data['LOG_Y_MID'] - y_pred

    # True z-standardized residuals for comparison plot
    residuals_z = (residuals - residuals.mean()) / residuals.std()

    # Z-standardize the residuals with a cumulative calculation of mean and std
    cum_mean = residuals.expanding().mean()
    cum_mean_std = residuals.expanding().std()
    residuals = np.array((residuals - cum_mean) / cum_mean_std)

    # Initialize entry and exit signals tracking
    entry_signals = ['hold'] * len(data)
    exit_signals = ['hold'] * len(data)

    # Array for tracking cumulative gains, unrealized gains, and portfolio value at each step
    capital_gains = np.zeros(len(data))
    unrealized_gains = np.zeros(len(data))
    cash_balances = np.zeros(len(data))
    portfolio_values = np.zeros(len(data))

    # Track trade frequency
    steps_from_last_trade = trade_freq

    # Execute trades
    for i in range(warm_up_period, len(data)):
        # Get prices at current time step
        curr_prices = {
            'X': {'buy': data['X_ASK'].iloc[i], 'sell': data['X_BID'].iloc[i]},
            'Y': {'buy': data['Y_ASK'].iloc[i], 'sell': data['Y_BID'].iloc[i]}
        }

        # Only execute trade if a certain cooling period has passed
        if steps_from_last_trade >= trade_freq:
            # Generate trading signals
            entry_signal, exit_signal = generate_signal(residuals[i], short_entry, long_entry, short_exit, long_exit)

            # Update entry and exit signals
            entry_signals[i] = entry_signal
            exit_signals[i] = exit_signal

            # Calculate and update profits
            # Short the spread: short Y, long X
            if entry_signal == 'short':
                # Buy and sell the maximum volume in market
                y_vol_market = data['Y_BID_VOL'].iloc[i]
                x_vol_market = data['X_ASK_VOL'].iloc[i]
                x_vol, y_vol = calc_max_volumes(x_vol_market, y_vol_market, beta)
                # print(beta, y_vol, x_vol)

                # Buy stocks from market
                portfolio.buy_long('X', data['X_ASK'].iloc[i], x_vol, i)
                portfolio.short_sell('Y', data['Y_BID'].iloc[i], y_vol, i)

            # Long the spread: long Y, short X
            elif entry_signal == 'long':
                y_vol_market = data['Y_ASK_VOL'].iloc[i]
                x_vol_market = data['X_BID_VOL'].iloc[i]
                x_vol, y_vol = calc_max_volumes(x_vol_market, y_vol_market, beta)

                portfolio.buy_long('Y', data['Y_ASK'].iloc[i], y_vol, i)
                portfolio.short_sell('X', data['X_BID'].iloc[i], x_vol, i)

            # Close short position: buy Y, sell X
            if exit_signal == 'short':
                # Buy and sell the limiting volume between market and current inventory to preserve hedge ratio
                y_vol_market = data['Y_ASK_VOL'].iloc[i]
                x_vol_market = data['X_BID_VOL'].iloc[i]
                y_vol_max = min(portfolio.stocks['Y'].n_shorts, y_vol_market)
                x_vol_max = min(portfolio.stocks['X'].n_longs, x_vol_market)
                x_vol, y_vol = calc_max_volumes(x_vol_max, y_vol_max, beta)

                gains = 0.0
                gains += portfolio.buy_to_cover('Y', data['Y_ASK'].iloc[i], y_vol, i)
                gains += portfolio.sell_long('X', data['X_BID'].iloc[i], x_vol, i)
                capital_gains[i:] += gains

            # Close long position: sell Y, buy X
            elif exit_signal == 'long':
                y_vol_market = data['Y_BID_VOL'].iloc[i]
                x_vol_market = data['X_ASK_VOL'].iloc[i]
                y_vol_max = min(portfolio.stocks['Y'].n_longs, y_vol_market)
                x_vol_max = min(portfolio.stocks['X'].n_shorts, x_vol_market)
                x_vol, y_vol = calc_max_volumes(x_vol_max, y_vol_max, beta)

                gains = 0.0
                gains += portfolio.sell_long('Y', data['Y_BID'].iloc[i], y_vol, i)
                gains += portfolio.buy_to_cover('X', data['X_ASK'].iloc[i], x_vol, i)
                capital_gains[i:] += gains
            steps_from_last_trade = 1
        else:
            steps_from_last_trade += 1

        unrealized_gains[i] = portfolio.stocks['X'].unrealized_gains(data['X_BID'].iloc[i], data['X_ASK'].iloc[i]) + \
                              portfolio.stocks['Y'].unrealized_gains(data['Y_BID'].iloc[i], data['Y_ASK'].iloc[i])
        cash_balances[i] = portfolio.cash
        portfolio_values[i] = portfolio.total_portfolio_value(curr_prices)

    # print(portfolio.total_portfolio_value(curr_prices))

    long_entry_idx = np.where(np.array(entry_signals) == 'long')[0]
    short_entry_idx = np.where(np.array(entry_signals) == 'short')[0]
    long_exit_idx = np.where(np.array(exit_signals) == 'long')[0]
    short_exit_idx = np.where(np.array(exit_signals) == 'short')[0]

    # Plot residuals and trading signals if specified
    if plot:
        fig, ax = plt.subplots(4, 1, figsize=(10, 12))
        # Plot rolling z-score residuals and trading signals
        ax[0].plot(residuals, label='Residuals (rolling z-score)')
        ax[0].axhline(short_entry, color='red', linestyle='--', label='Short enter')
        ax[0].axhline(long_entry, color='green', linestyle='--', label='Long enter')
        ax[0].axhline(short_exit, color='red', linestyle='-', label='Short Exit')
        ax[0].axhline(long_exit, color='green', linestyle='-', label='Long Exit')
        ax[0].scatter(long_entry_idx, residuals[long_entry_idx], color='green', marker='v', s=50,
                      label='Long entry points')
        ax[0].scatter(short_entry_idx, residuals[short_entry_idx], color='red', marker='v', s=50,
                      label='Short entry points')
        ax[0].scatter(long_exit_idx, residuals[long_exit_idx], color='green', s=10, label='Long exit points')
        ax[0].scatter(short_exit_idx, residuals[short_exit_idx], color='red', s=10, label='Short exit points')
        ax[0].set_title('Residuals and Trading Signals of Backtest')
        ax[0].set_xlabel('Time (10s)')
        ax[0].set_ylabel('Residual (log(price))')
        ax[0].legend()

        # Plot true residuals for comparison
        ax[1].plot(residuals_z, label='Residuals (z-standardized)')
        ax[1].axhline(short_entry, color='red', linestyle='--', label='Short enter')
        ax[1].axhline(long_entry, color='green', linestyle='--', label='Long enter')
        ax[1].axhline(short_exit, color='red', linestyle='-', label='Short Exit')
        ax[1].axhline(long_exit, color='green', linestyle='-', label='Long Exit')
        ax[1].set_title('Residuals and signals if we know mean and std beforehand')
        ax[1].set_xlabel('Time (10s)')
        ax[1].set_ylabel('Residual (log(price))')

        # Plot capital gains, unrealized gains, and portfolio values
        ax[2].plot(capital_gains, label='Capital Gains', color='green', linestyle='--')
        ax[2].plot(unrealized_gains, label='Unrealized Gains', color='orange', linestyle='--')
        ax[2].plot(portfolio_values, label='Portfolio Value', color='blue')
        ax[2].set_title('Capital gain, unrealized gains, and portfolio value over time')
        ax[2].set_xlabel('Time (10s)')
        ax[2].set_ylabel('Gains and portfolio value ($)')
        ax[2].legend()

        # Plot cash balances (investment required to maintain positions)
        ax[3].plot(cash_balances, label='Cash balance')
        ax[3].set_title('Change in cash balance over time')
        ax[3].set_xlabel('Time (10s)')
        ax[3].set_ylabel('Cash balance ($)')
        ax[3].legend()

        plt.show()

    summary = {
        'long_entry_idx': long_entry_idx,
        'short_entry_idx': short_entry_idx,
        'long_exit_idx': long_exit_idx,
        'short_exit_idx': short_exit_idx,
        'capital_gains': capital_gains,
        'unrealized_gains': unrealized_gains,
        'cash_balances': cash_balances,
        'portfolio_values': portfolio_values,
    }
    return summary


def backtest_coint_strat_days(filtered_data_grouped, p_thresh=0.05, std_thresh_enter=1.96, std_thresh_exit=0.0, trade_freq=1):
    # Initialize portfolio and trade tracking
    capital_gains = []
    unrealized_gains = []
    cash_balances = []
    portfolio_values = []
    coints = []
    alphas = []
    betas = []

    portfolio = Portfolio(initial_cash=0.0)

    # Initial model parameters
    model = None
    short_entry = long_entry = short_exit = long_exit = 0.0

    # Loop through each session, perform inference with the previous model if a model exists
    # If a model does not exist, we wait until we find a session with high enough cointegration to use as the model
    # The model is updated to the current session if it is found to be cointegrated after inference is performed using the previous model
    for i, date in enumerate(filtered_data_grouped.groups):
        # Get the data for the day
        data_day = filtered_data_grouped.get_group(date)
        # Split into morning and afternoon
        if len(data_day) == 721:
            data_morning = data_afternoon = data_day
            phases = ["morning"]
        elif len(data_day) == 1081:
            data_afternoon = data_morning = data_day
            phases = ["afternoon"]
        else:
            data_morning = data_day.iloc[:721]
            data_afternoon = data_day.iloc[721:]
            phases = ["morning", "afternoon"]
        # Process each session independently
        for phase in phases:
            if phase == "morning":
                curr_data = data_morning
            else:
                curr_data = data_afternoon

            # Execute trades if we have found a model up to this point
            if model is not None:
                sess_summary = execute_coint_trades(curr_data, model, short_entry, long_entry, short_exit, long_exit,
                                                    portfolio, warm_up_period=360, trade_freq=trade_freq, plot=False)
                prev_capital_gains = capital_gains[-1]
                capital_gains.append(sess_summary['capital_gains'][-1] + prev_capital_gains)
                unrealized_gains.append(sess_summary['unrealized_gains'][-1])
                cash_balances.append(sess_summary['cash_balances'][-1])
                portfolio_values.append(sess_summary['portfolio_values'][-1])
            else:
                capital_gains.append(0.0)
                unrealized_gains.append(0.0)
                cash_balances.append(0.0)
                portfolio_values.append(0.0)

            # Perform cointegration test to look for suitable model
            lr, adf, coint = coint_test(curr_data, print_stats=False)
            alpha, beta = lr.params

            coints.append(adf[1])
            alphas.append(alpha)
            betas.append(beta)
            # If the p-value is less than the threshold, use this as the new model
            # if coint < p_thresh:
            if adf[1] < p_thresh:
                model = lr
                short_entry, long_entry, short_exit, long_exit = calculate_thresholds_coint(curr_data, lr,
                                                                                            std_thresh_enter=std_thresh_enter,
                                                                                            std_thresh_exit=std_thresh_exit,
                                                                                            plot=False)

            # Print progress
            # print(f"Day: {i}, Phase: {phase}, Capital gains: {capital_gains[-1]}, Portfolio value: {portfolio_values[-1]}")

    return portfolio, capital_gains, unrealized_gains, cash_balances, portfolio_values, coints, alphas, betas


def backtest_coint_strat_weeks(filtered_data_grouped_weeks, p_thresh=0.05, std_thresh_enter=1.96, std_thresh_exit=0.0, trade_freq=1):
    # Initialize portfolio and trade tracking
    capital_gains = []
    unrealized_gains = []
    cash_balances = []
    portfolio_values = []
    coints = []
    alphas = []
    betas = []

    portfolio = Portfolio(initial_cash=0.0)

    # Initial model parameters
    model = None
    short_entry = long_entry = short_exit = long_exit = 0.0

    # Loop through each session, perform inference with the previous model if a model exists
    # If a model does not exist, we wait until we find a week with high enough cointegration
    for i, week in enumerate(filtered_data_grouped_weeks.groups):
        # Get the data for the week
        curr_data = filtered_data_grouped_weeks.get_group(week)

        # Execute trades if we have found a model up to this point
        if model is not None:
            sess_summary = execute_coint_trades(curr_data, model, short_entry, long_entry, short_exit, long_exit,
                                                portfolio, warm_up_period=360, trade_freq=trade_freq, plot=False)
            prev_capital_gains = capital_gains[-1]
            capital_gains.append(sess_summary['capital_gains'][-1] + prev_capital_gains)
            unrealized_gains.append(sess_summary['unrealized_gains'][-1])
            cash_balances.append(sess_summary['cash_balances'][-1])
            portfolio_values.append(sess_summary['portfolio_values'][-1])
        else:
            capital_gains.append(0.0)
            unrealized_gains.append(0.0)
            cash_balances.append(0.0)
            portfolio_values.append(0.0)

        # Perform cointegration test to look for suitable model
        lr, adf, coint = coint_test(curr_data, print_stats=False)
        alpha, beta = lr.params

        coints.append(adf[1])
        alphas.append(alpha)
        betas.append(beta)
        # If the p-value is less than the threshold, use this as the new model
        if adf[1] < p_thresh:
            # if coint < p_thresh:
            model = lr
            short_entry, long_entry, short_exit, long_exit = calculate_thresholds_coint(curr_data, lr,
                                                                                        std_thresh_enter=std_thresh_enter,
                                                                                        std_thresh_exit=std_thresh_exit,
                                                                                        plot=False)

        # print(f"Week: {i}, Capital gains: {capital_gains[-1]}, Portfolio value: {portfolio_values[-1]}")

    return portfolio, capital_gains, unrealized_gains, cash_balances, portfolio_values, coints, alphas, betas


"""
Analysis
"""
def calculate_roi(portfolio, portfolio_value):
    long_buys_total = sum(
        [sum([price * qty for price, qty in stock.long_positions.items()]) for stock in portfolio.stocks.values()])
    short_sells_total = sum(
        [sum([price * qty for price, qty in stock.short_positions.items()]) for stock in portfolio.stocks.values()])
    investment_required = 0.3 * short_sells_total + max(1,
                                                        long_buys_total - short_sells_total)  # use 1 to prevent numerical instability
    roi = portfolio_value / investment_required
    return roi, investment_required


def grid_search_coint(train_data_grouped, std_thresh_enter_vals, std_thresh_exit_vals, p_thresh=0.05, type='day',
                      optimize='raw', trade_freq=1):
    values_grid = np.zeros((len(std_thresh_enter_vals), len(std_thresh_exit_vals)))

    for i, std_thresh_enter in enumerate(std_thresh_enter_vals):
        for j, std_thresh_exit in enumerate(std_thresh_exit_vals):
            # Run backtest with parameters
            if type == 'day':
                portfolio, gains, unrealized, cash, values, coints, alphas, betas = backtest_coint_strat_days(
                    train_data_grouped, p_thresh=p_thresh, std_thresh_enter=std_thresh_enter,
                    std_thresh_exit=std_thresh_exit, trade_freq=trade_freq)
            elif type == 'week':
                portfolio, gains, unrealized, cash, values, coints, alphas, betas = backtest_coint_strat_weeks(
                    train_data_grouped, p_thresh=p_thresh, std_thresh_enter=std_thresh_enter,
                    std_thresh_exit=std_thresh_exit, trade_freq=trade_freq)
            else:
                raise ValueError("Invalid type")

            roi, investment = calculate_roi(portfolio, values[-1])
            # Optimize based on ROI or raw returns
            if optimize == 'raw':
                values_grid[i, j] = values[-1]
            elif optimize == 'roi':
                values_grid[i, j] = roi
            else:
                raise ValueError("Invalid optimize")

            print(
                f"std_thresh_enter: {std_thresh_enter}, std_thresh_exit: {std_thresh_exit}, capital_returns: {gains[-1]}, portfolio_value: {values[-1]}, roi: {roi}")

    print(f"Max return: {np.max(values_grid)}, Min return: {np.min(values_grid)}")

    # Determine the optimal threshold values
    max_idx = np.unravel_index(np.argmax(values_grid, axis=None), values_grid.shape)
    optimal_std_thresh_enter = std_thresh_enter_vals[max_idx[0]]
    optimal_std_thresh_exit = std_thresh_exit_vals[max_idx[1]]
    print(f"Optimal std_thresh_enter: {optimal_std_thresh_enter}")
    print(f"Optimal std_thresh_exit: {optimal_std_thresh_exit}")

    return values_grid, optimal_std_thresh_enter, optimal_std_thresh_exit


# Legacy implementations
"""
# Class to store stock inventory using min and max heaps for efficiency
# When a close signal is given, we want to sell the longs corresponding to the lowest prices first and buy to cover the shorts corresponding to the highest prices to maximize profit
class Inventory:
    def __init__(self, initial_cash=0.0):
        self.n_longs = 0
        self.n_shorts = 0
        self.longs = []  # Min-heap for long positions
        self.shorts = []  # Max-heap for short positions (using negative prices)
        self.long_positions = {}  # Dictionary to keep track of quantities at each price
        self.short_positions = {}  # Dictionary to keep track of quantities at each price
        self.initial_cash = initial_cash  # Initial cash balance
        self.cash = initial_cash  # Track cash balance

    def buy_long(self, price, quantity):
        # Cleaner indexing with ints
        quantity = int(quantity)
        price = int(round(price))

        self.n_longs += quantity
        self.cash -= price * quantity
        if price in self.long_positions:
            self.long_positions[price] += quantity
        else:
            self.long_positions[price] = quantity
            heapq.heappush(self.longs, price)

    def sell_long(self, current_price, quantity):
        quantity = int(quantity)
        current_price = int(round(current_price))

        total_profit = 0
        while quantity > 0 and self.longs:
            lowest_price = self.longs[0]
            if self.long_positions[lowest_price] > quantity:
                self.n_longs -= quantity
                self.long_positions[lowest_price] -= quantity
                total_profit += quantity * (current_price - lowest_price)
                self.cash += quantity * current_price
                quantity = 0
            else:
                self.n_longs -= self.long_positions[lowest_price]
                quantity -= self.long_positions[lowest_price]
                total_profit += self.long_positions[lowest_price] * (current_price - lowest_price)
                self.cash += self.long_positions[lowest_price] * current_price
                del self.long_positions[lowest_price]
                heapq.heappop(self.longs)
        return total_profit

    def short_sell(self, price, quantity):
        quantity = int(quantity)
        price = int(round(price))

        self.n_shorts += quantity
        self.cash += price * quantity
        if price in self.short_positions:
            self.short_positions[price] += quantity
        else:
            self.short_positions[price] = quantity
            heapq.heappush(self.shorts, -price)

    def buy_to_cover(self, current_price, quantity):
        quantity = int(quantity)
        current_price = int(round(current_price))

        total_profit = 0
        while quantity > 0 and self.shorts:
            highest_price = -self.shorts[0]
            if self.short_positions[highest_price] > quantity:
                self.n_shorts -= quantity
                self.short_positions[highest_price] -= quantity
                total_profit += quantity * (highest_price - current_price)
                self.cash -= quantity * current_price
                quantity = 0
            else:
                self.n_shorts -= self.short_positions[highest_price]
                quantity -= self.short_positions[highest_price]
                total_profit += self.short_positions[highest_price] * (highest_price - current_price)
                self.cash -= self.short_positions[highest_price] * current_price
                del self.short_positions[highest_price]
                heapq.heappop(self.shorts)
        return total_profit

    def total_portfolio_value(self, curr_sell_price, curr_buy_price):
        long_value = sum(curr_sell_price * qty for qty in self.long_positions.values())
        short_value = sum(curr_buy_price * qty for qty in self.short_positions.values())
        # Portfolio value is cash plus the value of long positions minus the value of short positions
        return self.cash + long_value - short_value

    def __str__(self):
        return f"Longs: {self.long_positions}, Shorts: {self.short_positions}"
"""


