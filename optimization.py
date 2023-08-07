"""
Just an example
"""
  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt  		  	   		  		 			  		 			     			  	 
import pandas as pd  		  	   		  		 			  		 			     			  	 
from util import get_data
from math import sqrt
import scipy.optimize as spo


def optimize_portfolio(  		  	   		  		 			  		 			     			  	 
    sd=dt.datetime(2008, 6, 1),
    ed=dt.datetime(2009, 6, 1),
    syms=["IBM", "X", "GLD", "JPM"],
    gen_plot=False,  		  	   		  		 			  		 			     			  	 
):
    dates = pd.date_range(sd, ed)  		  	   		  		 			  		 			     			  	 
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		  		 			  		 			     			  	 
    prices = prices_all[syms]  # only portfolio symbols  		  	   		  		 			  		 			     			  	 
    prices_spy = prices_all["SPY"]  # only SPY, for comparison later
    prices = prices.ffill()
    prices = prices.bfill()

    initial_guess = 1 / len(syms)
    initial_values = np.empty(len(syms))
    initial_values.fill(initial_guess)
    allocs = initial_values

    low = 0
    high = 1

    bounds = [(low, high) for stocks in range(len(syms))]
    constraints = ({'type': 'eq', 'fun': lambda inputs: np.sum(inputs) - 1})

    normed_prices = prices.divide(prices.iloc[0])
    result = spo.minimize(get_negative_sharp_ratio, allocs, args=normed_prices, method='SLSQP', bounds=bounds, constraints=constraints, options={'disp': True})

    allocs = result.x

    cr, adr, sddr, sr, port_val = get_portfolio_statistics(allocs, normed_prices)

    normed_spy = prices_spy / prices_spy[0]

    # Compare daily portfolio value with SPY using a normalized plot  		  	   		  		 			  		 			     			  	 
    if gen_plot:
        df_temp = pd.concat(
            [port_val, normed_spy], keys=["Portfolio", "SPY"], axis=1
        )
        ax = df_temp.plot(title='Daily Portfolio Value and SPY')
        ax.set_ylabel('Price')
        ax.set_xlabel('Date')
        ax.margins(x=0)
        plt.grid(linestyle='dotted')
        plt.savefig('./images/Figure1.png')

    return allocs, cr, adr, sddr, sr

def test_code():

    """  		  	   		  		 			  		 			     			  	 
    This function WILL NOT be called by the auto grader.  		  	   		  		 			  		 			     			  	 
    """
    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ["IBM", "X", "GLD", "JPM"]

    # Assess the portfolio  		  	   		  		 			  		 			     			  	 
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		  		 			  		 			     			  	 
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )  		  	   		  		 			  		 			     			  	 

    # Print statistics  		  	   		  		 			  		 			     			  	 
    print(f"Start Date: {start_date}")  		  	   		  		 			  		 			     			  	 
    print(f"End Date: {end_date}")  		  	   		  		 			  		 			     			  	 
    print(f"Symbols: {symbols}")  		  	   		  		 			  		 			     			  	 
    print(f"Allocations:{allocations}")  		  	   		  		 			  		 			     			  	 
    print(f"Sharpe Ratio: {sr}")  		  	   		  		 			  		 			     			  	 
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		  		 			  		 			     			  	 
    print(f"Average Daily Return: {adr}")  		  	   		  		 			  		 			     			  	 
    print(f"Cumulative Return: {cr}")


def get_portfolio_statistics(allocs, normed_prices):
    start_val = 1
    alloced = normed_prices * allocs
    pos_vals = alloced * start_val
    port_val = pos_vals.sum(axis=1)
    daily_rets = get_daily_return(port_val)
    daily_rets = daily_rets[1:]
    cum_ret = (port_val.iloc[-1] / port_val.iloc[0]) - 1
    avg_daily_ret = daily_rets.mean()
    std_daily_ret = daily_rets.std()
    sharp_ratio = sqrt(252) * (avg_daily_ret / std_daily_ret)
    return [cum_ret, avg_daily_ret, std_daily_ret, sharp_ratio, port_val]


def get_negative_sharp_ratio(allocs, prices):
   return -1*get_portfolio_statistics(allocs, prices)[3]


def get_daily_return(prices):
    return prices.pct_change(1)


if __name__ == "__main__":
    test_code()  		  	   		  		 			  		 			     			  	 

