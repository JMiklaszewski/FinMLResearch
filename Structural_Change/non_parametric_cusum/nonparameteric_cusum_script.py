import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import rankdata
from tkinter.messagebox import showinfo
from tkinter.filedialog import askopenfilename
from tkinter import Tk
from tkinter.simpledialog import askstring, askinteger

# CUSUM with nonparametric test
def cusum_nonparametric_test(data:pd.Series, window_size:int):
    """
    Apply CUSUM with a Nonparametric Test to detect change points and output a flag when threshold is breached.
    
    Parameters:
    - data: The time series data (e.g., S&P 500 returns).
    - window_size: The rolling window size (e.g., 60 days).
    
    Returns:
    - flags: A list of flags (True/False) indicating threshold breach.
    - rolling_mean: The rolling mean of the ranked data.
    """
    ranked_data = rankdata(data)
    
    # Calculate rolling mean of ranked data
    rolling_mean = pd.Series(ranked_data).rolling(window=window_size, min_periods=1).mean()
    
    # Calculate CUSUM
    cusum_nonp = np.cumsum(ranked_data - rolling_mean)
    cusum_nonp -= np.mean(cusum_nonp)  # Normalize
    
    # Detect change points using a threshold based on the standard deviation of CUSUM
    threshold = 3 * np.std(cusum_nonp)
    
    flags = np.abs(cusum_nonp) > threshold  # Boolean flags for threshold breaches
    
    return flags, rolling_mean

# Rolling volatility function
def rolling_volatility(data:pd.Series, window_size:int) -> pd.Series:
    """
    Calculate rolling volatility (standard deviation) of daily returns.
    
    Parameters:
    - data: Time series data (e.g., S&P 500 returns).
    - window_size: The size of the rolling window (e.g., 30 days).
    
    Returns:
    - rolling_vol: The rolling volatility (standard deviation) over the window.
    """
    return data.rolling(window=window_size).std()

def cusum_nonparam_algo(series:pd.Series, rets=True, rolling_window_ret=None, rolling_window_vol=None, plot=True) -> pd.DataFrame:
    # Compute returns if not already given
    if rets is not True:
        series = series.pct_change().dropna() # please note this method will just produce t=1 returns

    # Apply CUSUM nonparametric test on returns
    flags_returns, rolling_mean = cusum_nonparametric_test(series, window_size=rolling_window_ret)

    # Compute rolling volatility
    rolling_vol = rolling_volatility(series, window_size=rolling_window_vol)

    # Detect change points on voliatility (using 3X threshold) - same logic as per returns
    flags_vol = np.abs(rolling_vol - rolling_vol.mean()) > (3 * rolling_vol.std())

    # Collect change points indicators
    out_list = pd.DataFrame(index=series.index)
    out_list['ret_flag'] = [1 if flag_r else 0 for flag_r in flags_returns]
    out_list['vol_flag'] = [1 if flag_v else 0 for flag_v in flags_vol]

    if plot: out_list.plot(title = f'Non-parametric CUSUM on Equity {[TASK_TYPE]} : {series.name}')

    plt.show()

    return out_list

def open_csv_file(date_col=0) -> pd.Series:
    # Read the file path
    csv_path = askopenfilename(title='Select Input File', filetypes=[("CSV Files", "*.csv")])
    # Return the input file
    in_df = pd.read_csv(csv_path, index_col=date_col, parse_dates=True)
    # Make sure we return pd.Series
    return in_df[in_df.columns[0]]

def get_user_input(message : str, type=int) -> int:
    """
    Gets user input for the type of analysis (return or volatility) and window size if needed.
    """
    root = Tk()
    root.withdraw()  # Hide the main tkinter window

    # Ask for rolling window size for CUSUM calculation
    in_window = askinteger("Input", message) if type is int else askstring('Input', message)
    return in_window


if __name__ == '__main__':

    # Set the variables for non-paremetric CUSUM test
    global TASK
    TASK_TYPE = get_user_input('Specify the type of task (Return : ret, Volatlity : vol)', type=str)

    # Read the input file
    series = open_csv_file()

    print(f'Non-param CUSUM: Input file read, running program for: {TASK_TYPE}')

    # If the volatility option has been selected
    if TASK_TYPE == 'vol':
        # Ask user for rolling window of vol
        VOL_WINDOW = get_user_input('Specify the base (window) for volatility input:', type=int)

        # Compute rolling volatlity input
        series = series.rolling(window=VOL_WINDOW, min_periods=1).std().apply(lambda x: np.log(x))

        print(f'Non-param CUSUM: Rolling base (window) for vol: {VOL_WINDOW}')

    ROLLING_WINDOW_RETURNS = get_user_input('Specify the size of rolling window - returns', type=int)
    ROLLING_WINDOW_VOL = get_user_input('Specify the size of rolling window - volatility', type=int)

    print(f'Non-param CUSUM: Rolling windows for CUSUM: [Return - {ROLLING_WINDOW_RETURNS}; Vol - {ROLLING_WINDOW_VOL}]')

    # Run the non-parametric CUSUM algorithm
    nonparam_cusum_results = cusum_nonparam_algo(
        series, 
        rets=True, # make sure to double-check if the input file is truly returns
        rolling_window_ret=ROLLING_WINDOW_RETURNS,
        rolling_window_vol=ROLLING_WINDOW_VOL,
        plot=True)
    
    # Output results
    out_file_name = f'nonp_cusum_results_{series.name}_{TASK_TYPE}.csv'
    nonparam_cusum_results.to_csv(out_file_name, index=True)
    showinfo('Information', f'Program has finished running, results saved ({out_file_name})')