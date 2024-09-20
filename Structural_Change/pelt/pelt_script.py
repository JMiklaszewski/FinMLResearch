import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import ruptures as rpt
import cProfile
import math
from tkinter import Tk, filedialog
from tkinter import Tk
from tkinter.simpledialog import askstring, askinteger

## Function to dynamically select input file
def read_input_file() -> pd.Series:
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select a CSV file", filetypes=[("CSV files", "*.csv")])
    file = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return file[file.columns[0]] # return a pd.Series

# Function to apply PELT algorithm with a dynamically calculated penalty
def apply_pelt(signal:np.array):
    # Calculate penalty based on BIC
    penalty = math.ceil(np.log(len(signal)))  # BIC penalty (log of data length)
    
    # Apply the PELT algorithm
    algo = rpt.Pelt(model="rbf").fit(signal)
    best_segments = algo.predict(pen=penalty)
    
    return penalty, best_segments

def get_user_input(message : str, type=int) -> int:
    """
    Gets user input for the type of analysis (return or volatility) and window size if needed.
    """
    root = Tk()
    root.withdraw()  # Hide the main tkinter window

    # Ask for rolling window size for CUSUM calculation
    in_window = askinteger("Input", message) if type is int else askstring('Input', message)
    return in_window

# Main function to profile the code
def main(plot=True):

    # Ask user to specify the input file
    in_returns = read_input_file()

    print('PELT: Input file read')

    # Ask user todefine the type of task
    TASK_TYPE = get_user_input(message='Specify the type of program task ("ret" - Returns; "vol" - Volatility)', type=str)

    # If the volatility option has been selected, ask for the volatility window, and estimate std
    if TASK_TYPE == 'vol':
        # Get the rolling window for vol
        VOL_WINDOW = get_user_input(message='Specify the rolling windows base for volatility:', type=int)

        # Compute the rolling vol - and transform to log scale
        signal = in_returns.rolling(window=VOL_WINDOW, min_periods=1).std().apply(lambda x: np.log(x))
    
        print(f'PELT: Running with std window - {VOL_WINDOW}')

    # Apply the PELT algorithm using the dynamically calculated penalty (BIC-based)
    signal = in_returns.values.reshape(-1, 1)  # Convert to NumPy array
    penalty, best_segments = apply_pelt(signal)
    
    # Output the penalty and detected change points
    print(f"Used Penalty (BIC-based): {penalty}")
    print(f"Detected change points: {best_segments}")
    
    # Create a list of 0s and 1s where 1 means it's a change point and 0 means it's not
    change_point_flags = [0] * len(in_returns)
    for cp in best_segments[:-1]:  # Exclude the last change point (it's at the end)
        change_point_flags[cp] = 1

    # Create a DataFrame with dates and change point flags
    df = pd.DataFrame({
        'Date': in_returns.index,
        'Return': in_returns.values,
        'Change Point Flag': change_point_flags
    })

    # Export the results of pelt algorithm
    df.to_csv(f'pelt_results_{in_returns.name}.csv')
    
    if plot:
        # Plot the results with detected change points
        plt.figure(figsize=(10, 6))
        plt.plot(in_returns.index, in_returns, label=f"{in_returns.name}_{TASK_TYPE}")
    
        for cp in best_segments[:-1]:  # Exclude the last change point (it's at the end)
            plt.axvline(x=in_returns.index[cp], color='red', linestyle='--', label='Change Point')
    
        plt.title(f'Equities [{TASK_TYPE}] with Change Point Detection (PELT, Penalty={penalty})')
        plt.legend()
        plt.show()

# Profile the main function to identify bottlenecks
if __name__ == "__main__":
    main()