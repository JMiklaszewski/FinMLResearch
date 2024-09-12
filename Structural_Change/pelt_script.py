import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import ruptures as rpt
import cProfile
import math
from tkinter import Tk, filedialog

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

# Main function to profile the code
def main(plot=True):

    # Ask user to specify the input file
    in_returns = read_input_file()

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
        plt.plot(in_returns.index, in_returns, label=f"{in_returns.name} Returns")
    
        for cp in best_segments[:-1]:  # Exclude the last change point (it's at the end)
            plt.axvline(x=in_returns.index[cp], color='red', linestyle='--', label='Change Point')
    
        plt.title(f'Equities Returns with Change Point Detection (PELT, Penalty={penalty})')
        plt.legend()
        plt.show()

# Profile the main function to identify bottlenecks
if __name__ == "__main__":
    main()