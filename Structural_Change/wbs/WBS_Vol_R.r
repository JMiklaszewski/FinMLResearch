# Load necessary libraries
library(quantmod)
library(wbs)
library(zoo)

# Step 1: Fetch S&P 500 Data from Yahoo Finance
ticker <- "^GSPC"
getSymbols(ticker, src = "yahoo", from = "1994-01-01", to = "2024-08-31")

# Step 2: Calculate daily log returns
log_returns <- diff(log(Cl(GSPC)))  # Calculate log returns
log_returns <- na.omit(log_returns)  # Remove NA values

# Step 3: Calculate 120-day rolling volatility (6-month window)
rolling_vol_120 <- rollapply(log_returns, width = 30, FUN = sd, by = 1, align = "right", fill = NA)

# Step 4: Log-transform the rolling volatility to handle skewness
log_volatility_120 <- log(rolling_vol_120)

# Step 5: Standardize the log-transformed volatility (mean = 0, sd = 1)
scaled_volatility_120 <- scale(log_volatility_120)

# Remove NA values from the scaled volatility before applying WBS
scaled_volatility_120 <- na.omit(scaled_volatility_120)

# Step 6: Apply Wild Binary Segmentation (WBS) to the cleaned, scaled volatility
set.seed(123)  # For reproducibility
wbs_result_vol <- wbs(as.numeric(scaled_volatility_120), M = 1000, rand.intervals = TRUE, integrated = TRUE)

# Check WBS result to see if it's been computed correctly
print(paste("Number of intervals analyzed by WBS:", length(wbs_result_vol$res[,1])))

# Step 7: Use the MBIC penalty for stricter changepoint detection
seg_vol <- changepoints(wbs_result_vol, penalty = "mbic.penalty")

# Check if changepoints were detected
if (length(seg_vol$cpt.ic$mbic) == 0) {
  print("No changepoints were detected.")
} else {
  print(paste("Number of changepoints detected:", length(seg_vol$cpt.ic$mbic)))
}

# Step 8: Create a binary vector with 1 for changepoints and 0 for others
change_point_flag_vol <- rep(0, NROW(log_volatility_120))
change_point_flag_vol[seg_vol$cpt.ic$mbic] <- 1  # Set 1 at detected changepoints using MBIC

# Combine with dates
output_data_vol <- data.frame(Date = index(rolling_vol_120), ChangePoint = change_point_flag_vol)

# Step 9: Save the output as a CSV file
write.csv(output_data_vol, "S&P500_120_day_volatility_change_points_mbic.csv", row.names = FALSE)

# Step 10: Plot the rolling volatility with detected changepoints indicated
plot(rolling_vol_120, main = "120-Day Rolling Volatility with WBS Detected Change Points (MBIC Penalty)", col = "blue", type = "l")
if (length(seg_vol$cpt.ic$mbic) > 0) {
  abline(v = index(rolling_vol_120)[seg_vol$cpt.ic$mbic], col = "red", lty = 2)  # Add changepoint lines
} else {
  print("No changepoints to plot.")
}
