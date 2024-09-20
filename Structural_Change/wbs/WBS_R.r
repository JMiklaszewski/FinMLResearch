# Load libraries
library(quantmod)
library(wbs)
library(zoo)

# Step 1: Fetch S&P 500 Data from Yahoo Finance (using daily returns)
ticker <- "^GSPC"
getSymbols(ticker, src = "yahoo", from = "1995-01-01", to = "2024-08-31")

# Calculate daily log returns (no downsampling for more granular detection)
log_returns <- diff(log(Cl(GSPC)))  # Calculate log returns
log_returns <- na.omit(log_returns)  # Remove NA values

# Check if log_returns has valid data
print(paste("Number of log returns:", length(log_returns)))

# Step 2: Apply Wild Binary Segmentation (WBS) on daily returns with more intervals
set.seed(123)  # For reproducibility
wbs_result <- wbs(as.numeric(log_returns), M = 10000, rand.intervals = TRUE, integrated = TRUE)

# Check WBS result to see if it's been computed correctly
print(paste("Number of intervals analyzed by WBS:", length(wbs_result$res[,1])))

# Step 3: Use a BIC penalty for less strict detection
seg <- changepoints(wbs_result, penalty = "bic.penalty")

# Check if changepoints were detected
if (length(seg$cpt.ic$bic) == 0) {
  print("No changepoints were detected.")
} else {
  print(paste("Number of changepoints detected:", length(seg$cpt.ic$bic)))
}

# Step 4: Create a binary vector with 1 for changepoints and 0 for others
change_point_flag <- rep(0, NROW(log_returns))
change_point_flag[seg$cpt.ic$bic] <- 1  # Set 1 at detected changepoints using BIC

# Combine with dates
output_data <- data.frame(Date = index(log_returns), ChangePoint = change_point_flag)

# Step 5: Save the output as a CSV file
write.csv(output_data, "S&P500_daily_returns_change_points_bic_final.csv", row.names = FALSE)

# Step 6: Plot the results with changepoints indicated
plot(log_returns, main = "S&P 500 Daily Returns with WBS Detected Change Points (BIC Penalty)")
if (length(seg$cpt.ic$bic) > 0) {
  abline(v = index(log_returns)[seg$cpt.ic$bic], col = "red", lty = 2)  # Add changepoint lines
} else {
  print("No changepoints to plot.")
}
