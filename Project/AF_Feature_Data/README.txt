Taken raw data
Split up in windows of 30 seconds
If more than 75% AF, then label 1, else label 0
Every windows was bounded to remove the erroneous measurements in the raw data labeled with PAUSE
For every window we calculated:
     - number of samples
 * on original
     - mean
     - var
     - max
     - min
     - spread
     - mean absolute deviation
     - kurtosis
     - skew
     - median
     - first quantile
     - last quantile
     - sum
 * on moving average 10 timestamps
     - mean
     - var
     - max
     - min
     - spread
     - mean absolute deviation
     - kurtosis
     - skew
     - median
     - first quantile
     - last quantile
     - sum
 * on first order diff
     - mean
     - var
     - max
     - min
     - spread
     - mean absolute deviation
     - kurtosis
     - skew
     - median
     - first quantile
     - last quantile
     - sum
