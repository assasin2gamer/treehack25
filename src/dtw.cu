// File: dtw.cpp
#include "signal_processing.h"
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

float computeDTWDistance(const std::vector<float>& series1, const std::vector<float>& series2) {
    int n = series1.size();
    int m = series2.size();
    std::vector<std::vector<float>> dtw(n+1, std::vector<float>(m+1, std::numeric_limits<float>::infinity()));
    dtw[0][0] = 0.0f;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            float cost = fabs(series1[i-1] - series2[j-1]);
            dtw[i][j] = cost + std::min({dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1]});
        }
    }
    return dtw[n][m];
}
