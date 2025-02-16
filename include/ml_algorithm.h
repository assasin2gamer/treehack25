// File: ml_algorithm.h
#ifndef ML_ALGORITHM_H
#define ML_ALGORITHM_H

#include <string>

// Returns "buy" or "short" based on SVM prediction using currentPrice, volatility, and an ICA-derived feature.
std::string mlTradeDecision(double currentPrice, double volatility);

#endif
