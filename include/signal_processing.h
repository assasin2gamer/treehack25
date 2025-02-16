// File: signal_processing.h
#ifndef ICA_PROCESSOR_H
#define ICA_PROCESSOR_H

#include <vector>

// Perform ICA on the provided signals using CUDA fastICA implementation.
// Each inner vector represents one channel with n_samples values.
std::vector<std::vector<float>> performICA(const std::vector<std::vector<float>>& signals, int n_samples);
// Computes DTW Distance
float computeDTWDistance(const std::vector<float>& series1, const std::vector<float>& series2);



#endif // ICA_PROCESSOR_H
