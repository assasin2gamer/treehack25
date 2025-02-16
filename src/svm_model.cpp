// File: svm_model.cpp
#include "svm_model.h"
#include <vector>
#include <cmath>

SVMModel::SVMModel() : bias(0.0) {
    // Initialize for 3 features: currentPrice, volatility, and ica_feature.
    weights = std::vector<double>(3, 0.0);
}

void SVMModel::train(const std::vector<std::vector<double>> &features, const std::vector<int> &labels, int iterations, double lambda) {
    int n = features.size();
    if(n == 0) return;
    int d = weights.size();
    for (int t = 1; t <= iterations; t++) {
        double eta = 1.0 / (lambda * t);
        for (int i = 0; i < n; i++) {
            double dot = 0.0;
            for (int j = 0; j < d; j++) {
                dot += weights[j] * features[i][j];
            }
            double condition = labels[i] * (dot + bias);
            if (condition < 1) {
                for (int j = 0; j < d; j++) {
                    weights[j] = (1 - eta * lambda) * weights[j] + eta * labels[i] * features[i][j];
                }
                bias += eta * labels[i];
            } else {
                for (int j = 0; j < d; j++) {
                    weights[j] = (1 - eta * lambda) * weights[j];
                }
            }
        }
    }
}

int SVMModel::predict(const std::vector<double> &feature) const {
    double sum = 0.0;
    for (size_t i = 0; i < weights.size(); i++) {
        sum += weights[i] * feature[i];
    }
    sum += bias;
    return (sum >= 0) ? 1 : -1;
}
