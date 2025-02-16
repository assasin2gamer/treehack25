// File: ml_algorithm.cpp
#include "ml_algorithm.h"
#include <vector>
#include <mutex>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <limits>

// Helper function to compute average absolute diffusion from a series of prices.
double computeAverageDiffusion(const std::vector<double>& prices) {
    if (prices.size() < 2) return 0.0;
    double total = 0.0;
    for (size_t i = 1; i < prices.size(); i++) {
        total += std::fabs(prices[i] - prices[i - 1]);
    }
    return total / (prices.size() - 1);
}

// Helper function to simulate historical prices based on current price and volatility.
std::vector<double> simulateHistoricalPrices(double currentPrice, double volatility) {
    std::vector<double> prices;
    prices.push_back(currentPrice);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, currentPrice * volatility * 0.05);
    for (int i = 0; i < 9; i++) {
        double nextPrice = prices.back() + d(gen);
        prices.push_back(nextPrice);
    }
    return prices;
}

static const int FEATURE_DIM = 4;
static const int NUM_STATES = 2; // 0: short, 1: buy

class HiddenMarkovModel {
public:
    HiddenMarkovModel() : prevState(-1) {
        counts.assign(NUM_STATES, 0);
        means.assign(NUM_STATES, std::vector<double>(FEATURE_DIM, 0.0));
        variances.assign(NUM_STATES, std::vector<double>(FEATURE_DIM, 1.0));
        transitionCounts.assign(NUM_STATES, std::vector<int>(NUM_STATES, 1)); // Laplace smoothing
    }

    int predictState(const std::vector<double>& obs) {
        std::vector<double> stateProb(NUM_STATES, 0.0);
        for (int s = 0; s < NUM_STATES; s++) {
            double emitProb = emissionProbability(s, obs);
            double transProb = 1.0;
            if (prevState != -1) {
                transProb = static_cast<double>(transitionCounts[prevState][s]) /
                            (transitionCounts[prevState][0] + transitionCounts[prevState][1]);
            }
            stateProb[s] = emitProb * transProb;
        }
        int bestState = 0;
        double bestProb = stateProb[0];
        for (int s = 1; s < NUM_STATES; s++) {
            if (stateProb[s] > bestProb) {
                bestProb = stateProb[s];
                bestState = s;
            }
        }
        return bestState;
    }

    void updateModel(const std::vector<double>& obs, int state) {
        // Update transition counts if previous state exists.
        if (prevState != -1) {
            transitionCounts[prevState][state]++;
        }
        prevState = state;

        // Update emission parameters for the chosen state using incremental updates.
        counts[state]++;
        int n = counts[state];
        for (int i = 0; i < FEATURE_DIM; i++) {
            double oldMean = means[state][i];
            double newMean = oldMean + (obs[i] - oldMean) / n;
            double oldVar = variances[state][i];
            double newVar = (n == 1) ? 0.0 : oldVar + (obs[i] - oldMean) * (obs[i] - newMean);
            means[state][i] = newMean;
            variances[state][i] = (n > 1) ? newVar / (n - 1) : 1e-6;
            if (variances[state][i] < 1e-6) {
                variances[state][i] = 1e-6;
            }
        }
    }

private:
    int prevState;
    std::vector<int> counts;
    std::vector<std::vector<double>> means;
    std::vector<std::vector<double>> variances;
    std::vector<std::vector<int>> transitionCounts;

    double gaussianProbability(double x, double mean, double var) {
        double diff = x - mean;
        double exponent = -(diff * diff) / (2 * var);
        double denom = std::sqrt(2 * M_PI * var);
        return std::exp(exponent) / denom;
    }

    double emissionProbability(int state, const std::vector<double>& obs) {
        double prob = 1.0;
        for (int i = 0; i < FEATURE_DIM; i++) {
            double p = gaussianProbability(obs[i], means[state][i], variances[state][i]);
            prob *= p;
        }
        return prob;
    }
};

static HiddenMarkovModel hmm;
static std::mutex hmmMutex;

std::string mlTradeDecision(double currentPrice, double volatility) {
    std::vector<double> historicalPrices = simulateHistoricalPrices(currentPrice, volatility);
    double avgDiffusion = computeAverageDiffusion(historicalPrices);
    double diffusionRatio = currentPrice / (volatility + 1e-6);
    std::vector<double> feature = { currentPrice, volatility, avgDiffusion, diffusionRatio };

    int predictedState;
    {
        std::lock_guard<std::mutex> lock(hmmMutex);
        predictedState = hmm.predictState(feature);
        hmm.updateModel(feature, predictedState);
    }

    // 10% chance to randomly switch the decision
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    if (dis(gen) < 0.5) {
        predictedState = 1 - predictedState;
    }

    return (predictedState == 1) ? "buy" : "short";
}

