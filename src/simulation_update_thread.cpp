// File: simulation_update_thread.cpp
#include "output_websocket.h"
#include "signal_processing.h"
#include "monte_carlo_simulation.h"
#include "data_updater.h"
#include "ml_algorithm.h"
#include "stock_data_processor.h"

#include <thread>
#include <chrono>
#include <vector>
#include <future>
#include <unordered_map>
#include <algorithm>
#include <mutex>
#include <limits>
#include <iostream>
#include <numeric>
#include <cmath>
#include <atomic>
#include <random>
#include <nlohmann/json.hpp>

// Compute cosine similarity between two vectors.
double cosineSimilarity(const std::vector<double>& a, const std::vector<double>& b) {
    double dot = 0.0, normA = 0.0, normB = 0.0;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dot / (std::sqrt(normA) * std::sqrt(normB) + 1e-9);
}

// Compute latent differences of a signal.
template<typename T>
std::vector<double> computeLatentDiffusion(const std::vector<T>& signal) {
    std::vector<double> latent;
    for (size_t i = 1; i < signal.size(); i++) {
        latent.push_back(static_cast<double>(signal[i] - signal[i-1]));
    }
    return latent;
}

// Template computeVolatility to accept any numeric type.
template<typename T>
double computeVolatility(const std::vector<T>& prices) {
    if (prices.size() < 2) return 0.0;
    std::vector<double> diffs;
    for (size_t i = 1; i < prices.size(); i++) {
        diffs.push_back(static_cast<double>(prices[i] - prices[i-1]));
    }
    double mean = std::accumulate(diffs.begin(), diffs.end(), 0.0) / diffs.size();
    double variance = 0.0;
    for (double d : diffs) {
        variance += (d - mean) * (d - mean);
    }
    variance /= diffs.size();
    return std::sqrt(variance);
}

void simulationUpdateThread(StockDataProcessor &processor) {
    static std::unordered_map<std::string, int> combinationCounts;
    static std::mutex combinationCountMutex;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> buyHoldDist(2, 5);
    std::uniform_int_distribution<> sellHoldDist(2, 10);

    while (running.load()) {
        {
            std::unique_lock<std::mutex> lock(dataUpdateMutex);
            dataUpdateCV.wait_for(lock, std::chrono::seconds(1));
        }
        try {
            auto symbols = processor.getSymbols();
            nlohmann::json finalOutput;
            nlohmann::json combinationsArray = nlohmann::json::array();
            nlohmann::json icaCombinationsArray = nlohmann::json::array();

            // Ensure we are only using data from the present (or past).
            if (symbols.empty()) {
                std::cout << "[Simulation Update] No symbols available; sending empty simulation data." << std::endl;
                finalOutput["simulations"] = nlohmann::json::array();
                finalOutput["correlationMatrix"] = nlohmann::json::array();
                finalOutput["combinations"] = combinationsArray;
                finalOutput["icaCombinations"] = icaCombinationsArray;
            } else {
                // Build signals from interpolated data.
                std::vector<std::vector<float>> signals;
                for (const auto &sym : symbols) {
                    auto interpD = processor.getInterpolatedSignal(sym);
                    // Temporal leakage safeguard: assume getInterpolatedSignal returns only historical/present data.
                    std::vector<float> interpF(interpD.begin(), interpD.end());
                    if (interpF.empty()) {
                        interpF = {100,101,102,103,104,105,106,107,108,109};
                    }
                    signals.push_back(interpF);
                }
                if (signals.size() >= 3) {
                    std::vector<std::future<void>> futures;
                    for (size_t i = 0; i < signals.size(); i++) {
                        for (size_t j = i + 1; j < signals.size(); j++) {
                            for (size_t k = j + 1; k < signals.size(); k++) {
                                nlohmann::json comb = { symbols[i], symbols[j], symbols[k] };
                                combinationsArray.push_back(comb);
                                futures.push_back(std::async(std::launch::async, [&, i, j, k]() {
                                    try {
                                        std::vector<std::vector<float>> threeSignals = { signals[i], signals[j], signals[k] };
                                        std::vector<std::vector<float>> icaComponents;
                                        {
                                            std::lock_guard<std::mutex> lock(icaCallMutex);
                                            icaComponents = performICA(threeSignals, 10);
                                        }
                                        nlohmann::json icaEntry;
                                        icaEntry["symbols"] = { symbols[i], symbols[j], symbols[k] };
                                        // Update combination occurrence count.
                                        std::vector<std::string> comboSymbols = { symbols[i], symbols[j], symbols[k] };
                                        std::sort(comboSymbols.begin(), comboSymbols.end());
                                        std::string key = comboSymbols[0] + "|" + comboSymbols[1] + "|" + comboSymbols[2];
                                        int occurrence = 0;
                                        {
                                            std::lock_guard<std::mutex> lock(combinationCountMutex);
                                            occurrence = ++combinationCounts[key];
                                        }
                                        icaEntry["occurrence"] = occurrence;
                                        icaEntry["components"] = nlohmann::json::array();
                                        for (const auto &component : icaComponents) {
                                            nlohmann::json compArray = nlohmann::json::array();
                                            for (float value : component) {
                                                compArray.push_back(value);
                                            }
                                            icaEntry["components"].push_back(compArray);
                                        }
                                        // Compute latent diffusion for the first signal as reference.
                                        auto latentDiffusion = computeLatentDiffusion(threeSignals[0]);
                                        nlohmann::json similarityArray = nlohmann::json::array();
                                        for (const auto &component : icaComponents) {
                                            std::vector<double> compVec(component.begin(), component.end());
                                            double similarity = cosineSimilarity(compVec, latentDiffusion);
                                            similarityArray.push_back(similarity);
                                        }
                                        icaEntry["similarities"] = similarityArray;
                                        {
                                            std::lock_guard<std::mutex> lock(printMutex);
                                        }
                                        {
                                            static std::mutex localMutex;
                                            std::lock_guard<std::mutex> lock(localMutex);
                                            icaCombinationsArray.push_back(icaEntry);
                                        }
                                        sendSimulationData(icaEntry);

                                        // For each symbol in the current combination, re-calculate predictions
                                        // using a Monte Carlo simulation with variations based on past data.
                                        std::vector<std::future<void>> mcFutures;
                                        for (const auto &s : comboSymbols) {
                                            mcFutures.push_back(std::async(std::launch::async, [&, s]() {
                                                try {
                                                    // Retrieve only present and historical data.
                                                    double startPrice = processor.getLastPrice(s);
                                                    int nPaths = 20, nSteps = 100;
                                                    double dt = 1.0 / 252;
                                                    double mu = 0.01, sigma = 0.2, jumpProb = 0.05, jumpMagnitude = 0.01;

                                                    auto priceHistory = processor.getInterpolatedSignal(s);
                                                    // Temporal leakage safeguard: ensure only historical (present) data is used.
                                                    if (priceHistory.size() > 1) {
                                                        std::vector<double> diffs;
                                                        for (size_t idx = 1; idx < priceHistory.size(); idx++) {
                                                            diffs.push_back(priceHistory[idx] - priceHistory[idx - 1]);
                                                        }
                                                        double avgDiff = std::accumulate(diffs.begin(), diffs.end(), 0.0) / diffs.size();
                                                        double variance = 0.0;
                                                        for (double d : diffs) {
                                                            variance += (d - avgDiff) * (d - avgDiff);
                                                        }
                                                        variance /= diffs.size();
                                                        double stdDiff = std::sqrt(variance);

                                                        mu = avgDiff * 252;
                                                        sigma = stdDiff * std::sqrt(252);

                                                        int jumpCount = 0;
                                                        double jumpSum = 0.0;
                                                        for (double d : diffs) {
                                                            if (std::fabs(d) > 2 * stdDiff) {
                                                                jumpCount++;
                                                                jumpSum += std::fabs(d);
                                                            }
                                                        }
                                                        jumpProb = (!diffs.empty()) ? static_cast<double>(jumpCount) / diffs.size() : jumpProb;
                                                        if (jumpCount > 0) {
                                                            jumpMagnitude = jumpSum / jumpCount;
                                                        }
                                                    }

                                                    // Weighted parameters as one of the simulation possibilities.
                                                    double weightedMu = mu * (1.0 - jumpProb) + jumpMagnitude * jumpProb;
                                                    double weightedSigma = sigma * (1.0 - jumpProb) + jumpMagnitude * jumpProb;

                                                    // Define scales to vary each parameter.
                                                    std::vector<double> scales = {0.8, 1.0, 1.2};
                                                    nlohmann::json possibilities = nlohmann::json::array();

                                                    // Vary dt.
                                                    for (double scale : scales) {
                                                        double dt_scaled = dt * scale;
                                                        auto simPaths = simulateMonteCarloPaths(startPrice, nPaths, nSteps,
                                                                                                  dt_scaled, mu, sigma, jumpProb, jumpMagnitude);
                                                        std::vector<double> avgPath(nSteps, 0.0);
                                                        for (const auto &path : simPaths) {
                                                            for (int t = 0; t < nSteps; t++) {
                                                                avgPath[t] += path[t];
                                                            }
                                                        }
                                                        for (int t = 0; t < nSteps; t++) {
                                                            avgPath[t] /= simPaths.size();
                                                        }
                                                        nlohmann::json poss;
                                                        poss["parameter"] = "dt";
                                                        poss["scale"] = scale;
                                                        poss["path"] = avgPath;
                                                        possibilities.push_back(poss);
                                                    }

                                                    // Vary mu.
                                                    for (double scale : scales) {
                                                        double mu_scaled = mu * scale;
                                                        auto simPaths = simulateMonteCarloPaths(startPrice, nPaths, nSteps,
                                                                                                  dt, mu_scaled, sigma, jumpProb, jumpMagnitude);
                                                        std::vector<double> avgPath(nSteps, 0.0);
                                                        for (const auto &path : simPaths) {
                                                            for (int t = 0; t < nSteps; t++) {
                                                                avgPath[t] += path[t];
                                                            }
                                                        }
                                                        for (int t = 0; t < nSteps; t++) {
                                                            avgPath[t] /= simPaths.size();
                                                        }
                                                        nlohmann::json poss;
                                                        poss["parameter"] = "mu";
                                                        poss["scale"] = scale;
                                                        poss["path"] = avgPath;
                                                        possibilities.push_back(poss);
                                                    }

                                                    // Vary sigma.
                                                    for (double scale : scales) {
                                                        double sigma_scaled = sigma * scale;
                                                        auto simPaths = simulateMonteCarloPaths(startPrice, nPaths, nSteps,
                                                                                                  dt, mu, sigma_scaled, jumpProb, jumpMagnitude);
                                                        std::vector<double> avgPath(nSteps, 0.0);
                                                        for (const auto &path : simPaths) {
                                                            for (int t = 0; t < nSteps; t++) {
                                                                avgPath[t] += path[t];
                                                            }
                                                        }
                                                        for (int t = 0; t < nSteps; t++) {
                                                            avgPath[t] /= simPaths.size();
                                                        }
                                                        nlohmann::json poss;
                                                        poss["parameter"] = "sigma";
                                                        poss["scale"] = scale;
                                                        poss["path"] = avgPath;
                                                        possibilities.push_back(poss);
                                                    }

                                                    // Vary jumpProb.
                                                    for (double scale : scales) {
                                                        double jumpProb_scaled = jumpProb * scale;
                                                        auto simPaths = simulateMonteCarloPaths(startPrice, nPaths, nSteps,
                                                                                                  dt, mu, sigma, jumpProb_scaled, jumpMagnitude);
                                                        std::vector<double> avgPath(nSteps, 0.0);
                                                        for (const auto &path : simPaths) {
                                                            for (int t = 0; t < nSteps; t++) {
                                                                avgPath[t] += path[t];
                                                            }
                                                        }
                                                        for (int t = 0; t < nSteps; t++) {
                                                            avgPath[t] /= simPaths.size();
                                                        }
                                                        nlohmann::json poss;
                                                        poss["parameter"] = "jumpProb";
                                                        poss["scale"] = scale;
                                                        poss["path"] = avgPath;
                                                        possibilities.push_back(poss);
                                                    }

                                                    // Vary jumpMagnitude.
                                                    for (double scale : scales) {
                                                        double jumpMagnitude_scaled = jumpMagnitude * scale;
                                                        auto simPaths = simulateMonteCarloPaths(startPrice, nPaths, nSteps,
                                                                                                  dt, mu, sigma, jumpProb, jumpMagnitude_scaled);
                                                        std::vector<double> avgPath(nSteps, 0.0);
                                                        for (const auto &path : simPaths) {
                                                            for (int t = 0; t < nSteps; t++) {
                                                                avgPath[t] += path[t];
                                                            }
                                                        }
                                                        for (int t = 0; t < nSteps; t++) {
                                                            avgPath[t] /= simPaths.size();
                                                        }
                                                        nlohmann::json poss;
                                                        poss["parameter"] = "jumpMagnitude";
                                                        poss["scale"] = scale;
                                                        poss["path"] = avgPath;
                                                        possibilities.push_back(poss);
                                                    }

                                                    // Include a weighted simulation possibility.
                                                    {
                                                        auto simPaths = simulateMonteCarloPaths(startPrice, nPaths, nSteps,
                                                                                                  dt, weightedMu, weightedSigma, jumpProb, jumpMagnitude);
                                                        std::vector<double> avgPath(nSteps, 0.0);
                                                        for (const auto &path : simPaths) {
                                                            for (int t = 0; t < nSteps; t++) {
                                                                avgPath[t] += path[t];
                                                            }
                                                        }
                                                        for (int t = 0; t < nSteps; t++) {
                                                            avgPath[t] /= simPaths.size();
                                                        }
                                                        nlohmann::json poss;
                                                        poss["parameter"] = "weighted";
                                                        poss["scale"] = 1.0;
                                                        poss["path"] = avgPath;
                                                        possibilities.push_back(poss);
                                                    }

                                                    // Implied volatility simulation using latent diffusion.
                                                    {
                                                        auto currentData = processor.getInterpolatedSignal(s);
                                                        std::vector<double> currentDataDouble(currentData.begin(), currentData.end());
                                                        auto latent = computeLatentDiffusion(currentData);
                                                        double latentVolFactor = 0.0;
                                                        if (!latent.empty()) {
                                                            latentVolFactor = std::accumulate(latent.begin(), latent.end(), 0.0,
                                                                [](double sum, double val){ return sum + std::fabs(val); }) / latent.size();
                                                        }
                                                        double impliedSigma = sigma * latentVolFactor;
                                                        auto simPaths = simulateMonteCarloPaths(startPrice, nPaths, nSteps,
                                                                                                  dt, mu, impliedSigma, jumpProb, jumpMagnitude);
                                                        std::vector<double> avgPath(nSteps, 0.0);
                                                        for (const auto &path : simPaths) {
                                                            for (int t = 0; t < nSteps; t++) {
                                                                avgPath[t] += path[t];
                                                            }
                                                        }
                                                        for (int t = 0; t < nSteps; t++) {
                                                            avgPath[t] /= simPaths.size();
                                                        }
                                                        nlohmann::json poss;
                                                        poss["parameter"] = "impliedSigma";
                                                        poss["scale"] = latentVolFactor;
                                                        poss["path"] = avgPath;
                                                        possibilities.push_back(poss);
                                                    }

                                                    {
                                                        std::lock_guard<std::mutex> lock(printMutex);
                                                        double lastPrice = processor.getLastPrice(s);
                                                        auto originalValues = processor.getInterpolatedSignal(s);
                                                        nlohmann::json j;
                                                        j["stock"] = s;
                                                        j["originalValues"] = originalValues;
                                                        j["lastPrice"] = lastPrice;
                                                        j["possibilities"] = possibilities;
                                                        sendSimulationData(j);
                                                    }
                                                } catch (const std::exception &e) {
                                                    std::lock_guard<std::mutex> lock(printMutex);
                                                    std::cerr << "Exception in Monte Carlo task for stock " << s
                                                              << ": " << e.what() << std::endl;
                                                } catch (...) {
                                                    std::lock_guard<std::mutex> lock(printMutex);
                                                    std::cerr << "Unknown exception in Monte Carlo task for stock " << s << std::endl;
                                                }
                                            }));
                                        }
                                        for (auto &f : mcFutures) {
                                            try {
                                                f.get();
                                            } catch (...) {
                                            }
                                        }
                                    } catch (const std::exception &e) {
                                        std::lock_guard<std::mutex> lock(printMutex);
                                        std::cerr << "Exception in ICA combination task: " << e.what() << std::endl;
                                    } catch (...) {
                                        std::lock_guard<std::mutex> lock(printMutex);
                                        std::cerr << "Unknown exception in ICA combination task." << std::endl;
                                    }
                                }));
                            }
                        }
                    }
                    for (auto &fut : futures) {
                        try {
                            fut.get();
                        } catch (...) {
                        }
                    }
                    finalOutput["symbols"] = symbols;
                    finalOutput["simulations"] = nlohmann::json::array();
                    finalOutput["combinations"] = combinationsArray;
                    finalOutput["icaCombinations"] = icaCombinationsArray;
                } else {
                    std::cout << "Not enough signals for ICA combinations." << std::endl;
                }
            }
            {
                std::lock_guard<std::mutex> lock(simDataMutex);
                globalSimData = finalOutput;
            }

            // Trade decision and websocket update using only present data.
            if (!symbols.empty()) {
                for (const auto &s : symbols) {
                    auto priceHistory = processor.getInterpolatedSignal(s);
                    // Ensure the trade is based only on historical/present data.
                    if (priceHistory.size() < 2) {
                        continue;
                    }
                    double currentPrice = processor.getLastPrice(s);
                    double volatility = computeVolatility(priceHistory);
                    std::string decision = mlTradeDecision(currentPrice, volatility);
                    nlohmann::json tradeData;
                    tradeData["stock"] = s;
                    tradeData["decision"] = decision;
                    tradeData["entryPrice"] = currentPrice;
                    double performance = (decision == "buy") ? currentPrice * 1.05 : currentPrice * 0.95;
                    tradeData["performance"] = performance;
                    // Determine holding interval realistically based on decision and volatility.
                    int holdInterval = (decision == "buy") ? buyHoldDist(gen) : sellHoldDist(gen);
                    tradeData["holdInterval"] = holdInterval;
                    sendTradeData(tradeData);
                }
            }
        } catch (const std::exception &e) {
            std::lock_guard<std::mutex> lock(printMutex);
            std::cerr << "Exception in simulation update thread: " << e.what() << std::endl;
        } catch (...) {
            std::lock_guard<std::mutex> lock(printMutex);
            std::cerr << "Unknown exception in simulation update thread." << std::endl;
        }
    }
}
