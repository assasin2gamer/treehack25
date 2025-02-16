// File: stock_data_processor.cpp
#include "stock_data_processor.h"
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <mutex>
#include <iostream>
#include <algorithm>
#include <stdexcept>

namespace {
    std::unordered_map<std::string, std::vector<double>> processorStockPrices;
    std::mutex processorMutex;

    // Interpolates the given price vector to 10 evenly spaced points.
    std::vector<double> interpolatePrices(const std::vector<double>& prices) {
        std::vector<double> interp(10, 0.0);
        size_t n = prices.size();
        if(n == 0) return interp;
        if(n == 1) {
            std::fill(interp.begin(), interp.end(), prices[0]);
            return interp;
        }
        for (int i = 0; i < 10; i++) {
            double pos = i * (n - 1) / 9.0;
            size_t idx = static_cast<size_t>(pos);
            double frac = pos - idx;
            if(idx + 1 < n)
                interp[i] = prices[idx] * (1 - frac) + prices[idx + 1] * frac;
            else
                interp[i] = prices[idx];
        }
        return interp;
    }
}

StockDataProcessor::StockDataProcessor(int targetSamples) : targetSamples(targetSamples) {}

void StockDataProcessor::processMessage(const std::string& msg) {
    try {
        auto j = nlohmann::json::parse(msg);
        std::string ticker = j.value("ticker", "");
        double price = 0.0;
        if(j.contains("p")) {
            if(j["p"].is_string())
                price = std::stod(j["p"].get<std::string>());
            else
                price = j["p"].get<double>();
        } else {
            throw std::runtime_error("Price field 'p' missing in JSON");
        }
        if(!ticker.empty()){
            std::lock_guard<std::mutex> lock(processorMutex);
            processorStockPrices[ticker].push_back(price);
            if(processorStockPrices[ticker].size() > static_cast<size_t>(targetSamples)) {
                processorStockPrices[ticker].erase(processorStockPrices[ticker].begin());
            }
        }
    } catch(const std::exception& e) {
        std::cerr << "Error processing message in StockDataProcessor: " << e.what() << std::endl;
    } catch(...) {
        std::cerr << "Unknown error processing message in StockDataProcessor." << std::endl;
    }
}

std::vector<std::string> StockDataProcessor::getSymbols() {
    std::lock_guard<std::mutex> lock(processorMutex);
    std::vector<std::string> symbols;
    for(const auto& kv : processorStockPrices) {
        symbols.push_back(kv.first);
    }
    return symbols;
}

std::vector<double> StockDataProcessor::getInterpolatedSignal(const std::string& ticker) {
    std::lock_guard<std::mutex> lock(processorMutex);
    if(processorStockPrices.find(ticker) != processorStockPrices.end()) {
        return interpolatePrices(processorStockPrices[ticker]);
    }
    return {};
}

// Added method: getLastPrice returns the last price for the given ticker.
double StockDataProcessor::getLastPrice(const std::string &ticker) {
    std::lock_guard<std::mutex> lock(processorMutex);
    if(processorStockPrices.find(ticker) != processorStockPrices.end() &&
       !processorStockPrices[ticker].empty()) {
        return processorStockPrices[ticker].back();
    }
    return 0.0;
}
