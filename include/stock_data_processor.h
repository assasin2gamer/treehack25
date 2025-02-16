#ifndef STOCK_DATA_PROCESSOR_H
#define STOCK_DATA_PROCESSOR_H

#include <string>
#include <vector>

class StockDataProcessor {
public:
    StockDataProcessor(int targetSamples);
    void processMessage(const std::string& msg);
    std::vector<std::string> getSymbols();
    // Returns the interpolated signal for a given ticker.
    std::vector<double> getInterpolatedSignal(const std::string& ticker);
    // Returns the last price for the given ticker.
    double getLastPrice(const std::string &ticker);
private:
    int targetSamples;
};

#endif // STOCK_DATA_PROCESSOR_H
