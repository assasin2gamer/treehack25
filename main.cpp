// File: main.cu
#include "websocket_client.h"
#include "stock_data_processor.h"
#include "signal_processing.h"
#include "output_websocket.h"
#include "monte_carlo_simulation.h"

#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <string>
#include <atomic>
#include <mutex>
#include <cmath>
#include <sstream>
#include <future>
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <queue>
#include <condition_variable>
#include <random>
#include <nlohmann/json.hpp>

// ------------------------
// Global Variables
// ------------------------
std::mutex simDataMutex;
nlohmann::json globalSimData;
std::atomic<bool> running{true};
std::mutex icaCallMutex;
std::mutex printMutex;
std::atomic<int> outgoingRequestCount(0);

// ------------------------
// Main Function
// ------------------------
int main() {
    int targetSamples = 1024;
    StockDataProcessor dataProcessor(targetSamples);
    std::string uri = "ws://localhost:8766";
    WebsocketClient wsClient(uri);
    wsClient.setMessageHandler([&dataProcessor](const std::string &msg) {
        try {
            dataProcessor.processMessage(msg);
        } catch (const std::exception &e) {
            std::lock_guard<std::mutex> lock(printMutex);
            std::cerr << "Exception in message processing: " << e.what() << std::endl;
        }
    });
    wsClient.start();
try {
    uint16_t simPort = 9898;

    initSimulationWebsocketServer(simPort);
} catch (const std::exception &e) {
    uint16_t simPort = 9897;

    initSimulationWebsocketServer(simPort);
}


    std::thread simUpdateThread(simulationUpdateThread, std::ref(dataProcessor));
    std::this_thread::sleep_for(std::chrono::minutes(5));

    running = false;
    wsClient.stop();
    if (simUpdateThread.joinable())
        simUpdateThread.join();
    std::cout << "[Simulation Update] Exiting main." << std::endl;
    return 0;
}
