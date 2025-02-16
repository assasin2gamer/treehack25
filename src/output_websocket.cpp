#include "output_websocket.h"
#include <algorithm>
#include <iostream>
#include <atomic>
#include <thread>
#include <mutex>
#include <vector>

extern std::atomic<int> outgoingRequestCount;

// Globals
std::shared_ptr<server> gWsServer;
std::vector<websocketpp::connection_hdl> gConnections;
std::mutex gConnectionsMutex;

// Helper comparator: returns true if two connection_hdl refer to the same connection.
bool connection_hdl_equal(const websocketpp::connection_hdl &lhs, const websocketpp::connection_hdl &rhs) {
    return !lhs.owner_before(rhs) && !rhs.owner_before(lhs);
}

void initSimulationWebsocketServer(uint16_t port) {
    gWsServer = std::make_shared<server>();
    gWsServer->init_asio();

    gWsServer->set_open_handler([](websocketpp::connection_hdl hdl) {
        std::cout << "[Sim WS] Client connected." << std::endl;
        std::lock_guard<std::mutex> lock(gConnectionsMutex);
        gConnections.push_back(hdl);
    });

    gWsServer->set_close_handler([](websocketpp::connection_hdl hdl) {
        std::cout << "[Sim WS] Client disconnected." << std::endl;
        std::lock_guard<std::mutex> lock(gConnectionsMutex);
        gConnections.erase(std::remove_if(gConnections.begin(), gConnections.end(),
            [hdl](const websocketpp::connection_hdl &ptr) {
                return connection_hdl_equal(ptr, hdl);
            }),
            gConnections.end());
    });

    gWsServer->listen(port);
    gWsServer->start_accept();

    std::thread([=]() {
        gWsServer->run();
    }).detach();

    std::cout << "Simulation WebSocket server started on ws://localhost:" << port << std::endl;
}

void sendSimulationData(const nlohmann::json &data) {
    std::string message = data.dump();
    std::cout << "Sending Simulation Data: " << message << std::endl;

    std::lock_guard<std::mutex> lock(gConnectionsMutex);
    for (auto hdl : gConnections) {
        try {
            gWsServer->send(hdl, message, websocketpp::frame::opcode::text);
        } catch (const std::exception &e) {
            std::cerr << "Error sending simulation data: " << e.what() << std::endl;
        }
    }
}

void sendTradeData(const nlohmann::json &data) {
    std::string message = data.dump();
    std::cout << "Sending Trade Data: " << message << std::endl;

    std::lock_guard<std::mutex> lock(gConnectionsMutex);
    for (auto hdl : gConnections) {
        try {
            gWsServer->send(hdl, message, websocketpp::frame::opcode::text);
        } catch (const std::exception &e) {
            std::cerr << "Error sending trade data: " << e.what() << std::endl;
        }
    }
}
