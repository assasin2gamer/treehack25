// File: output_websocket.h
#pragma once
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <nlohmann/json.hpp>
#include <mutex>
#include <vector>
#include <memory>
#include "stock_data_processor.h"
#include <atomic>
#include <iostream>

typedef websocketpp::server<websocketpp::config::asio> server;

extern std::shared_ptr<server> gWsServer;
extern std::vector<websocketpp::connection_hdl> gConnections;
extern std::mutex gConnectionsMutex;

void initSimulationWebsocketServer(uint16_t port);
void sendSimulationData(const nlohmann::json &data);
void sendTradeData(const nlohmann::json &data);

extern std::mutex simDataMutex;
extern nlohmann::json globalSimData;
extern std::mutex printMutex;
extern std::mutex icaCallMutex;
extern std::atomic<bool> running;

void simulationUpdateThread(StockDataProcessor &processor);
