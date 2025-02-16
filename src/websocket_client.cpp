// File: websocket_client.cpp
#include "websocket_client.h"
#include <websocketpp/config/asio_client.hpp>
#include <websocketpp/client.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <algorithm>

using json = nlohmann::json;
typedef websocketpp::client<websocketpp::config::asio_client> client;

namespace {
    // Global map for storing stock prices and interpolation.
    std::unordered_map<std::string, std::vector<double>> stockPrices;
    std::mutex dataMutex;

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

    void updateAndPrintInterpolation(const std::string& ticker) {
        std::lock_guard<std::mutex> lock(dataMutex);
        const auto& prices = stockPrices[ticker];
        std::vector<double> interp = interpolatePrices(prices);
        /*
        std::cout << "Interpolated data for " << ticker << ": [";
        for (size_t i = 0; i < interp.size(); i++) {
            std::cout << interp[i];
            if(i < interp.size()-1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        */
    }
}

class WebsocketClient::Impl {
public:
    client m_client;
    websocketpp::connection_hdl m_hdl;
    bool m_connected = false;
};

WebsocketClient::WebsocketClient(const std::string& uri)
    : m_uri(uri), m_running(false), pImpl(std::make_unique<Impl>()) {}

WebsocketClient::~WebsocketClient() {
    stop();
}

void WebsocketClient::setMessageHandler(MessageHandler handler) {
    m_handler = std::move(handler);
}

void WebsocketClient::start() {
    m_running = true;
    m_thread = std::thread([this]() {
        try {
            pImpl->m_client.init_asio();

            pImpl->m_client.set_open_handler([this](websocketpp::connection_hdl hdl) {
                pImpl->m_connected = true;
                pImpl->m_hdl = hdl;
                //std::cout << "Connection opened." << std::endl;
            });

            pImpl->m_client.set_message_handler([this](websocketpp::connection_hdl, client::message_ptr msg) {
                try {
                    json j = json::parse(msg->get_payload());
                    std::string ticker = j.value("ticker", "");
                    if(ticker.empty()) return;
                    double price = 0.0;
                    try {
                        if(j["p"].is_string()) {
                            price = std::stod(j["p"].get<std::string>());
                        } else {
                            price = j["p"].get<double>();
                        }
                    } catch(const std::exception& e) {
                        std::cerr << "Error converting price for ticker " << ticker << ": " << e.what() << std::endl;
                        return;
                    }
                    {
                        std::lock_guard<std::mutex> lock(dataMutex);
                        stockPrices[ticker].push_back(price);
                    }
                    updateAndPrintInterpolation(ticker);
                    if(m_handler) {
                        m_handler(j.dump());
                    } else {
                        //std::cout << "Received: " << j.dump() << std::endl;
                    }
                } catch(const std::exception& e) {
                    std::cerr << "Error processing message: " << e.what() << std::endl;
                }
            });

            websocketpp::lib::error_code ec;
            client::connection_ptr con = pImpl->m_client.get_connection(m_uri, ec);
            if (ec) {
                std::cerr << "Could not create connection: " << ec.message() << std::endl;
                return;
            }
            pImpl->m_client.connect(con);
            pImpl->m_client.run();
        } catch (const std::exception &e) {
            std::cerr << "WebSocket exception: " << e.what() << std::endl;
        }
    });
}

void WebsocketClient::stop() {
    if (m_running) {
        m_running = false;
        try {
            if (pImpl->m_connected) {
                websocketpp::lib::error_code ec;
                pImpl->m_client.close(pImpl->m_hdl, websocketpp::close::status::going_away, "Client closing", ec);
                if (ec) {
                    std::cerr << "Close Error: " << ec.message() << std::endl;
                }
            }
            pImpl->m_client.stop();
        } catch (...) {
            std::cerr << "Exception during WebSocket stop" << std::endl;
        }
        if (m_thread.joinable()) {
            m_thread.join();
        }
    }
}
