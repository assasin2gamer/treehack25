// File: websocket_client.h
#ifndef WEBSOCKET_CLIENT_H
#define WEBSOCKET_CLIENT_H

#include <string>
#include <functional>
#include <memory>
#include <thread>

class WebsocketClient {
public:
    using MessageHandler = std::function<void(const std::string&)>;
    WebsocketClient(const std::string& uri);
    ~WebsocketClient();
    void setMessageHandler(MessageHandler handler);
    void start();
    void stop();
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
    std::string m_uri;
    bool m_running;
    std::thread m_thread;
    MessageHandler m_handler;
};

#endif // WEBSOCKET_CLIENT_H
