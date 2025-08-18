#pragma once
#include <module.h>
#include <dsp/stream.h>
#include <dsp/types.h>
#include <gui/gui.h>
#include <imgui.h>
#include <config.h>
#include <thread>
#include <mutex>
#include <chrono>
#include <json/json.h>

/**
 * SDR++ Plugin for PISAD Integration
 *
 * Bridges SDR++ desktop application with drone PISAD systems
 * for dual-SDR coordination while preserving safety authority.
 *
 * PRD References:
 * - FR1: Enhanced dual SDR interface coordination
 * - FR9: Enhanced ground station telemetry streaming
 * - FR14: Enhanced SDR++ desktop operator interface
 * - NFR1: TCP communication reliability requirements
 */

class PisadBridgeModule : public ModuleManager::Instance {
public:
    PisadBridgeModule(std::string name);
    ~PisadBridgeModule();

    void postInit() override;
    void enable() override;
    void disable() override;
    bool isEnabled() override;

    // [8a] Enhanced TCP client connection interface
    void setConnectionSettings(const std::string& host, int port);
    std::string getHost() const { return pisad_host; }
    int getPort() const { return pisad_port; }
    bool isConnected() const { return connected; }
    
    // [8b] JSON message serialization for outbound commands  
    std::string serializeFrequencyControl(double frequency, int sequence);
    
    // [8c] JSON message deserialization for inbound data
    bool deserializeRSSIData(const std::string& json_data);
    float getCurrentRSSI() const { return current_rssi; }
    int getLastSequence() const { return last_sequence; }
    
    // [8d] Exponential backoff algorithm interface
    void simulateConnectionLoss();
    int getReconnectInterval() const { return current_reconnect_interval; }
    
    // [8e] Connection status monitoring and error reporting
    void setConnectionError(const std::string& error);
    std::string getConnectionStatus() const { return connection_status; }
    bool hasConnectionError() const { return has_connection_error; }
    
    // [8f] GUI integration and configuration display
    std::string getGUIConnectionStatus() const;

private:
    void worker();
    void connectToPISAD();
    void disconnectFromPISAD();
    void sendFrequencyUpdate(double freq);
    void handlePISADMessage(const Json::Value& message);
    void updateGUI();
    
    // [8d] Enhanced reconnection logic with exponential backoff
    void handleReconnection();
    void resetReconnectInterval();
    void increaseReconnectInterval();

    std::string name;
    bool enabled = false;
    bool connected = false;

    // TCP Communication
    int socket_fd = -1;
    std::string pisad_host = "localhost";
    int pisad_port = 8081;

    // Worker thread for TCP communication
    std::thread worker_thread;
    std::mutex data_mutex;
    bool worker_running = false;

    // RSSI data from PISAD
    float current_rssi = -999.0f;
    bool rssi_valid = false;
    int last_sequence = 0;

    // Connection status and error handling
    std::string connection_status = "Disconnected";
    bool has_connection_error = false;
    std::string connection_error_message;

    // [8d] Exponential backoff reconnection
    int current_reconnect_interval = 1000; // Start with 1 second
    int max_reconnect_interval = 60000;    // Max 60 seconds
    std::chrono::steady_clock::time_point last_reconnect_attempt;

    // [8e] Enhanced connection monitoring
    std::chrono::steady_clock::time_point connection_start_time;
    int connection_attempts = 0;
    bool socket_configured = false;

    // GUI state
    bool show_config = false;
    char host_buffer[256] = "localhost";
    int port_buffer = 8081;
};
