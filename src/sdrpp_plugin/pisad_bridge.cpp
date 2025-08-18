#include "pisad_bridge.h"
#include <spdlog/spdlog.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <chrono>
#include <cstring>
#include <string>

SDRPP_MOD_INFO{
    /* Name:            */ "pisad_bridge",
    /* Description:     */ "PISAD Integration Bridge for dual-SDR coordination",
    /* Author:          */ "RF-Homing SAR Drone Project",
    /* Version:         */ 1, 0, 0,
    /* Max instances    */ 1
};

ConfigManager config;

MOD_EXPORT void _INIT_() {
    // Module initialization
    config.setPath(std::string(getRoot()) + "/pisad_bridge_config.json");
    config.load();
    config.enableAutoSave();
}

MOD_EXPORT ModuleManager::Instance* _CREATE_INSTANCE_(std::string name) {
    return new PisadBridgeModule(name);
}

MOD_EXPORT void _DELETE_INSTANCE_(ModuleManager::Instance* instance) {
    delete (PisadBridgeModule*)instance;
}

MOD_EXPORT void _END_() {
    config.disableAutoSave();
    config.save();
}

PisadBridgeModule::PisadBridgeModule(std::string name) : name(name) {
    // Load configuration
    if (config.conf.contains(name)) {
        if (config.conf[name].contains("host")) {
            pisad_host = config.conf[name]["host"];
            strcpy(host_buffer, pisad_host.c_str());
        }
        if (config.conf[name].contains("port")) {
            pisad_port = config.conf[name]["port"];
            port_buffer = pisad_port;
        }
    }
}

PisadBridgeModule::~PisadBridgeModule() {
    disable();
}

void PisadBridgeModule::postInit() {
    // GUI initialization will be handled in updateGUI()
}

void PisadBridgeModule::enable() {
    if (enabled) return;

    enabled = true;
    worker_running = true;
    worker_thread = std::thread(&PisadBridgeModule::worker, this);

    spdlog::info("PISAD Bridge enabled");
}

void PisadBridgeModule::disable() {
    if (!enabled) return;

    enabled = false;
    worker_running = false;

    disconnectFromPISAD();

    if (worker_thread.joinable()) {
        worker_thread.join();
    }

    spdlog::info("PISAD Bridge disabled");
}

bool PisadBridgeModule::isEnabled() {
    return enabled;
}

void PisadBridgeModule::worker() {
    while (worker_running) {
        if (!connected) {
            // [8d] Enhanced reconnection with exponential backoff
            handleReconnection();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // Check for incoming messages from PISAD
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(socket_fd, &read_fds);

        struct timeval timeout;
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;

        int activity = select(socket_fd + 1, &read_fds, NULL, NULL, &timeout);

        if (activity > 0 && FD_ISSET(socket_fd, &read_fds)) {
            char buffer[1024];
            int bytes_received = recv(socket_fd, buffer, sizeof(buffer) - 1, 0);

            if (bytes_received > 0) {
                buffer[bytes_received] = '\0';

                Json::Value message;
                Json::Reader reader;

                if (reader.parse(buffer, message)) {
                    handlePISADMessage(message);
                }
            } else if (bytes_received <= 0) {
                // Connection lost
                disconnectFromPISAD();
            }
        }
    }
}

void PisadBridgeModule::connectToPISAD() {
    // [8a] Enhanced TCP client connection with configurable host/port settings
    connection_start_time = std::chrono::steady_clock::now();
    connection_attempts++;

    socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd < 0) {
        std::lock_guard<std::mutex> lock(data_mutex);
        setConnectionError("Socket creation failed: " + std::string(strerror(errno)));
        return;
    }

    // [8d] Add TCP socket options for reliable communication
    int keepalive = 1;
    int nodelay = 1;
    if (setsockopt(socket_fd, SOL_SOCKET, SO_KEEPALIVE, &keepalive, sizeof(keepalive)) < 0) {
        spdlog::warn("Failed to set SO_KEEPALIVE: {}", strerror(errno));
    }
    if (setsockopt(socket_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay)) < 0) {
        spdlog::warn("Failed to set TCP_NODELAY: {}", strerror(errno));
    }
    socket_configured = true;

    // [8c] Connection timeout handling to prevent blocking indefinitely
    struct timeval timeout;
    timeout.tv_sec = 5;  // 5 second timeout
    timeout.tv_usec = 0;
    setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(socket_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(pisad_port);

    // [8e] Enhanced host/port validation before connection attempts
    if (pisad_host.empty() || pisad_port <= 0 || pisad_port > 65535) {
        std::lock_guard<std::mutex> lock(data_mutex);
        setConnectionError("Invalid host/port configuration");
        close(socket_fd);
        socket_fd = -1;
        return;
    }

    if (inet_pton(AF_INET, pisad_host.c_str(), &server_addr.sin_addr) <= 0) {
        std::lock_guard<std::mutex> lock(data_mutex);
        setConnectionError("Invalid address: " + pisad_host);
        close(socket_fd);
        socket_fd = -1;
        return;
    }

    if (connect(socket_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::lock_guard<std::mutex> lock(data_mutex);
        setConnectionError("Connection failed to " + pisad_host + ":" + std::to_string(pisad_port) + " - " + strerror(errno));
        close(socket_fd);
        socket_fd = -1;
        return;
    }

    std::lock_guard<std::mutex> lock(data_mutex);
    connected = true;
    has_connection_error = false;
    connection_status = "Connected";
    resetReconnectInterval();

    spdlog::info("Connected to PISAD at {}:{} (attempt {})", pisad_host, pisad_port, connection_attempts);
}

void PisadBridgeModule::disconnectFromPISAD() {
    if (socket_fd >= 0) {
        close(socket_fd);
        socket_fd = -1;
    }

    std::lock_guard<std::mutex> lock(data_mutex);
    connected = false;
    connection_status = "Disconnected";
}

void PisadBridgeModule::sendFrequencyUpdate(double freq) {
    if (!connected || socket_fd < 0) return;

    Json::Value message;
    message["type"] = "frequency_update";
    message["frequency"] = freq;
    message["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    Json::StreamWriterBuilder builder;
    std::string json_string = Json::writeString(builder, message);

    send(socket_fd, json_string.c_str(), json_string.length(), 0);
}

void PisadBridgeModule::handlePISADMessage(const Json::Value& message) {
    if (!message.isObject()) return;

    std::string msg_type = message.get("type", "").asString();

    if (msg_type == "rssi_update") {
        std::lock_guard<std::mutex> lock(data_mutex);
        current_rssi = message.get("rssi", -999.0f).asFloat();
        rssi_valid = true;
    }
}

void PisadBridgeModule::updateGUI() {
    if (ImGui::CollapsingHeader(("PISAD Bridge##" + name).c_str())) {
        // Connection status
        ImGui::Text("Status: %s", connection_status.c_str());

        if (connected) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0, 1, 0, 1), "CONNECTED");
        } else {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1, 0, 0, 1), "DISCONNECTED");
        }

        // RSSI display
        std::lock_guard<std::mutex> lock(data_mutex);
        if (rssi_valid) {
            ImGui::Text("RSSI: %.1f dB", current_rssi);
        } else {
            ImGui::Text("RSSI: No data");
        }

        // Configuration section
        if (ImGui::Button("Configuration")) {
            show_config = !show_config;
        }

        if (show_config) {
            ImGui::InputText("Host", host_buffer, sizeof(host_buffer));
            ImGui::InputInt("Port", &port_buffer);

            if (ImGui::Button("Apply")) {
                pisad_host = std::string(host_buffer);
                pisad_port = port_buffer;

                config.conf[name]["host"] = pisad_host;
                config.conf[name]["port"] = pisad_port;

                // Reconnect with new settings
                if (connected) {
                    disconnectFromPISAD();
                }
            }
        }

        // Manual connection control
        if (!connected) {
            if (ImGui::Button("Connect")) {
                connectToPISAD();
            }
        } else {
            if (ImGui::Button("Disconnect")) {
                disconnectFromPISAD();
            }
        }
    }
}

// [8a] Enhanced TCP client connection with configurable host/port settings
void PisadBridgeModule::setConnectionSettings(const std::string& host, int port) {
    std::lock_guard<std::mutex> lock(data_mutex);
    pisad_host = host;
    pisad_port = port;
    strcpy(host_buffer, host.c_str());
    port_buffer = port;
    
    // Save to configuration
    config.conf[name]["host"] = pisad_host;
    config.conf[name]["port"] = pisad_port;
    
    spdlog::info("Connection settings updated: {}:{}", host, port);
}

// [8b] JSON message serialization for outbound frequency control commands
std::string PisadBridgeModule::serializeFrequencyControl(double frequency, int sequence) {
    Json::Value message;
    message["type"] = "freq_control";
    message["frequency"] = static_cast<int64_t>(frequency);
    message["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    message["sequence"] = sequence;
    
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "";  // Compact format
    return Json::writeString(builder, message);
}

// [8c] JSON message deserialization for inbound RSSI data streaming
bool PisadBridgeModule::deserializeRSSIData(const std::string& json_data) {
    Json::Value message;
    Json::Reader reader;
    
    if (!reader.parse(json_data, message)) {
        spdlog::warn("Failed to parse RSSI JSON: {}", reader.getFormattedErrorMessages());
        return false;
    }
    
    if (!message.isObject() || message.get("type", "").asString() != "rssi_data") {
        spdlog::warn("Invalid RSSI message type");
        return false;
    }
    
    std::lock_guard<std::mutex> lock(data_mutex);
    current_rssi = message.get("rssi", -999.0f).asFloat();
    last_sequence = message.get("sequence", 0).asInt();
    rssi_valid = true;
    
    return true;
}

// [8d] Automatic reconnection logic with exponential backoff algorithm
void PisadBridgeModule::simulateConnectionLoss() {
    spdlog::info("Simulating connection loss for testing");
    disconnectFromPISAD();
    increaseReconnectInterval();
}

void PisadBridgeModule::handleReconnection() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_reconnect_attempt).count();
    
    if (elapsed >= current_reconnect_interval) {
        last_reconnect_attempt = now;
        connectToPISAD();
        
        if (!connected) {
            increaseReconnectInterval();
        }
    }
}

void PisadBridgeModule::resetReconnectInterval() {
    current_reconnect_interval = 1000; // Reset to 1 second
}

void PisadBridgeModule::increaseReconnectInterval() {
    current_reconnect_interval = std::min(current_reconnect_interval * 2, max_reconnect_interval);
    spdlog::info("Reconnect interval increased to {} ms", current_reconnect_interval);
}

// [8e] Connection status monitoring and error reporting with GUI integration
void PisadBridgeModule::setConnectionError(const std::string& error) {
    has_connection_error = true;
    connection_error_message = error;
    connection_status = "Error: " + error;
    spdlog::error("Connection error: {}", error);
}

// [8f] GUI integration with SDR++ plugin framework and configuration display
std::string PisadBridgeModule::getGUIConnectionStatus() const {
    std::lock_guard<std::mutex> lock(data_mutex);
    if (connected) {
        return "CONNECTED to " + pisad_host + ":" + std::to_string(pisad_port);
    } else if (has_connection_error) {
        return "DISCONNECTED - " + connection_error_message;
    } else {
        return "DISCONNECTED";
    }
}
