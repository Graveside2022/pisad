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

        // Send heartbeat if needed
        auto now = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lock(data_mutex);
            if (heartbeat_enabled &&
                std::chrono::duration_cast<std::chrono::milliseconds>(now - last_heartbeat_sent).count() >= heartbeat_interval_ms) {

                Json::Value heartbeat;
                heartbeat["type"] = "heartbeat";
                heartbeat["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();

                Json::StreamWriterBuilder builder;
                std::string heartbeat_msg = Json::writeString(builder, heartbeat);

                if (send(socket_fd, heartbeat_msg.c_str(), heartbeat_msg.length(), 0) > 0) {
                    last_heartbeat_sent = now;
                }
            }
        }

        if (activity > 0 && FD_ISSET(socket_fd, &read_fds)) {
            char buffer[1024];
            int bytes_received = recv(socket_fd, buffer, sizeof(buffer) - 1, 0);

            if (bytes_received > 0) {
                buffer[bytes_received] = '\0';

                Json::Value message;
                Json::Reader reader;

                if (reader.parse(buffer, message)) {
                    handlePISADMessage(message);

                    // Update heartbeat time for any received message
                    std::lock_guard<std::mutex> lock(data_mutex);
                    last_heartbeat_time = now;
                }
            } else if (bytes_received <= 0) {
                // Connection lost
                std::lock_guard<std::mutex> lock(data_mutex);
                reconnection_state = ReconnectionState::RECONNECTING;
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

    // Initialize heartbeat monitoring
    last_heartbeat_time = std::chrono::steady_clock::now();
    last_heartbeat_sent = last_heartbeat_time;

    // Reset graceful degradation
    graceful_degradation_mode = false;
    blocking_operations = false;
    responsive_state = true;

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

    if (msg_type == "heartbeat") {
        // Handle heartbeat acknowledgment
        std::lock_guard<std::mutex> lock(data_mutex);
        last_heartbeat_time = std::chrono::steady_clock::now();
        spdlog::debug("Heartbeat received from PISAD server");
    } else if (msg_type == "rssi_update") {
        // [10b] Enhanced RSSI data validation and range checking
        if (!validateRSSIData(message)) {
            spdlog::warn("Invalid RSSI data received, rejecting message");
            return;
        }

        std::lock_guard<std::mutex> lock(data_mutex);

        // Handle both flat and nested message formats for compatibility
        float new_rssi;
        if (message.isMember("data") && message["data"].isObject()) {
            // Nested format from actual PISAD bridge service
            new_rssi = message["data"].get("rssi", -999.0f).asFloat();
        } else {
            // Flat format for direct compatibility
            new_rssi = message.get("rssi", -999.0f).asFloat();
        }

        current_rssi = new_rssi;
        rssi_valid = true;
        last_sequence = message.get("sequence", 0).asInt();

        // [10c] Add to RSSI history buffer for trend analysis
        addRSSIToHistory(new_rssi, std::chrono::steady_clock::now());

        // [10f] Log RSSI data if logging is enabled
        if (rssi_logging_enabled) {
            logRSSIData(message);
        }

        spdlog::debug("RSSI updated: {} dBm (sequence: {})", new_rssi, last_sequence);
    }
}

// [10a] Enhanced RSSI stream processing implementation
void PisadBridgeModule::handleRSSIMessage(const Json::Value& message) {
    // This method provides the interface for tests - delegates to handlePISADMessage
    handlePISADMessage(message);
}

// [10b] RSSI data validation and range checking implementation
bool PisadBridgeModule::validateRSSIData(const Json::Value& message) {
    if (!message.isObject()) {
        return false;
    }

    // Check required fields exist
    if (!message.isMember("type")) {
        return false;
    }

    // Handle both flat and nested message formats
    float rssi;
    double frequency = 0.0;

    if (message.isMember("data") && message["data"].isObject()) {
        // Nested format from actual PISAD bridge service
        const Json::Value& data = message["data"];
        if (!data.isMember("rssi")) {
            return false;
        }
        rssi = data.get("rssi", -999.0f).asFloat();
        frequency = data.get("frequency", 0.0).asDouble();
    } else {
        // Flat format for direct compatibility
        if (!message.isMember("rssi")) {
            return false;
        }
        rssi = message.get("rssi", -999.0f).asFloat();
        frequency = message.get("frequency", 0.0).asDouble();
    }

    // Validate RSSI range (-120 to 0 dBm typical for RF applications)
    if (rssi < -120.0f || rssi > 0.0f) {
        return false;
    }

    // Validate frequency if present (PRD-FR1: 850 MHz - 6.5 GHz)
    if (frequency > 0.0 && (frequency < 850e6 || frequency > 6.5e9)) {
        return false;
    }

    return true;
}

// [10c] RSSI history buffer implementation
size_t PisadBridgeModule::getRSSIHistorySize() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));
    return rssi_history.size();
}

void PisadBridgeModule::addRSSIToHistory(float rssi, std::chrono::steady_clock::time_point timestamp) {
    std::lock_guard<std::mutex> lock(data_mutex);

    // Add new entry
    rssi_history.push_back({rssi, timestamp});

    // Maintain maximum history size
    if (rssi_history.size() > MAX_RSSI_HISTORY) {
        rssi_history.erase(rssi_history.begin());
    }
}

float PisadBridgeModule::getLatestRSSI() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));
    if (rssi_history.empty()) {
        return -999.0f;
    }
    return rssi_history.back().rssi;
}

float PisadBridgeModule::calculateRSSITrend() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));

    if (rssi_history.size() < 3) {
        return 0.0f; // Not enough data for trend calculation
    }

    // Simple linear trend calculation using first and last samples
    float first_rssi = rssi_history.front().rssi;
    float last_rssi = rssi_history.back().rssi;

    return last_rssi - first_rssi; // Positive = improving signal
}

// [10d] Signal quality indicators implementation
std::string PisadBridgeModule::getSignalQuality(float rssi) const {
    if (rssi >= -30.0f) return "Excellent";
    if (rssi >= -50.0f) return "Good";
    if (rssi >= -70.0f) return "Fair";
    if (rssi >= -90.0f) return "Poor";
    return "Very Poor";
}

bool PisadBridgeModule::hasGoodSignalQuality() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));
    return current_rssi >= -70.0f; // Good or better signal quality threshold
}

void PisadBridgeModule::updateRSSI(float rssi) {
    std::lock_guard<std::mutex> lock(data_mutex);
    current_rssi = rssi;
    rssi_valid = true;

    // Add to history buffer
    addRSSIToHistory(rssi, std::chrono::steady_clock::now());
}

// [10f] RSSI data logging implementation
void PisadBridgeModule::enableRSSILogging(const std::string& log_path) {
    std::lock_guard<std::mutex> lock(data_mutex);

    if (rssi_log_file.is_open()) {
        rssi_log_file.close();
    }

    rssi_log_path = log_path;
    rssi_log_file.open(log_path, std::ios::out | std::ios::app);
    rssi_logging_enabled = rssi_log_file.is_open();

    if (rssi_logging_enabled) {
        rssi_log_file << "# RSSI Log Started: " << std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count() << std::endl;
        rssi_log_file.flush();
        spdlog::info("RSSI logging enabled: {}", log_path);
    } else {
        spdlog::error("Failed to open RSSI log file: {}", log_path);
    }
}

void PisadBridgeModule::disableRSSILogging() {
    std::lock_guard<std::mutex> lock(data_mutex);

    if (rssi_log_file.is_open()) {
        rssi_log_file << "# RSSI Log Ended: " << std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count() << std::endl;
        rssi_log_file.close();
    }

    rssi_logging_enabled = false;
    spdlog::info("RSSI logging disabled");
}

bool PisadBridgeModule::isRSSILoggingEnabled() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));
    return rssi_logging_enabled;
}

void PisadBridgeModule::logRSSIData(const Json::Value& rssi_data) {
    if (!rssi_logging_enabled) {
        return;
    }

    // Check if file is open, attempt to open if not
    if (!rssi_log_file.is_open()) {
        std::string log_path = "/tmp/pisad_rssi.log";
        rssi_log_file.open(log_path, std::ios::app);
        if (!rssi_log_file.is_open()) {
            spdlog::error("Failed to open RSSI log file: {}", log_path);
            rssi_logging_enabled = false;  // Disable logging to prevent repeated attempts
            return;
        }
    }

    // Create log entry with timestamp, RSSI, frequency, and other metadata
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    float rssi = rssi_data.get("rssi", -999.0f).asFloat();
    double frequency = rssi_data.get("frequency", 0.0).asDouble();
    int sequence = rssi_data.get("sequence", 0).asInt();

    // Write data and check for errors
    rssi_log_file << timestamp << "," << rssi << "," << frequency << "," << sequence << std::endl;
    
    if (rssi_log_file.fail()) {
        spdlog::error("Failed to write RSSI data (timestamp: {}, sequence: {})", timestamp, sequence);
        // Attempt recovery: close and reopen file
        rssi_log_file.close();
        rssi_log_file.clear();  // Clear error flags
        std::string log_path = "/tmp/pisad_rssi.log";
        rssi_log_file.open(log_path, std::ios::app);
        if (!rssi_log_file.is_open()) {
            spdlog::error("Failed to recover RSSI log file, disabling logging");
            rssi_logging_enabled = false;
        }
        return;
    }

    // Flush and check for errors
    rssi_log_file.flush();
    if (!rssi_log_file.good()) {
        spdlog::error("Failed to flush RSSI log data (timestamp: {}, sequence: {})", timestamp, sequence);
        // Attempt recovery: close and reopen file
        rssi_log_file.close();
        rssi_log_file.clear();  // Clear error flags
        std::string log_path = "/tmp/pisad_rssi.log";
        rssi_log_file.open(log_path, std::ios::app);
        if (!rssi_log_file.is_open()) {
            spdlog::error("Failed to recover RSSI log file after flush error, disabling logging");
            rssi_logging_enabled = false;
        }
    }
}

void PisadBridgeModule::updateGUI() {
    if (ImGui::CollapsingHeader(("PISAD Bridge##" + name).c_str())) {
        // Apply professional styling
        if (!gui_styling_applied) {
            applyProfessionalStyling();
        }

        // Enhanced connection section with health monitoring
        renderConnectionSection();

        // Enhanced RSSI section with graphing and quality indicators
        renderRSSISection();

        // Status dashboard toggle
        renderStatusDashboard();

        // Advanced settings and manual controls
        renderAdvancedSettingsSection();

        // Configuration persistence controls
        ImGui::Separator();
        if (ImGui::Button("Save Config")) {
            saveSessionConfiguration();
        }
        ImGui::SameLine();
        if (ImGui::Button("Load Config")) {
            loadSessionConfiguration();
        }

        // Legacy configuration section (simplified)
        if (ImGui::Button("Basic Config")) {
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

        // Manual connection control (enhanced)
        ImGui::Separator();
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

    std::lock_guard<std::mutex> lock(data_mutex);
    reconnection_state = ReconnectionState::RECONNECTING;
    connection_stats.failed_connections++;
    connection_stats.total_attempts++;

    increaseReconnectInterval();
}

void PisadBridgeModule::handleReconnection() {
    std::unique_lock<std::mutex> lock(data_mutex);

    // Don't auto-reconnect if manual override is active
    if (manual_override_active || !auto_reconnect_enabled) {
        return;
    }

    // Set state to reconnecting if disconnected
    if (reconnection_state == ReconnectionState::DISCONNECTED) {
        reconnection_state = ReconnectionState::RECONNECTING;
    }

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_reconnect_attempt).count();

    if (elapsed >= current_reconnect_interval) {
        last_reconnect_attempt = now;
        connection_attempt_start = now;
        connection_stats.total_attempts++;

        // Unlock for connection attempt
        lock.unlock();
        connectToPISAD();

        // Relock after connection attempt
        lock.lock();
        if (!connected) {
            increaseReconnectInterval();
            // Check if should enter graceful degradation mode
            if (connection_stats.total_attempts > 5) {
                reconnection_state = ReconnectionState::GRACEFUL_DEGRADATION;
                graceful_degradation_mode = true;
            }
        } else {
            // Successful connection
            connection_stats.successful_connections++;
            reconnection_state = ReconnectionState::DISCONNECTED;
            graceful_degradation_mode = false;

            // Calculate connection time
            auto connection_time = std::chrono::duration_cast<std::chrono::milliseconds>(now - connection_attempt_start).count();
            connection_stats.average_connection_time_ms =
                (connection_stats.average_connection_time_ms * (connection_stats.successful_connections - 1) + connection_time) /
                connection_stats.successful_connections;
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

// [11a] Reconnection state machine implementation
std::string PisadBridgeModule::getReconnectionState() const {
    std::lock_guard<std::mutex> lock(data_mutex);
    switch (reconnection_state) {
        case ReconnectionState::DISCONNECTED: return "DISCONNECTED";
        case ReconnectionState::RECONNECTING: return "RECONNECTING";
        case ReconnectionState::MANUAL_CONTROL: return "MANUAL_CONTROL";
        case ReconnectionState::GRACEFUL_DEGRADATION: return "GRACEFUL_DEGRADATION";
        default: return "UNKNOWN";
    }
}

void PisadBridgeModule::setRetryIntervals(const std::vector<int>& intervals) {
    std::lock_guard<std::mutex> lock(data_mutex);
    retry_intervals = intervals;
    current_retry_index = 0;
    spdlog::info("Retry intervals configured: {} intervals", intervals.size());
}

int PisadBridgeModule::getCurrentRetryInterval() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));
    if (current_retry_index < retry_intervals.size()) {
        return retry_intervals[current_retry_index];
    }
    return max_retry_interval;
}

int PisadBridgeModule::getConnectionAttempts() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));
    return connection_stats.total_attempts;
}

void PisadBridgeModule::resetReconnectionState() {
    std::lock_guard<std::mutex> lock(data_mutex);
    reconnection_state = ReconnectionState::DISCONNECTED;
    current_retry_index = 0;
    current_reconnect_interval = retry_intervals.empty() ? min_retry_interval : retry_intervals[0];
    spdlog::info("Reconnection state reset");
}

// [11b] Enhanced exponential backoff algorithm implementation
void PisadBridgeModule::simulateConnectionFailure() {
    std::lock_guard<std::mutex> lock(data_mutex);
    connection_stats.failed_connections++;
    connection_stats.total_attempts++;

    // Advance to next retry interval
    if (current_retry_index < retry_intervals.size() - 1) {
        current_retry_index++;
        current_reconnect_interval = retry_intervals[current_retry_index];
    } else {
        current_reconnect_interval = std::min(current_reconnect_interval * static_cast<int>(backoff_multiplier), max_retry_interval);
    }

    spdlog::warn("Connection failure simulated, next retry in {} ms", current_reconnect_interval);
}

double PisadBridgeModule::getBackoffMultiplier() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));
    return backoff_multiplier;
}

// [11c] Connection health monitoring with heartbeat implementation
bool PisadBridgeModule::isHeartbeatEnabled() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));
    return heartbeat_enabled;
}

int PisadBridgeModule::getHeartbeatInterval() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));
    return heartbeat_interval_ms;
}

std::chrono::steady_clock::time_point PisadBridgeModule::getLastHeartbeatTime() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));
    return last_heartbeat_time;
}

// [11d] Graceful degradation implementation
std::string PisadBridgeModule::getDegradationMode() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));
    return graceful_degradation_mode ? "GRACEFUL" : "NORMAL";
}

bool PisadBridgeModule::isBlocking() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));
    return blocking_operations;
}

bool PisadBridgeModule::isResponsive() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));
    return responsive_state;
}

bool PisadBridgeModule::canProcessCommands() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));
    return responsive_state && !blocking_operations;
}

// [11e] Manual reconnection override implementation
bool PisadBridgeModule::isAutoReconnectEnabled() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));
    return auto_reconnect_enabled && !manual_override_active;
}

void PisadBridgeModule::enableManualOverride() {
    std::lock_guard<std::mutex> lock(data_mutex);
    manual_override_active = true;
    auto_reconnect_enabled = false;
    reconnection_state = ReconnectionState::MANUAL_CONTROL;
    spdlog::info("Manual reconnection override enabled");
}

void PisadBridgeModule::disableManualOverride() {
    std::lock_guard<std::mutex> lock(data_mutex);
    manual_override_active = false;
    auto_reconnect_enabled = true;
    reconnection_state = connected ? ReconnectionState::DISCONNECTED : ReconnectionState::RECONNECTING;
    spdlog::info("Manual reconnection override disabled, returning to automatic mode");
}

bool PisadBridgeModule::triggerManualReconnection() {
    if (!manual_override_active) {
        return false;
    }

    spdlog::info("Manual reconnection triggered");
    connectToPISAD();
    return true; // Return success for manual trigger
}

// [11f] Connection statistics tracking implementation
PisadBridgeModule::ConnectionStatistics PisadBridgeModule::getConnectionStatistics() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));
    return connection_stats;
}

void PisadBridgeModule::resetConnectionStatistics() {
    std::lock_guard<std::mutex> lock(data_mutex);
    connection_stats = ConnectionStatistics{};
    spdlog::info("Connection statistics reset");
}

// [12a] Enhanced connection status display implementation
std::string PisadBridgeModule::getDetailedConnectionStatus() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));

    if (connected) {
        return "CONNECTED to " + pisad_host + ":" + std::to_string(pisad_port) +
               " (latency: " + std::to_string(connection_latency_ms) + "ms)";
    } else if (has_connection_error) {
        std::string status = "DISCONNECTED - " + connection_error_message;
        if (last_network_error_code > 0) {
            status += " [" + last_network_error_name + " (" + std::to_string(last_network_error_code) + ")]";
        }
        return status;
    } else {
        return "DISCONNECTED";
    }
}

// [12b] Network-level error detection implementation
void PisadBridgeModule::simulateNetworkError(const std::string& error_name, int error_code) {
    std::lock_guard<std::mutex> lock(data_mutex);
    last_network_error_name = error_name;
    last_network_error_code = error_code;

    // Log the error event
    if (connection_event_logging_enabled) {
        logConnectionEvent("NETWORK_ERROR", error_name + " (" + std::to_string(error_code) + ")");
    }

    spdlog::warn("Network error simulated: {} ({})", error_name, error_code);
}

int PisadBridgeModule::getLastNetworkErrorCode() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));
    return last_network_error_code;
}

std::string PisadBridgeModule::getLastNetworkErrorName() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));
    return last_network_error_name;
}

// [12c] Connection quality metrics implementation
int PisadBridgeModule::getConnectionLatencyMs() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));
    return connection_latency_ms;
}

float PisadBridgeModule::getPacketLossPercentage() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));
    return packet_loss_percentage;
}

int PisadBridgeModule::getThroughputBytesPerSec() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));
    return throughput_bytes_per_sec;
}

void PisadBridgeModule::updateConnectionLatency(int latency_ms) {
    std::lock_guard<std::mutex> lock(data_mutex);
    connection_latency_ms = latency_ms;

    // Log latency update
    if (connection_event_logging_enabled) {
        logConnectionEvent("LATENCY_UPDATE", "Latency updated to " + std::to_string(latency_ms) + "ms");
    }
}

void PisadBridgeModule::updatePacketLoss(float loss_percentage) {
    std::lock_guard<std::mutex> lock(data_mutex);
    packet_loss_percentage = loss_percentage;

    // Log packet loss update
    if (connection_event_logging_enabled) {
        logConnectionEvent("PACKET_LOSS_UPDATE", "Packet loss updated to " + std::to_string(loss_percentage) + "%");
    }
}

void PisadBridgeModule::updateThroughput(int bytes_per_sec) {
    std::lock_guard<std::mutex> lock(data_mutex);
    throughput_bytes_per_sec = bytes_per_sec;

    // Log throughput update
    if (connection_event_logging_enabled) {
        logConnectionEvent("THROUGHPUT_UPDATE", "Throughput updated to " + std::to_string(bytes_per_sec) + " bytes/sec");
    }
}

// [12d] Visual connection status indicators implementation
std::string PisadBridgeModule::getConnectionStatusColor() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));

    if (!connected) {
        return "red";
    } else if (connection_latency_ms <= 50 && packet_loss_percentage <= 1.0f) {
        return "green";  // Excellent connection
    } else if (connection_latency_ms <= 100 && packet_loss_percentage <= 5.0f) {
        return "yellow"; // Good connection
    } else {
        return "orange"; // Poor but usable connection
    }
}

int PisadBridgeModule::getConnectionHealthMeter() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));

    if (!connected) {
        return 0;
    }

    int health = 100;

    // Reduce health based on latency (PRD-NFR2: <100ms target)
    if (connection_latency_ms > 50) {
        health -= (connection_latency_ms - 50) / 2;  // Reduce by 1 per 2ms over 50ms
    }

    // Reduce health based on packet loss
    health -= static_cast<int>(packet_loss_percentage * 10);  // Reduce by 10 per 1% loss

    return std::max(0, std::min(100, health));
}

void PisadBridgeModule::setConnected(bool connected_state) {
    std::lock_guard<std::mutex> lock(data_mutex);
    connected = connected_state;
    connection_status = connected_state ? "Connected" : "Disconnected";
}

// [12e] Error notification system implementation
bool PisadBridgeModule::hasPendingNotifications() const {
    std::lock_guard<std::mutex> lock(notification_mutex);
    return !pending_notifications.empty();
}

std::string PisadBridgeModule::getNextNotification() {
    std::lock_guard<std::mutex> lock(notification_mutex);

    if (pending_notifications.empty()) {
        return "";
    }

    std::string notification = pending_notifications.front();
    pending_notifications.erase(pending_notifications.begin());
    return notification;
}

void PisadBridgeModule::triggerCriticalError(const std::string& error_message) {
    std::lock_guard<std::mutex> lock(notification_mutex);

    std::string notification = "[CRITICAL] " + error_message + " (timestamp: " +
                               std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::system_clock::now().time_since_epoch()).count()) + ")";

    pending_notifications.push_back(notification);

    // Also log the critical error
    if (connection_event_logging_enabled) {
        logConnectionEvent("CRITICAL_ERROR", error_message);
    }

    spdlog::error("Critical error triggered: {}", error_message);
}

// [12f] Connection event logging implementation
void PisadBridgeModule::enableConnectionEventLogging(const std::string& log_path) {
    std::lock_guard<std::mutex> lock(data_mutex);

    if (connection_event_log_file.is_open()) {
        connection_event_log_file.close();
    }

    connection_event_log_path = log_path;
    connection_event_log_file.open(log_path, std::ios::out | std::ios::app);
    connection_event_logging_enabled = connection_event_log_file.is_open();

    if (connection_event_logging_enabled) {
        connection_event_log_file << "# Connection Event Log Started: " <<
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count() << std::endl;
        connection_event_log_file.flush();
        spdlog::info("Connection event logging enabled: {}", log_path);
    } else {
        spdlog::error("Failed to open connection event log file: {}", log_path);
    }
}

bool PisadBridgeModule::isConnectionEventLoggingEnabled() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));
    return connection_event_logging_enabled;
}

void PisadBridgeModule::logConnectionEvent(const std::string& event_type, const std::string& message) {
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    std::string event_entry = std::to_string(timestamp) + "," + event_type + "," + message;

    // Add to history buffer
    {
        std::lock_guard<std::mutex> lock(data_mutex);
        connection_event_history.push_back(event_entry);

        // Maintain maximum history size
        if (connection_event_history.size() > MAX_EVENT_HISTORY) {
            connection_event_history.erase(connection_event_history.begin());
        }
    }

    // Write to log file if enabled
    if (connection_event_logging_enabled && connection_event_log_file.is_open()) {
        connection_event_log_file << event_entry << std::endl;
        connection_event_log_file.flush();
    }
}

std::vector<std::string> PisadBridgeModule::getConnectionEventHistory() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));
    return connection_event_history;
}

// [13a] Enhanced GUI with professional layout implementation
void PisadBridgeModule::renderConnectionSection() {
    if (!gui_styling_applied) {
        applyProfessionalStyling();
    }

    if (ImGui::CollapsingHeader("Connection Status", ImGuiTreeNodeFlags_DefaultOpen)) {
        // Status indicator with color coding
        std::string color = getConnectionStatusColor();
        ImVec4 status_color;
        if (color == "green") status_color = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
        else if (color == "yellow") status_color = ImVec4(1.0f, 1.0f, 0.0f, 1.0f);
        else if (color == "orange") status_color = ImVec4(1.0f, 0.5f, 0.0f, 1.0f);
        else status_color = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);

        ImGui::TextColored(status_color, "%s", getDetailedConnectionStatus().c_str());

        // Connection health meter
        int health = getConnectionHealthMeter();
        ImGui::Text("Health: ");
        ImGui::SameLine();
        ImGui::ProgressBar(health / 100.0f, ImVec2(200, 0), (std::to_string(health) + "%").c_str());
    }
}

void PisadBridgeModule::renderRSSISection() {
    if (ImGui::CollapsingHeader("Signal Information", ImGuiTreeNodeFlags_DefaultOpen)) {
        std::lock_guard<std::mutex> lock(data_mutex);

        if (rssi_valid) {
            // RSSI meter
            renderRSSIMeter();

            // Signal quality indicator
            std::string quality = getSignalQuality(current_rssi);
            ImGui::Text("Signal Quality: %s", quality.c_str());

            // RSSI graph toggle
            ImGui::Checkbox("Show RSSI Graph", &show_rssi_graph);
            if (show_rssi_graph) {
                renderRSSIGraph();
            }
        } else {
            ImGui::Text("RSSI: No data available");
        }
    }
}

void PisadBridgeModule::renderAdvancedSettingsSection() {
    if (ImGui::Button("Advanced Settings")) {
        show_advanced_settings = !show_advanced_settings;
    }

    if (show_advanced_settings) {
        if (ImGui::BeginChild("AdvancedSettings", ImVec2(0, 200), true)) {
            renderAdvancedConnectionSettings();
            ImGui::Separator();
            renderManualControlButtons();
        }
        ImGui::EndChild();
    }
}

void PisadBridgeModule::applyProfessionalStyling() {
    // Apply professional ImGui styling for operational clarity
    ImGuiStyle& style = ImGui::GetStyle();

    // Color scheme for emergency/operational use
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.1f, 0.1f, 0.15f, 0.95f);
    style.Colors[ImGuiCol_Header] = ImVec4(0.2f, 0.4f, 0.6f, 0.8f);
    style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.3f, 0.5f, 0.7f, 0.8f);
    style.Colors[ImGuiCol_Button] = ImVec4(0.2f, 0.3f, 0.5f, 0.8f);
    style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.3f, 0.4f, 0.6f, 0.8f);

    // Spacing for operational clarity
    style.FramePadding = ImVec2(8, 4);
    style.ItemSpacing = ImVec2(8, 6);
    style.WindowRounding = 4.0f;
    style.FrameRounding = 3.0f;

    gui_styling_applied = true;
}

// [13b] Real-time RSSI graph/meter display implementation
void PisadBridgeModule::renderRSSIGraph() {
    updateRSSIPlotData();

    auto times = getRSSIPlotTimes();
    auto values = getRSSIPlotValues();

    if (times.size() > 1 && values.size() > 1) {
        // Note: This would use ImPlot in actual SDR++ environment
        // For now, we'll use a simple text representation
        ImGui::Text("RSSI Trend Graph:");
        ImGui::Text("Points: %zu, Latest: %.1f dBm", values.size(), values.back());

        // Simple ASCII-style trend indicator
        if (values.size() >= 2) {
            float trend = values.back() - values[values.size()-2];
            if (trend > 1.0f) ImGui::TextColored(ImVec4(0,1,0,1), "↗ Improving");
            else if (trend < -1.0f) ImGui::TextColored(ImVec4(1,0,0,1), "↘ Degrading");
            else ImGui::Text("→ Stable");
        }
    }
}

void PisadBridgeModule::renderRSSIMeter() {
    std::lock_guard<std::mutex> lock(data_mutex);

    // RSSI meter as progress bar
    float rssi_normalized = (current_rssi - rssi_meter_min) / (rssi_meter_max - rssi_meter_min);
    rssi_normalized = std::max(0.0f, std::min(1.0f, rssi_normalized));

    // Color based on signal strength
    ImVec4 meter_color;
    if (current_rssi >= -50.0f) meter_color = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);  // Green
    else if (current_rssi >= -70.0f) meter_color = ImVec4(1.0f, 1.0f, 0.0f, 1.0f);  // Yellow
    else meter_color = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);  // Red

    ImGui::Text("RSSI: %.1f dBm", current_rssi);
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, meter_color);
    ImGui::ProgressBar(rssi_normalized, ImVec2(300, 20), "");
    ImGui::PopStyleColor();
}

void PisadBridgeModule::updateRSSIPlotData() {
    std::lock_guard<std::mutex> lock(data_mutex);

    if (!rssi_valid) return;

    auto now = std::chrono::steady_clock::now();
    float time_sec = std::chrono::duration<float>(now - last_gui_update).count();

    rssi_plot_times.push_back(time_sec);
    rssi_plot_values.push_back(current_rssi);

    // Maintain maximum plot points
    if (rssi_plot_times.size() > MAX_PLOT_POINTS) {
        rssi_plot_times.erase(rssi_plot_times.begin());
        rssi_plot_values.erase(rssi_plot_values.begin());
    }

    last_gui_update = now;
}

std::vector<float> PisadBridgeModule::getRSSIPlotTimes() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));
    return rssi_plot_times;
}

std::vector<float> PisadBridgeModule::getRSSIPlotValues() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));
    return rssi_plot_values;
}

// [13c] Configuration persistence implementation
void PisadBridgeModule::saveSessionConfiguration() {
    std::lock_guard<std::mutex> lock(data_mutex);

    session_config["host"] = pisad_host;
    session_config["port"] = pisad_port;
    session_config["heartbeat_interval"] = heartbeat_interval_ms;
    session_config["connection_timeout"] = connection_timeout_ms;
    session_config["auto_reconnect"] = auto_reconnect_enabled;
    session_config["rssi_logging"] = rssi_logging_enabled;
    session_config["show_rssi_graph"] = show_rssi_graph;
    session_config["show_advanced_settings"] = show_advanced_settings;

    // Save to config file
    config_file_path = std::string(getRoot()) + "/pisad_session.json";
    Json::StreamWriterBuilder builder;
    std::ofstream config_file(config_file_path);
    if (config_file.is_open()) {
        config_file << Json::writeString(builder, session_config);
        config_file.close();
        spdlog::info("Session configuration saved to {}", config_file_path);
    }
}

void PisadBridgeModule::loadSessionConfiguration() {
    config_file_path = std::string(getRoot()) + "/pisad_session.json";
    std::ifstream config_file(config_file_path);

    if (config_file.is_open()) {
        Json::Reader reader;
        if (reader.parse(config_file, session_config)) {
            std::lock_guard<std::mutex> lock(data_mutex);

            if (session_config.isMember("host")) {
                pisad_host = session_config["host"].asString();
                strcpy(host_buffer, pisad_host.c_str());
            }
            if (session_config.isMember("port")) {
                pisad_port = session_config["port"].asInt();
                port_buffer = pisad_port;
            }
            if (session_config.isMember("heartbeat_interval")) {
                heartbeat_interval_ms = session_config["heartbeat_interval"].asInt();
                int ret = snprintf(heartbeat_buffer, sizeof(heartbeat_buffer), "%d", heartbeat_interval_ms);
                if (ret < 0 || ret >= sizeof(heartbeat_buffer)) {
                    spdlog::warn("Heartbeat interval value truncated in buffer");
                }
            }
            if (session_config.isMember("connection_timeout")) {
                connection_timeout_ms = session_config["connection_timeout"].asInt();
                int ret = snprintf(timeout_buffer, sizeof(timeout_buffer), "%d", connection_timeout_ms);
                if (ret < 0 || ret >= sizeof(timeout_buffer)) {
                    spdlog::warn("Connection timeout value truncated in buffer");
                }
            }
            if (session_config.isMember("show_rssi_graph")) {
                show_rssi_graph = session_config["show_rssi_graph"].asBool();
            }

            session_config_loaded = true;
            spdlog::info("Session configuration loaded from {}", config_file_path);
        }
        config_file.close();
    }
}

bool PisadBridgeModule::hasValidSessionConfig() const {
    return session_config_loaded && !session_config.empty();
}

void PisadBridgeModule::exportConfiguration(const std::string& file_path) {
    saveSessionConfiguration();  // Update current config

    Json::StreamWriterBuilder builder;
    std::ofstream export_file(file_path);
    if (export_file.is_open()) {
        export_file << Json::writeString(builder, session_config);
        export_file.close();
        spdlog::info("Configuration exported to {}", file_path);
    }
}

void PisadBridgeModule::importConfiguration(const std::string& file_path) {
    std::ifstream import_file(file_path);
    if (import_file.is_open()) {
        Json::Reader reader;
        Json::Value imported_config;
        if (reader.parse(import_file, imported_config)) {
            session_config = imported_config;
            loadSessionConfiguration();  // Apply the imported configuration
            spdlog::info("Configuration imported from {}", file_path);
        }
        import_file.close();
    }
}

// [13d] Advanced connection settings panel implementation
void PisadBridgeModule::renderAdvancedConnectionSettings() {
    if (ImGui::CollapsingHeader("Connection Settings")) {
        ImGui::InputText("Timeout (ms)", timeout_buffer, sizeof(timeout_buffer));
        if (ImGui::Button("Apply Timeout")) {
            setConnectionTimeout(std::atoi(timeout_buffer));
        }

        ImGui::InputText("Heartbeat (ms)", heartbeat_buffer, sizeof(heartbeat_buffer));
        if (ImGui::Button("Apply Heartbeat")) {
            setHeartbeatInterval(std::atoi(heartbeat_buffer));
        }

        ImGui::Checkbox("Auto Reconnect", &auto_reconnect_enabled);
        ImGui::Checkbox("Manual Override", &manual_override_active);
    }
}

void PisadBridgeModule::setConnectionTimeout(int timeout_ms) {
    std::lock_guard<std::mutex> lock(data_mutex);
    connection_timeout_ms = timeout_ms;
    logConnectionEvent("CONFIG_CHANGE", "Connection timeout set to " + std::to_string(timeout_ms) + "ms");
}

void PisadBridgeModule::setHeartbeatInterval(int interval_ms) {
    std::lock_guard<std::mutex> lock(data_mutex);
    heartbeat_interval_ms = interval_ms;
    logConnectionEvent("CONFIG_CHANGE", "Heartbeat interval set to " + std::to_string(interval_ms) + "ms");
}

void PisadBridgeModule::setRetryConfiguration(const std::vector<int>& intervals) {
    setRetryIntervals(intervals);  // Use existing method
}

// [13e] Status dashboard implementation
void PisadBridgeModule::renderStatusDashboard() {
    if (ImGui::Button("Status Dashboard")) {
        show_status_dashboard = !show_status_dashboard;
    }

    if (show_status_dashboard) {
        if (ImGui::BeginChild("StatusDashboard", ImVec2(0, 150), true)) {
            renderConnectionMetrics();
            ImGui::Separator();
            renderStatisticsPanel();
        }
        ImGui::EndChild();
    }
}

void PisadBridgeModule::renderConnectionMetrics() {
    std::lock_guard<std::mutex> lock(data_mutex);

    ImGui::Text("Connection Metrics:");
    ImGui::Text("Latency: %d ms", connection_latency_ms);
    ImGui::Text("Packet Loss: %.1f%%", packet_loss_percentage);
    ImGui::Text("Throughput: %d bytes/sec", throughput_bytes_per_sec);
    ImGui::Text("Success Rate: %.1f%%", getConnectionSuccessRate());
}

void PisadBridgeModule::renderStatisticsPanel() {
    auto stats = getConnectionStatistics();

    ImGui::Text("Statistics:");
    ImGui::Text("Total Attempts: %d", stats.total_attempts);
    ImGui::Text("Successful: %d", stats.successful_connections);
    ImGui::Text("Failed: %d", stats.failed_connections);
    ImGui::Text("Avg Connection Time: %d ms", stats.average_connection_time_ms);
}

float PisadBridgeModule::getConnectionSuccessRate() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(data_mutex));
    if (connection_stats.total_attempts == 0) return 0.0f;
    return (float)connection_stats.successful_connections / connection_stats.total_attempts * 100.0f;
}

// [13f] Manual control buttons implementation
void PisadBridgeModule::renderManualControlButtons() {
    if (ImGui::CollapsingHeader("Manual Controls")) {
        if (ImGui::Button("Force Reconnect")) {
            forceReconnection();
        }
        ImGui::SameLine();
        if (ImGui::Button("Connection Test")) {
            performConnectionTest();
        }

        ImGui::InputText("Test Message Type", test_message_buffer, sizeof(test_message_buffer));
        ImGui::InputText("Test Data", test_data_buffer, sizeof(test_data_buffer));
        if (ImGui::Button("Inject Test Message")) {
            Json::Value test_data;
            Json::Reader reader;
            if (reader.parse(test_data_buffer, test_data)) {
                injectTestMessage(test_message_buffer, test_data);
            }
        }
    }
}

void PisadBridgeModule::performConnectionTest() {
    logConnectionEvent("MANUAL_TEST", "Connection test initiated by operator");

    auto start = std::chrono::steady_clock::now();
    connectToPISAD();
    auto end = std::chrono::steady_clock::now();

    int test_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    logConnectionEvent("MANUAL_TEST", "Connection test completed in " + std::to_string(test_duration) + "ms");
}

void PisadBridgeModule::injectTestMessage(const std::string& message_type, const Json::Value& data) {
    Json::Value test_message;
    test_message["type"] = message_type;
    test_message["data"] = data;
    test_message["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    handlePISADMessage(test_message);
    logConnectionEvent("MESSAGE_INJECTION", "Test message injected: " + message_type);
}

void PisadBridgeModule::forceReconnection() {
    logConnectionEvent("MANUAL_RECONNECT", "Manual reconnection triggered by operator");
    disconnectFromPISAD();
    connectToPISAD();
}
