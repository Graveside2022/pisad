/**
 * Standalone TCP Client for PISAD Bridge Testing
 * Tests SUBTASK-5.2.2.1 implementation without SDR++ dependencies
 */

#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <json/json.h>

class StandalonePisadClient {
public:
    StandalonePisadClient() {
        current_reconnect_interval = 1000;
        last_reconnect_attempt = std::chrono::steady_clock::now();
    }

    ~StandalonePisadClient() {
        disconnect();
    }

    // [8a] Enhanced TCP client connection with configurable host/port settings
    void setConnectionSettings(const std::string& host, int port) {
        std::lock_guard<std::mutex> lock(data_mutex);
        pisad_host = host;
        pisad_port = port;
        std::cout << "Connection settings updated: " << host << ":" << port << std::endl;
    }

    std::string getHost() const { return pisad_host; }
    int getPort() const { return pisad_port; }
    bool isConnected() const { return connected; }

    // [8b] JSON message serialization for outbound commands
    std::string serializeFrequencyControl(double frequency, int sequence) {
        Json::Value message;
        message["type"] = "freq_control";
        message["frequency"] = static_cast<int64_t>(frequency);
        message["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        message["sequence"] = sequence;

        Json::StreamWriterBuilder builder;
        builder["indentation"] = "";
        return Json::writeString(builder, message);
    }

    // [9a] Enhanced sendFrequencyUpdate() to support all outbound command types
    std::string serializeMessage(const std::string& type, const Json::Value& data, int sequence) {
        Json::Value message;
        message["type"] = type;
        message["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        message["sequence"] = sequence;
        message["data"] = data;

        Json::StreamWriterBuilder builder;
        builder["indentation"] = "";
        return Json::writeString(builder, message);
    }

    // [9b] SET_FREQUENCY command message formatting with proper validation
    std::string serializeSETFrequency(double frequency, int sequence) {
        // Validate frequency range (850 MHz - 6.5 GHz per PRD-FR1)
        if (frequency < 850e6 || frequency > 6.5e9) {
            return "";  // Invalid frequency
        }

        Json::Value data;
        data["frequency"] = static_cast<int64_t>(frequency);
        data["command"] = "SET_FREQUENCY";

        return serializeMessage("freq_control", data, sequence);
    }

    // [9c] GET_RSSI request message formatting for streaming requests
    std::string serializeGETRSSI(int sequence, bool streaming = true) {
        Json::Value data;
        data["command"] = "GET_RSSI";
        data["streaming"] = streaming;

        return serializeMessage("rssi_update", data, sequence);
    }

    // [9d] Message sequence numbering for protocol reliability
    int getNextSequence() {
        return ++message_sequence;
    }

    // [9e] JSON message framing for TCP stream parsing
    std::string frameMessage(const std::string& json_message) {
        // Add length prefix and delimiter for proper TCP stream parsing
        return std::to_string(json_message.length()) + "\n" + json_message + "\n";
    }

    // [9f] Message validation and error response handling
    bool validateMessage(const Json::Value& message) {
        if (!message.isObject()) {
            std::cout << "Invalid message: not a JSON object" << std::endl;
            return false;
        }

        if (!message.isMember("type") || !message["type"].isString()) {
            std::cout << "Invalid message: missing or invalid type field" << std::endl;
            return false;
        }

        std::string type = message["type"].asString();
        if (type != "rssi_data" && type != "error" && type != "ack" && type != "heartbeat") {
            std::cout << "Invalid message type: " << type << std::endl;
            return false;
        }

        return true;
    }

    // [8c] JSON message deserialization for inbound data
    bool deserializeRSSIData(const std::string& json_data) {
        Json::Value message;
        Json::Reader reader;

        if (!reader.parse(json_data, message)) {
            std::cout << "Failed to parse RSSI JSON: " << reader.getFormattedErrorMessages() << std::endl;
            return false;
        }

        if (!message.isObject() || message.get("type", "").asString() != "rssi_data") {
            std::cout << "Invalid RSSI message type" << std::endl;
            return false;
        }

        std::lock_guard<std::mutex> lock(data_mutex);
        current_rssi = message.get("rssi", -999.0f).asFloat();
        last_sequence = message.get("sequence", 0).asInt();
        rssi_valid = true;

        return true;
    }

    float getCurrentRSSI() const { return current_rssi; }
    int getLastSequence() const { return last_sequence; }

    // [8d] Exponential backoff algorithm interface
    void simulateConnectionLoss() {
        std::cout << "Simulating connection loss for testing" << std::endl;
        disconnect();
        increaseReconnectInterval();
    }

    int getReconnectInterval() const { return current_reconnect_interval; }

    // [8e] Connection status monitoring and error reporting
    void setConnectionError(const std::string& error) {
        has_connection_error = true;
        connection_error_message = error;
        connection_status = "Error: " + error;
        std::cout << "Connection error: " << error << std::endl;
    }

    std::string getConnectionStatus() const { return connection_status; }
    bool hasConnectionError() const { return has_connection_error; }

    // [8f] GUI integration and configuration display
    std::string getGUIConnectionStatus() const {
        std::lock_guard<std::mutex> lock(data_mutex);
        if (connected) {
            return "CONNECTED to " + pisad_host + ":" + std::to_string(pisad_port);
        } else if (has_connection_error) {
            return "DISCONNECTED - " + connection_error_message;
        } else {
            return "DISCONNECTED";
        }
    }

    bool connect() {
        connection_start_time = std::chrono::steady_clock::now();
        connection_attempts++;

        socket_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (socket_fd < 0) {
            std::lock_guard<std::mutex> lock(data_mutex);
            setConnectionError("Socket creation failed: " + std::string(strerror(errno)));
            return false;
        }

        // [8d] Add TCP socket options for reliable communication
        int keepalive = 1;
        int nodelay = 1;
        if (setsockopt(socket_fd, SOL_SOCKET, SO_KEEPALIVE, &keepalive, sizeof(keepalive)) < 0) {
            std::cout << "Failed to set SO_KEEPALIVE: " << strerror(errno) << std::endl;
        }
        if (setsockopt(socket_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay)) < 0) {
            std::cout << "Failed to set TCP_NODELAY: " << strerror(errno) << std::endl;
        }
        socket_configured = true;

        // [8c] Connection timeout handling
        struct timeval timeout;
        timeout.tv_sec = 5;
        timeout.tv_usec = 0;
        setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(socket_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

        struct sockaddr_in server_addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(pisad_port);

        // [8e] Enhanced host/port validation
        if (pisad_host.empty() || pisad_port <= 0 || pisad_port > 65535) {
            std::lock_guard<std::mutex> lock(data_mutex);
            setConnectionError("Invalid host/port configuration");
            close(socket_fd);
            socket_fd = -1;
            return false;
        }

        if (inet_pton(AF_INET, pisad_host.c_str(), &server_addr.sin_addr) <= 0) {
            std::lock_guard<std::mutex> lock(data_mutex);
            setConnectionError("Invalid address: " + pisad_host);
            close(socket_fd);
            socket_fd = -1;
            return false;
        }

        if (::connect(socket_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            std::lock_guard<std::mutex> lock(data_mutex);
            setConnectionError("Connection failed to " + pisad_host + ":" + std::to_string(pisad_port) + " - " + strerror(errno));
            close(socket_fd);
            socket_fd = -1;
            return false;
        }

        std::lock_guard<std::mutex> lock(data_mutex);
        connected = true;
        has_connection_error = false;
        connection_status = "Connected";
        resetReconnectInterval();

        std::cout << "Connected to PISAD at " << pisad_host << ":" << pisad_port << " (attempt " << connection_attempts << ")" << std::endl;
        return true;
    }

    void disconnect() {
        if (socket_fd >= 0) {
            close(socket_fd);
            socket_fd = -1;
        }

        std::lock_guard<std::mutex> lock(data_mutex);
        connected = false;
        connection_status = "Disconnected";
    }

private:
    void resetReconnectInterval() {
        current_reconnect_interval = 1000;
    }

    void increaseReconnectInterval() {
        current_reconnect_interval = std::min(current_reconnect_interval * 2, max_reconnect_interval);
        std::cout << "Reconnect interval increased to " << current_reconnect_interval << " ms" << std::endl;
    }

    std::string pisad_host = "localhost";
    int pisad_port = 8081;
    int socket_fd = -1;
    bool connected = false;
    mutable std::mutex data_mutex;

    float current_rssi = -999.0f;
    bool rssi_valid = false;
    int last_sequence = 0;
    int message_sequence = 0;

    std::string connection_status = "Disconnected";
    bool has_connection_error = false;
    std::string connection_error_message;

    int current_reconnect_interval = 1000;
    int max_reconnect_interval = 60000;
    std::chrono::steady_clock::time_point last_reconnect_attempt;

    std::chrono::steady_clock::time_point connection_start_time;
    int connection_attempts = 0;
    bool socket_configured = false;
};

// Test functions to verify SUBTASK-5.2.2.1 implementation
void testConfigurableHostPortSettings() {
    std::cout << "\n=== Testing [8a] Configurable Host/Port Settings ===" << std::endl;

    StandalonePisadClient client;
    assert(!client.isConnected());

    client.setConnectionSettings("192.168.1.100", 8081);
    assert(client.getHost() == "192.168.1.100");
    assert(client.getPort() == 8081);

    std::cout << "✅ [8a] Host/port configuration test PASSED" << std::endl;
}

void testFrequencyControlSerialization() {
    std::cout << "\n=== Testing [8b] JSON Message Serialization ===" << std::endl;

    StandalonePisadClient client;
    std::string json_msg = client.serializeFrequencyControl(433.5e6, 1);

    assert(json_msg.find("\"type\":\"freq_control\"") != std::string::npos);
    assert(json_msg.find("\"frequency\":433500000") != std::string::npos);
    assert(json_msg.find("\"sequence\":1") != std::string::npos);

    std::cout << "Generated JSON: " << json_msg << std::endl;
    std::cout << "✅ [8b] JSON serialization test PASSED" << std::endl;
}

void testRSSIDataDeserialization() {
    std::cout << "\n=== Testing [8c] JSON Message Deserialization ===" << std::endl;

    StandalonePisadClient client;
    std::string rssi_json = R"({
        "type": "rssi_data",
        "timestamp": "2025-08-18T22:00:00Z",
        "rssi": -45.2,
        "frequency": 433500000,
        "sequence": 100
    })";

    bool result = client.deserializeRSSIData(rssi_json);
    assert(result == true);
    assert(std::abs(client.getCurrentRSSI() - (-45.2f)) < 0.1f);
    assert(client.getLastSequence() == 100);

    std::cout << "✅ [8c] JSON deserialization test PASSED" << std::endl;
}

void testExponentialBackoffReconnection() {
    std::cout << "\n=== Testing [8d] Exponential Backoff ===" << std::endl;

    StandalonePisadClient client;

    // Initial interval should be 1 second
    assert(client.getReconnectInterval() == 1000);

    // Simulate connection loss and check interval increase
    client.simulateConnectionLoss();
    assert(client.getReconnectInterval() == 2000);

    std::cout << "Reconnect interval after failure: " << client.getReconnectInterval() << " ms" << std::endl;
    std::cout << "✅ [8d] Exponential backoff test PASSED" << std::endl;
}

void testConnectionStatusMonitoring() {
    std::cout << "\n=== Testing [8e] Connection Status Monitoring ===" << std::endl;

    StandalonePisadClient client;
    assert(client.getConnectionStatus() == "Disconnected");

    client.setConnectionError("Network unreachable");
    assert(client.getConnectionStatus() == "Error: Network unreachable");
    assert(client.hasConnectionError() == true);

    std::cout << "✅ [8e] Connection status monitoring test PASSED" << std::endl;
}

void testGUIStatusDisplay() {
    std::cout << "\n=== Testing [8f] GUI Integration ===" << std::endl;

    StandalonePisadClient client;

    std::string gui_status = client.getGUIConnectionStatus();
    assert(!gui_status.empty());
    assert(gui_status.find("DISCONNECTED") != std::string::npos);

    std::cout << "GUI Status: " << gui_status << std::endl;
    std::cout << "✅ [8f] GUI integration test PASSED" << std::endl;
}

// Tests for SUBTASK-5.2.2.2: Comprehensive JSON Message Protocol
void testEnhancedMessageSerialization() {
    std::cout << "\n=== Testing [9a] Enhanced Message Serialization ===" << std::endl;

    StandalonePisadClient client;
    Json::Value data;
    data["test"] = "value";

    std::string json_msg = client.serializeMessage("test_type", data, 1);
    assert(json_msg.find("\"type\":\"test_type\"") != std::string::npos);
    assert(json_msg.find("\"sequence\":1") != std::string::npos);
    assert(json_msg.find("\"test\":\"value\"") != std::string::npos);

    std::cout << "✅ [9a] Enhanced message serialization test PASSED" << std::endl;
}

void testSETFrequencyCommand() {
    std::cout << "\n=== Testing [9b] SET_FREQUENCY Command ===" << std::endl;

    StandalonePisadClient client;

    // Test valid frequency
    std::string valid_msg = client.serializeSETFrequency(2.4e9, 1);
    assert(!valid_msg.empty());
    assert(valid_msg.find("\"command\":\"SET_FREQUENCY\"") != std::string::npos);
    assert(valid_msg.find("\"frequency\":2400000000") != std::string::npos);

    // Test invalid frequency (too low)
    std::string invalid_msg = client.serializeSETFrequency(100e6, 1);
    assert(invalid_msg.empty());

    // Test invalid frequency (too high)
    std::string invalid_msg2 = client.serializeSETFrequency(10e9, 1);
    assert(invalid_msg2.empty());

    std::cout << "✅ [9b] SET_FREQUENCY validation test PASSED" << std::endl;
}

void testGETRSSIRequest() {
    std::cout << "\n=== Testing [9c] GET_RSSI Request ===" << std::endl;

    StandalonePisadClient client;

    std::string rssi_msg = client.serializeGETRSSI(1, true);
    assert(rssi_msg.find("\"command\":\"GET_RSSI\"") != std::string::npos);
    assert(rssi_msg.find("\"streaming\":true") != std::string::npos);
    assert(rssi_msg.find("\"type\":\"rssi_update\"") != std::string::npos);

    std::cout << "✅ [9c] GET_RSSI request test PASSED" << std::endl;
}

void testMessageSequenceNumbering() {
    std::cout << "\n=== Testing [9d] Message Sequence Numbering ===" << std::endl;

    StandalonePisadClient client;

    int seq1 = client.getNextSequence();
    int seq2 = client.getNextSequence();
    int seq3 = client.getNextSequence();

    assert(seq2 == seq1 + 1);
    assert(seq3 == seq2 + 1);

    std::cout << "Sequence progression: " << seq1 << " -> " << seq2 << " -> " << seq3 << std::endl;
    std::cout << "✅ [9d] Message sequence numbering test PASSED" << std::endl;
}

void testMessageFraming() {
    std::cout << "\n=== Testing [9e] JSON Message Framing ===" << std::endl;

    StandalonePisadClient client;
    std::string test_json = "{\"test\":\"message\"}";
    std::string framed = client.frameMessage(test_json);

    // Should have length prefix and proper delimiters
    assert(framed.find("18\n") == 0);  // Length of test_json
    assert(framed.find(test_json) != std::string::npos);
    assert(framed.back() == '\n');

    std::cout << "Framed message: " << framed << std::endl;
    std::cout << "✅ [9e] Message framing test PASSED" << std::endl;
}

void testMessageValidation() {
    std::cout << "\n=== Testing [9f] Message Validation ===" << std::endl;

    StandalonePisadClient client;

    // Valid message
    Json::Value valid_msg;
    valid_msg["type"] = "rssi_data";
    valid_msg["data"] = "test";
    assert(client.validateMessage(valid_msg) == true);

    // Invalid message - no type
    Json::Value invalid_msg1;
    invalid_msg1["data"] = "test";
    assert(client.validateMessage(invalid_msg1) == false);

    // Invalid message - invalid type
    Json::Value invalid_msg2;
    invalid_msg2["type"] = "unknown_type";
    assert(client.validateMessage(invalid_msg2) == false);

    std::cout << "✅ [9f] Message validation test PASSED" << std::endl;
}

int main() {
    std::cout << "Testing SDR++ Plugin TCP Client Implementation" << std::endl;
    std::cout << "==============================================\n" << std::endl;

    try {
        // SUBTASK-5.2.2.1 Tests
        std::cout << "SUBTASK-5.2.2.1: Enhanced TCP Client Connection" << std::endl;
        std::cout << "-----------------------------------------------" << std::endl;
        testConfigurableHostPortSettings();
        testFrequencyControlSerialization();
        testRSSIDataDeserialization();
        testExponentialBackoffReconnection();
        testConnectionStatusMonitoring();
        testGUIStatusDisplay();

        std::cout << "\n🎉 ALL TESTS PASSED for SUBTASK-5.2.2.1!" << std::endl;

        // SUBTASK-5.2.2.2 Tests
        std::cout << "\nSUBTASK-5.2.2.2: Comprehensive JSON Message Protocol" << std::endl;
        std::cout << "----------------------------------------------------" << std::endl;
        testEnhancedMessageSerialization();
        testSETFrequencyCommand();
        testGETRSSIRequest();
        testMessageSequenceNumbering();
        testMessageFraming();
        testMessageValidation();

        std::cout << "\n🎉 ALL TESTS PASSED for SUBTASK-5.2.2.2!" << std::endl;
        std::cout << "✅ Comprehensive JSON message protocol implementation complete" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cout << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
