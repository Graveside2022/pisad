#include "pisad_bridge.h"
#include <spdlog/spdlog.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <chrono>

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
            connectToPISAD();
            std::this_thread::sleep_for(std::chrono::seconds(5));
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
    socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd < 0) {
        std::lock_guard<std::mutex> lock(data_mutex);
        connection_status = "Socket creation failed";
        return;
    }

    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(pisad_port);

    if (inet_pton(AF_INET, pisad_host.c_str(), &server_addr.sin_addr) <= 0) {
        std::lock_guard<std::mutex> lock(data_mutex);
        connection_status = "Invalid address";
        close(socket_fd);
        socket_fd = -1;
        return;
    }

    if (connect(socket_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::lock_guard<std::mutex> lock(data_mutex);
        connection_status = "Connection failed";
        close(socket_fd);
        socket_fd = -1;
        return;
    }

    std::lock_guard<std::mutex> lock(data_mutex);
    connected = true;
    connection_status = "Connected";

    spdlog::info("Connected to PISAD at {}:{}", pisad_host, pisad_port);
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
