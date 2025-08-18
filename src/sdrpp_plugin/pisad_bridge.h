#pragma once
#include <module.h>
#include <dsp/stream.h>
#include <dsp/types.h>
#include <gui/gui.h>
#include <imgui.h>
#include <config.h>
#include <thread>
#include <mutex>
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

private:
    void worker();
    void connectToPISAD();
    void disconnectFromPISAD();
    void sendFrequencyUpdate(double freq);
    void handlePISADMessage(const Json::Value& message);
    void updateGUI();

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

    // Connection status
    std::string connection_status = "Disconnected";

    // GUI state
    bool show_config = false;
    char host_buffer[256] = "localhost";
    int port_buffer = 8081;
};
