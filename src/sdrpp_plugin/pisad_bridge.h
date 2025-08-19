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
#include <vector>
#include <fstream>
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

    // [11a] Reconnection state machine with configurable retry intervals
    std::string getReconnectionState() const;
    void setRetryIntervals(const std::vector<int>& intervals);
    int getCurrentRetryInterval() const;
    int getConnectionAttempts() const;
    void resetReconnectionState();

    // [11b] Enhanced exponential backoff algorithm
    void simulateConnectionFailure();
    double getBackoffMultiplier() const;

    // [11c] Connection health monitoring with heartbeat messages
    bool isHeartbeatEnabled() const;
    int getHeartbeatInterval() const;
    std::chrono::steady_clock::time_point getLastHeartbeatTime() const;

    // [11d] Graceful degradation when server unavailable
    std::string getDegradationMode() const;
    bool isBlocking() const;
    bool isResponsive() const;
    bool canProcessCommands() const;

    // [11e] Manual reconnection override capability
    bool isAutoReconnectEnabled() const;
    void enableManualOverride();
    void disableManualOverride();
    bool triggerManualReconnection();

    // [11f] Connection statistics tracking for debugging
    struct ConnectionStatistics {
        int total_attempts = 0;
        int successful_connections = 0;
        int failed_connections = 0;
        int average_connection_time_ms = 0;
        std::chrono::steady_clock::time_point first_attempt_time;
        std::chrono::steady_clock::time_point last_attempt_time;
    };
    ConnectionStatistics getConnectionStatistics() const;
    void resetConnectionStatistics();

    // [12a] Enhanced connection status display with detailed error messages and visual indicators
    std::string getDetailedConnectionStatus() const;

    // [12b] Network-level error detection with specific codes
    void simulateNetworkError(const std::string& error_name, int error_code);
    int getLastNetworkErrorCode() const;
    std::string getLastNetworkErrorName() const;

    // [12c] Connection quality metrics (latency, packet loss, throughput monitoring)
    int getConnectionLatencyMs() const;
    float getPacketLossPercentage() const;
    int getThroughputBytesPerSec() const;
    void updateConnectionLatency(int latency_ms);
    void updatePacketLoss(float loss_percentage);
    void updateThroughput(int bytes_per_sec);

    // [12d] Visual connection status indicators (color-coded status, health meters)
    std::string getConnectionStatusColor() const;
    int getConnectionHealthMeter() const;  // 0-100 scale
    void setConnected(bool connected_state);  // For testing

    // [12e] Error notification system for critical connection failures
    bool hasPendingNotifications() const;
    std::string getNextNotification();
    void triggerCriticalError(const std::string& error_message);

    // [12f] Connection event logging for troubleshooting
    void enableConnectionEventLogging(const std::string& log_path);
    bool isConnectionEventLoggingEnabled() const;
    void logConnectionEvent(const std::string& event_type, const std::string& message);
    std::vector<std::string> getConnectionEventHistory() const;

    // [13a] Enhanced GUI with professional layout (ImGui styling, organized sections, responsive design)
    void renderConnectionSection();
    void renderRSSISection();
    void renderAdvancedSettingsSection();
    void applyProfessionalStyling();

    // [13b] Real-time RSSI graph/meter display (ImPlot integration, historical trends, signal quality visualization)
    void renderRSSIGraph();
    void renderRSSIMeter();
    void updateRSSIPlotData();
    std::vector<float> getRSSIPlotTimes() const;
    std::vector<float> getRSSIPlotValues() const;

    // [13c] Configuration persistence across SDR++ sessions (JSON config save/load, session restoration)
    void saveSessionConfiguration();
    void loadSessionConfiguration();
    bool hasValidSessionConfig() const;
    void exportConfiguration(const std::string& file_path);
    void importConfiguration(const std::string& file_path);

    // [13d] Advanced connection settings panel (timeout configuration, retry intervals, heartbeat settings)
    void renderAdvancedConnectionSettings();
    void setConnectionTimeout(int timeout_ms);
    void setHeartbeatInterval(int interval_ms);
    void setRetryConfiguration(const std::vector<int>& intervals);

    // [13e] Status dashboard showing connection health metrics (latency display, success rates, connection statistics)
    void renderStatusDashboard();
    void renderConnectionMetrics();
    void renderStatisticsPanel();
    float getConnectionSuccessRate() const;

    // [13f] Manual control buttons for testing (force reconnect, connection test, message injection)
    void renderManualControlButtons();
    void performConnectionTest();
    void injectTestMessage(const std::string& message_type, const Json::Value& data);
    void forceReconnection();

    // [10a] Enhanced RSSI stream processing
    void handleRSSIMessage(const Json::Value& message);
    bool isRSSIValid() const { return rssi_valid; }

    // [10b] RSSI data validation and range checking
    bool validateRSSIData(const Json::Value& message);

    // [10c] RSSI history buffer for trend display
    size_t getRSSIHistorySize() const;
    void addRSSIToHistory(float rssi, std::chrono::steady_clock::time_point timestamp);
    float getLatestRSSI() const;
    float calculateRSSITrend() const;

    // [10d] Signal quality indicators based on RSSI values
    std::string getSignalQuality(float rssi) const;
    bool hasGoodSignalQuality() const;
    void updateRSSI(float rssi);

    // [10f] RSSI data logging for debugging and analysis
    void enableRSSILogging(const std::string& log_path);
    void disableRSSILogging();
    bool isRSSILoggingEnabled() const;
    void logRSSIData(const Json::Value& rssi_data);

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

    // [10c] RSSI history buffer for trend analysis
    struct RSSIHistoryEntry {
        float rssi;
        std::chrono::steady_clock::time_point timestamp;
    };
    std::vector<RSSIHistoryEntry> rssi_history;
    static const size_t MAX_RSSI_HISTORY = 1000; // Configurable history size

    // [10f] RSSI logging support
    bool rssi_logging_enabled = false;
    std::string rssi_log_path;
    std::ofstream rssi_log_file;

    // [11a] Reconnection state machine variables
    enum class ReconnectionState {
        DISCONNECTED,
        RECONNECTING,
        MANUAL_CONTROL,
        GRACEFUL_DEGRADATION
    };
    ReconnectionState reconnection_state = ReconnectionState::DISCONNECTED;
    std::vector<int> retry_intervals = {1000, 2000, 4000, 8000, 16000, 32000, 60000}; // Default exponential
    size_t current_retry_index = 0;

    // [11b] Enhanced exponential backoff
    double backoff_multiplier = 2.0;
    int min_retry_interval = 1000; // 1 second
    int max_retry_interval = 60000; // 60 seconds

    // [11c] Connection health monitoring with heartbeat
    bool heartbeat_enabled = true;
    int heartbeat_interval_ms = 5000; // 5 second heartbeat
    std::chrono::steady_clock::time_point last_heartbeat_time;
    std::chrono::steady_clock::time_point last_heartbeat_sent;

    // [11d] Graceful degradation state
    bool graceful_degradation_mode = false;
    bool blocking_operations = false;
    bool responsive_state = true;

    // [11e] Manual reconnection override
    bool auto_reconnect_enabled = true;
    bool manual_override_active = false;

    // [11f] Connection statistics tracking
    ConnectionStatistics connection_stats;
    std::chrono::steady_clock::time_point connection_attempt_start;

    // [12a-12f] Enhanced connection monitoring variables

    // [12b] Network error detection
    int last_network_error_code = 0;
    std::string last_network_error_name;

    // [12c] Connection quality metrics
    int connection_latency_ms = -1;  // -1 = not measured
    float packet_loss_percentage = 0.0f;
    int throughput_bytes_per_sec = 0;

    // [12e] Error notification system
    std::vector<std::string> pending_notifications;
    mutable std::mutex notification_mutex;

    // [12f] Connection event logging
    bool connection_event_logging_enabled = false;
    std::string connection_event_log_path;
    std::ofstream connection_event_log_file;
    std::vector<std::string> connection_event_history;
    static const size_t MAX_EVENT_HISTORY = 500;  // Keep last 500 events

    // [13a-13f] Enhanced GUI and configuration variables

    // [13a] Professional GUI styling and layout
    bool show_advanced_settings = false;
    bool show_status_dashboard = false;
    bool gui_styling_applied = false;

    // [13b] RSSI graph and meter display
    std::vector<float> rssi_plot_times;
    std::vector<float> rssi_plot_values;
    static const size_t MAX_PLOT_POINTS = 300;  // 5 minutes at 1Hz
    bool show_rssi_graph = true;
    float rssi_meter_min = -120.0f;
    float rssi_meter_max = 0.0f;

    // [13c] Configuration persistence
    Json::Value session_config;
    bool session_config_loaded = false;
    std::string config_file_path;

    // [13d] Advanced connection settings
    int connection_timeout_ms = 5000;
    bool show_connection_settings = false;
    char timeout_buffer[16] = "5000";
    char heartbeat_buffer[16] = "5000";

    // [13e] Status dashboard metrics
    bool show_connection_metrics = true;
    bool show_statistics_panel = true;
    std::chrono::steady_clock::time_point last_gui_update;

    // [13f] Manual control state
    bool show_manual_controls = false;
    char test_message_buffer[256] = "";
    char test_data_buffer[512] = "";
};
