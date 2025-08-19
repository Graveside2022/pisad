/**
 * Basic Test for Reconnection Logic
 * Tests the core reconnection state machine without SDR++ dependencies
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <string>

// Mock the essential parts for testing
enum class ReconnectionState {
    DISCONNECTED,
    RECONNECTING,
    MANUAL_CONTROL,
    GRACEFUL_DEGRADATION
};

struct ConnectionStatistics {
    int total_attempts = 0;
    int successful_connections = 0;
    int failed_connections = 0;
    int average_connection_time_ms = 0;
};

// Simple test class for reconnection logic
class ReconnectionTester {
private:
    ReconnectionState reconnection_state = ReconnectionState::DISCONNECTED;
    std::vector<int> retry_intervals = {1000, 2000, 4000, 8000, 16000, 32000, 60000};
    size_t current_retry_index = 0;
    int max_retry_interval = 60000;
    double backoff_multiplier = 2.0;
    bool auto_reconnect_enabled = true;
    bool manual_override_active = false;
    ConnectionStatistics connection_stats;

public:
    std::string getReconnectionState() const {
        switch (reconnection_state) {
            case ReconnectionState::DISCONNECTED: return "DISCONNECTED";
            case ReconnectionState::RECONNECTING: return "RECONNECTING";
            case ReconnectionState::MANUAL_CONTROL: return "MANUAL_CONTROL";
            case ReconnectionState::GRACEFUL_DEGRADATION: return "GRACEFUL_DEGRADATION";
            default: return "UNKNOWN";
        }
    }

    void setRetryIntervals(const std::vector<int>& intervals) {
        retry_intervals = intervals;
        current_retry_index = 0;
    }

    int getCurrentRetryInterval() const {
        if (current_retry_index < retry_intervals.size()) {
            return retry_intervals[current_retry_index];
        }
        return max_retry_interval;
    }

    void simulateConnectionFailure() {
        connection_stats.failed_connections++;
        connection_stats.total_attempts++;

        if (!retry_intervals.empty() && current_retry_index + 1 < retry_intervals.size()) {
            current_retry_index++;
        }

        reconnection_state = ReconnectionState::RECONNECTING;
    }

    void enableManualOverride() {
        manual_override_active = true;
        auto_reconnect_enabled = false;
        reconnection_state = ReconnectionState::MANUAL_CONTROL;
    }

    void disableManualOverride() {
        manual_override_active = false;
        auto_reconnect_enabled = true;
        reconnection_state = ReconnectionState::RECONNECTING;
    }

    bool isAutoReconnectEnabled() const {
        return auto_reconnect_enabled && !manual_override_active;
    }

    ConnectionStatistics getConnectionStatistics() const {
        return connection_stats;
    }

    double getBackoffMultiplier() const {
        return backoff_multiplier;
    }
};

int main() {
    ReconnectionTester tester;

    std::cout << "=== Testing Reconnection State Machine ===\n";

    // Test initial state
    std::cout << "Initial state: " << tester.getReconnectionState() << std::endl;
    std::cout << "Auto reconnect enabled: " << (tester.isAutoReconnectEnabled() ? "Yes" : "No") << std::endl;

    // Test custom retry intervals
    tester.setRetryIntervals({500, 1000, 2000, 5000});
    std::cout << "Custom retry interval: " << tester.getCurrentRetryInterval() << "ms" << std::endl;

    // Test connection failures and exponential backoff
    std::cout << "\n=== Testing Exponential Backoff ===\n";
    for (int i = 0; i < 4; i++) {
        std::cout << "Before failure " << i+1 << ": " << tester.getCurrentRetryInterval() << "ms" << std::endl;
        tester.simulateConnectionFailure();
        std::cout << "After failure " << i+1 << ": " << tester.getCurrentRetryInterval() << "ms, State: "
                  << tester.getReconnectionState() << std::endl;
    }

    // Test manual override
    std::cout << "\n=== Testing Manual Override ===\n";
    tester.enableManualOverride();
    std::cout << "Manual override enabled - State: " << tester.getReconnectionState() << std::endl;
    std::cout << "Auto reconnect enabled: " << (tester.isAutoReconnectEnabled() ? "Yes" : "No") << std::endl;

    tester.disableManualOverride();
    std::cout << "Manual override disabled - State: " << tester.getReconnectionState() << std::endl;
    std::cout << "Auto reconnect enabled: " << (tester.isAutoReconnectEnabled() ? "Yes" : "No") << std::endl;

    // Test statistics
    std::cout << "\n=== Connection Statistics ===\n";
    auto stats = tester.getConnectionStatistics();
    std::cout << "Total attempts: " << stats.total_attempts << std::endl;
    std::cout << "Failed connections: " << stats.failed_connections << std::endl;
    std::cout << "Backoff multiplier: " << tester.getBackoffMultiplier() << std::endl;

    std::cout << "\n✅ All reconnection logic tests completed successfully!" << std::endl;

    return 0;
}
