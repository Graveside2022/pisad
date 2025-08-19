/**
 * Test Suite for PISAD Bridge TCP Client
 * Tests for TASK-5.2.2-CLIENT implementation
 */

#include "pisad_bridge.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <thread>
#include <chrono>
#include <filesystem>

class PisadBridgeTest : public ::testing::Test {
protected:
    void SetUp() override {
        module = std::make_unique<PisadBridgeModule>("test_bridge");
    }

    void TearDown() override {
        if (module) {
            module->disable();
        }
    }

    std::unique_ptr<PisadBridgeModule> module;
};

// RED PHASE: Test [8a] - TCP Client Connection Class
TEST_F(PisadBridgeTest, ConfigurableHostPortSettings) {
    // Test that connection uses configured host/port settings
    EXPECT_FALSE(module->isConnected());

    // This should fail initially (RED phase)
    module->setConnectionSettings("192.168.1.100", 8081);
    EXPECT_EQ(module->getHost(), "192.168.1.100");
    EXPECT_EQ(module->getPort(), 8081);
}

// RED PHASE: Test [8b] - JSON Message Serialization
TEST_F(PisadBridgeTest, FrequencyControlSerialization) {
    // Test serialization of frequency control messages
    std::string json_msg = module->serializeFrequencyControl(433.5e6, 1);

    // This should fail initially (RED phase)
    EXPECT_TRUE(json_msg.find("\"type\":\"freq_control\"") != std::string::npos);
    EXPECT_TRUE(json_msg.find("\"frequency\":433500000") != std::string::npos);
    EXPECT_TRUE(json_msg.find("\"sequence\":1") != std::string::npos);
}

// RED PHASE: Test [8c] - JSON Message Deserialization
TEST_F(PisadBridgeTest, RSSIDataDeserialization) {
    // Test deserialization of RSSI data from PISAD
    std::string rssi_json = R"({
        "type": "rssi_data",
        "timestamp": "2025-08-18T22:00:00Z",
        "rssi": -45.2,
        "frequency": 433500000,
        "sequence": 123
    })";

    // This should fail initially (RED phase)
    EXPECT_TRUE(module->deserializeRSSIData(rssi_json));
    EXPECT_NEAR(module->getCurrentRSSI(), -45.2f, 0.1f);
    EXPECT_EQ(module->getLastSequence(), 123);
}

// RED PHASE: Test [10a] - Enhanced RSSI Stream Processing
TEST_F(PisadBridgeTest, RobustRSSIStreamProcessing) {
    // Test that handlePISADMessage processes RSSI streams robustly
    Json::Value rssi_message;
    rssi_message["type"] = "rssi_update";
    rssi_message["rssi"] = -42.5;
    rssi_message["frequency"] = 2437000000.0;
    rssi_message["timestamp"] = "2025-08-18T23:50:00Z";
    rssi_message["sequence"] = 456;

    // This should fail initially - enhanced processing not implemented
    module->handleRSSIMessage(rssi_message);
    EXPECT_NEAR(module->getCurrentRSSI(), -42.5f, 0.1f);
    EXPECT_EQ(module->getLastSequence(), 456);
    EXPECT_TRUE(module->isRSSIValid());
}

// RED PHASE: Test [10b] - RSSI Data Validation and Range Checking
TEST_F(PisadBridgeTest, RSSIDataValidationAndRangeChecking) {
    Json::Value valid_message;
    valid_message["type"] = "rssi_update";
    valid_message["rssi"] = -50.0;
    valid_message["frequency"] = 433500000.0;

    Json::Value invalid_rssi_message;
    invalid_rssi_message["type"] = "rssi_update";
    invalid_rssi_message["rssi"] = -200.0; // Invalid RSSI value
    invalid_rssi_message["frequency"] = 433500000.0;

    // This should fail initially - validation not implemented
    EXPECT_TRUE(module->validateRSSIData(valid_message));
    EXPECT_FALSE(module->validateRSSIData(invalid_rssi_message));

    // Test range validation for RSSI (-120 to 0 dBm typical range)
    Json::Value edge_case_low;
    edge_case_low["type"] = "rssi_update";
    edge_case_low["rssi"] = -120.0;
    edge_case_low["frequency"] = 433500000.0;

    Json::Value edge_case_high;
    edge_case_high["type"] = "rssi_update";
    edge_case_high["rssi"] = 0.0;
    edge_case_high["frequency"] = 433500000.0;

    EXPECT_TRUE(module->validateRSSIData(edge_case_low));
    EXPECT_TRUE(module->validateRSSIData(edge_case_high));
}

// RED PHASE: Test [10c] - RSSI History Buffer Implementation
TEST_F(PisadBridgeTest, RSSIHistoryBufferForTrendDisplay) {
    // Test RSSI history buffer with configurable size
    EXPECT_EQ(module->getRSSIHistorySize(), 0);

    // Add RSSI samples to history
    module->addRSSIToHistory(-45.0f, std::chrono::steady_clock::now());
    module->addRSSIToHistory(-47.2f, std::chrono::steady_clock::now());
    module->addRSSIToHistory(-43.8f, std::chrono::steady_clock::now());

    // This should fail initially - history buffer not implemented
    EXPECT_EQ(module->getRSSIHistorySize(), 3);
    EXPECT_NEAR(module->getLatestRSSI(), -43.8f, 0.1f);

    // Test trend calculation
    float trend = module->calculateRSSITrend();
    EXPECT_GT(trend, 0.0f); // Trending upward (less negative)
}

// RED PHASE: Test [10d] - Signal Quality Indicators
TEST_F(PisadBridgeTest, SignalQualityIndicatorsBasedOnRSSI) {
    // Test signal quality categorization based on RSSI values
    EXPECT_EQ(module->getSignalQuality(-30.0f), "Excellent");
    EXPECT_EQ(module->getSignalQuality(-50.0f), "Good");
    EXPECT_EQ(module->getSignalQuality(-70.0f), "Fair");
    EXPECT_EQ(module->getSignalQuality(-90.0f), "Poor");
    EXPECT_EQ(module->getSignalQuality(-110.0f), "Very Poor");

    // This should fail initially - signal quality indicators not implemented
    EXPECT_TRUE(module->hasGoodSignalQuality());
    module->updateRSSI(-85.0f);
    EXPECT_FALSE(module->hasGoodSignalQuality());
}

// RED PHASE: Test [10f] - RSSI Data Logging for Debugging
TEST_F(PisadBridgeTest, RSSIDataLoggingForDebugging) {
    // Create a temporary file for testing
    std::filesystem::path temp_dir;
    try {
        temp_dir = std::filesystem::temp_directory_path();
    } catch (const std::exception&) {
        // Skip test if we can't get a writable temp directory
        GTEST_SKIP() << "Cannot access writable temporary directory for logging test";
        return;
    }
    
    std::string temp_log_file = (temp_dir / "rssi_test.log").string();
    
    // Test RSSI data logging with timestamps and metadata
    module->enableRSSILogging(temp_log_file);

    Json::Value rssi_data;
    rssi_data["type"] = "rssi_update";
    rssi_data["rssi"] = -42.0;
    rssi_data["frequency"] = 433500000.0;
    rssi_data["timestamp"] = "2025-08-18T23:50:00Z";

    // This should fail initially - logging not implemented
    module->logRSSIData(rssi_data);
    EXPECT_TRUE(module->isRSSILoggingEnabled());

    // Verify log file contains data
    std::ifstream log_file(temp_log_file);
    EXPECT_TRUE(log_file.good());

    std::string log_content;
    std::getline(log_file, log_content);
    EXPECT_TRUE(log_content.find("-42.0") != std::string::npos);

    module->disableRSSILogging();
    
    // Clean up temporary file
    try {
        std::filesystem::remove(temp_log_file);
    } catch (const std::exception&) {
        // Ignore cleanup errors
    }
}

// RED PHASE: Test [8d] - Exponential Backoff
TEST_F(PisadBridgeTest, ExponentialBackoffReconnection) {
    // Test automatic reconnection with exponential backoff
    module->enable();

    // Force disconnect and measure reconnection intervals
    module->simulateConnectionLoss();

    // This should fail initially (RED phase)
    EXPECT_EQ(module->getReconnectInterval(), 1000); // 1 second initially

    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    EXPECT_EQ(module->getReconnectInterval(), 2000); // 2 seconds after first failure
}

// RED PHASE: Test [8e] - Connection Status Monitoring
TEST_F(PisadBridgeTest, ConnectionStatusMonitoring) {
    // Test connection status monitoring and error reporting
    EXPECT_EQ(module->getConnectionStatus(), "Disconnected");

    // This should fail initially (RED phase)
    module->setConnectionError("Network unreachable");
    EXPECT_EQ(module->getConnectionStatus(), "Error: Network unreachable");
    EXPECT_TRUE(module->hasConnectionError());
}

// RED PHASE: Test [11a] - Reconnection State Machine with Configurable Intervals
TEST_F(PisadBridgeTest, ReconnectionStateMachineConfigurableIntervals) {
    // Test reconnection state machine states and transitions
    EXPECT_EQ(module->getReconnectionState(), "DISCONNECTED");

    // Configure custom retry intervals
    module->setRetryIntervals({500, 1000, 2000, 5000}); // Custom intervals in ms

    module->enable();
    module->simulateConnectionLoss();

    // This should fail initially (RED phase) - state machine not implemented
    EXPECT_EQ(module->getReconnectionState(), "RECONNECTING");
    EXPECT_EQ(module->getCurrentRetryInterval(), 500); // First interval

    // Wait for first retry attempt
    std::this_thread::sleep_for(std::chrono::milliseconds(600));
    EXPECT_EQ(module->getCurrentRetryInterval(), 1000); // Second interval
    EXPECT_GT(module->getConnectionAttempts(), 1);
}

// RED PHASE: Test [11b] - Enhanced Exponential Backoff Algorithm
TEST_F(PisadBridgeTest, EnhancedExponentialBackoffAlgorithm) {
    // Test exponential backoff with proper boundaries (1s to 60s)
    module->enable();

    // Reset to known state
    module->resetReconnectionState();
    EXPECT_EQ(module->getCurrentRetryInterval(), 1000); // 1 second start

    // Simulate multiple failures to test exponential growth
    for (int i = 0; i < 8; i++) {
        module->simulateConnectionFailure();
    }

    // This should fail initially (RED phase) - enhanced backoff not implemented
    EXPECT_EQ(module->getCurrentRetryInterval(), 60000); // Capped at 60 seconds
    EXPECT_EQ(module->getBackoffMultiplier(), 2.0); // Should use 2x multiplier
}

// RED PHASE: Test [11c] - Connection Health Monitoring with Heartbeat
TEST_F(PisadBridgeTest, ConnectionHealthMonitoringWithHeartbeat) {
    // Test heartbeat-based connection health monitoring
    module->enable();
    module->setConnectionSettings("127.0.0.1", 8081);

    // Simulate successful connection
    module->connectToPISAD();
    EXPECT_TRUE(module->isConnected());

    // Test heartbeat monitoring
    EXPECT_TRUE(module->isHeartbeatEnabled());
    EXPECT_GT(module->getHeartbeatInterval(), 0);

    // This should fail initially (RED phase) - heartbeat monitoring not implemented
    auto last_heartbeat = module->getLastHeartbeatTime();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Heartbeat should update automatically
    auto current_heartbeat = module->getLastHeartbeatTime();
    EXPECT_GT(current_heartbeat, last_heartbeat);
}

// RED PHASE: Test [11d] - Graceful Degradation When Server Unavailable
TEST_F(PisadBridgeTest, GracefulDegradationServerUnavailable) {
    // Test graceful degradation without blocking or crashes
    module->enable();
    module->setConnectionSettings("192.168.255.254", 12345); // Non-existent server

    // Attempt connection to unavailable server
    module->connectToPISAD();
    EXPECT_FALSE(module->isConnected());

    // This should fail initially (RED phase) - graceful degradation not implemented
    EXPECT_EQ(module->getDegradationMode(), "GRACEFUL");
    EXPECT_FALSE(module->isBlocking()); // Should not block operations
    EXPECT_TRUE(module->isResponsive()); // Should remain responsive

    // Test that other operations continue working
    EXPECT_TRUE(module->canProcessCommands());
    EXPECT_NO_THROW(module->serializeFrequencyControl(433.5e6, 1));
}

// RED PHASE: Test [11e] - Manual Reconnection Override Capability
TEST_F(PisadBridgeTest, ManualReconnectionOverrideCapability) {
    // Test manual reconnection override for operator control
    module->enable();
    module->simulateConnectionLoss();

    // Should be in automatic reconnection mode initially
    EXPECT_TRUE(module->isAutoReconnectEnabled());
    EXPECT_EQ(module->getReconnectionState(), "RECONNECTING");

    // This should fail initially (RED phase) - manual override not implemented
    module->enableManualOverride();
    EXPECT_FALSE(module->isAutoReconnectEnabled());
    EXPECT_EQ(module->getReconnectionState(), "MANUAL_CONTROL");

    // Test manual reconnection trigger
    bool manual_result = module->triggerManualReconnection();
    EXPECT_TRUE(manual_result);

    // Test returning to automatic mode
    module->disableManualOverride();
    EXPECT_TRUE(module->isAutoReconnectEnabled());
}

// RED PHASE: Test [11f] - Connection Statistics Tracking for Debugging
TEST_F(PisadBridgeTest, ConnectionStatisticsTrackingForDebugging) {
    // Test comprehensive connection statistics for debugging
    module->enable();

    // Get initial statistics
    auto stats = module->getConnectionStatistics();

    // This should fail initially (RED phase) - statistics tracking not implemented
    EXPECT_GE(stats.total_attempts, 0);
    EXPECT_GE(stats.successful_connections, 0);
    EXPECT_GE(stats.failed_connections, 0);
    EXPECT_GE(stats.average_connection_time_ms, 0);

    // Simulate multiple connection attempts
    for (int i = 0; i < 3; i++) {
        module->simulateConnectionLoss();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Verify statistics updated
    auto updated_stats = module->getConnectionStatistics();
    EXPECT_GT(updated_stats.total_attempts, stats.total_attempts);
    EXPECT_GT(updated_stats.failed_connections, stats.failed_connections);

    // Test statistics reset capability
    module->resetConnectionStatistics();
    auto reset_stats = module->getConnectionStatistics();
    EXPECT_EQ(reset_stats.total_attempts, 0);
}

// RED PHASE: Test [8f] - GUI Integration
TEST_F(PisadBridgeTest, GUIStatusDisplay) {
    // Test GUI displays connection status correctly
    module->enable();

    // This should fail initially (RED phase)
    std::string gui_status = module->getGUIConnectionStatus();
    EXPECT_FALSE(gui_status.empty());
    EXPECT_TRUE(gui_status.find("DISCONNECTED") != std::string::npos);
}

// RED PHASE: Test [12a] - Enhanced Connection Status Display
TEST_F(PisadBridgeTest, DetailedConnectionStatusDisplay) {
    // Test enhanced status display with detailed error messages
    module->enable();

    // This should fail initially (RED phase) - missing detailed status methods
    std::string detailed_status = module->getDetailedConnectionStatus();
    EXPECT_FALSE(detailed_status.empty());

    // Simulate connection error
    module->setConnectionError("TCP connection refused: ECONNREFUSED (111)");

    detailed_status = module->getDetailedConnectionStatus();
    EXPECT_TRUE(detailed_status.find("ECONNREFUSED") != std::string::npos);
    EXPECT_TRUE(detailed_status.find("111") != std::string::npos);
}

// RED PHASE: Test [12b] - Network-Level Error Detection
TEST_F(PisadBridgeTest, NetworkErrorDetection) {
    // Test network error classification and reporting
    module->enable();

    // This should fail initially (RED phase) - missing error detection methods
    module->simulateNetworkError("ETIMEDOUT", 110);
    EXPECT_EQ(module->getLastNetworkErrorCode(), 110);
    EXPECT_EQ(module->getLastNetworkErrorName(), "ETIMEDOUT");

    module->simulateNetworkError("ECONNREFUSED", 111);
    EXPECT_EQ(module->getLastNetworkErrorCode(), 111);
    EXPECT_EQ(module->getLastNetworkErrorName(), "ECONNREFUSED");
}

// RED PHASE: Test [12c] - Connection Quality Metrics
TEST_F(PisadBridgeTest, ConnectionQualityMetrics) {
    // Test latency measurement and quality tracking
    module->enable();

    // This should fail initially (RED phase) - missing quality metrics
    EXPECT_EQ(module->getConnectionLatencyMs(), -1);  // Invalid until measured
    EXPECT_EQ(module->getPacketLossPercentage(), 0.0f);
    EXPECT_EQ(module->getThroughputBytesPerSec(), 0);

    // Simulate quality measurement
    module->updateConnectionLatency(45);
    module->updatePacketLoss(2.5f);
    module->updateThroughput(1024);

    EXPECT_EQ(module->getConnectionLatencyMs(), 45);
    EXPECT_FLOAT_EQ(module->getPacketLossPercentage(), 2.5f);
    EXPECT_EQ(module->getThroughputBytesPerSec(), 1024);
}

// RED PHASE: Test [12d] - Visual Status Indicators
TEST_F(PisadBridgeTest, VisualStatusIndicators) {
    // Test color-coded status and health indicators
    module->enable();

    // This should fail initially (RED phase) - missing visual indicator methods
    EXPECT_EQ(module->getConnectionStatusColor(), "red");  // Disconnected = red
    EXPECT_EQ(module->getConnectionHealthMeter(), 0);      // 0-100 scale

    // Simulate good connection
    module->setConnected(true);
    module->updateConnectionLatency(25);  // Good latency

    EXPECT_EQ(module->getConnectionStatusColor(), "green");
    EXPECT_GT(module->getConnectionHealthMeter(), 80);
}

// RED PHASE: Test [12e] - Error Notification System
TEST_F(PisadBridgeTest, ErrorNotificationSystem) {
    // Test critical error notifications and alerts
    module->enable();

    // This should fail initially (RED phase) - missing notification methods
    EXPECT_FALSE(module->hasPendingNotifications());

    // Trigger critical error
    module->triggerCriticalError("Server unreachable for 60 seconds");

    EXPECT_TRUE(module->hasPendingNotifications());
    auto notification = module->getNextNotification();
    EXPECT_TRUE(notification.find("Server unreachable") != std::string::npos);
    EXPECT_FALSE(module->hasPendingNotifications());  // Should be consumed
}

// RED PHASE: Test [12f] - Connection Event Logging
TEST_F(PisadBridgeTest, ConnectionEventLogging) {
    // Test comprehensive connection event logging
    module->enable();

    // This should fail initially (RED phase) - missing logging methods
    module->enableConnectionEventLogging("/tmp/connection_events.log");
    EXPECT_TRUE(module->isConnectionEventLoggingEnabled());

    // Generate events
    module->logConnectionEvent("CONNECTION_ATTEMPT", "Attempting connection to localhost:8081");
    module->logConnectionEvent("CONNECTION_SUCCESS", "Connected successfully after 1200ms");
    module->logConnectionEvent("HEARTBEAT_TIMEOUT", "Heartbeat timeout detected after 5000ms");

    // Verify logging
    auto events = module->getConnectionEventHistory();
    EXPECT_GE(events.size(), 3);
    EXPECT_TRUE(events[0].find("CONNECTION_ATTEMPT") != std::string::npos);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
