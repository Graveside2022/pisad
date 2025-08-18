/**
 * Test Suite for PISAD Bridge TCP Client
 * Tests for TASK-5.2.2-CLIENT implementation
 */

#include "pisad_bridge.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <thread>
#include <chrono>

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
        "sequence": 100
    })";
    
    // This should fail initially (RED phase)
    bool result = module->deserializeRSSIData(rssi_json);
    EXPECT_TRUE(result);
    EXPECT_FLOAT_EQ(module->getCurrentRSSI(), -45.2f);
    EXPECT_EQ(module->getLastSequence(), 100);
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

// RED PHASE: Test [8f] - GUI Integration
TEST_F(PisadBridgeTest, GUIStatusDisplay) {
    // Test GUI displays connection status correctly
    module->enable();
    
    // This should fail initially (RED phase)
    std::string gui_status = module->getGUIConnectionStatus();
    EXPECT_FALSE(gui_status.empty());
    EXPECT_TRUE(gui_status.find("DISCONNECTED") != std::string::npos);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}