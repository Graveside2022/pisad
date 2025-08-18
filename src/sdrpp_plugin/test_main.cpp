#include <iostream>
#include <json/json.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>

/**
 * Test utility for SDR++ PISAD Bridge Plugin
 *
 * Validates TCP communication to PISAD services
 * without requiring full SDR++ environment.
 */

int main(int argc, char* argv[]) {
    std::string host = "localhost";
    int port = 8081;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--host" && i + 1 < argc) {
            host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        }
    }

    std::cout << "Testing PISAD Bridge Plugin communication to "
              << host << ":" << port << std::endl;

    // Create socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "Failed to create socket" << std::endl;
        return 1;
    }

    // Configure server address
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);

    if (inet_pton(AF_INET, host.c_str(), &server_addr.sin_addr) <= 0) {
        std::cerr << "Invalid address: " << host << std::endl;
        close(sock);
        return 1;
    }

    // Attempt connection
    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Connection failed to " << host << ":" << port << std::endl;
        close(sock);
        return 1;
    }

    std::cout << "Successfully connected to PISAD service!" << std::endl;

    // Send test message
    Json::Value test_message;
    test_message["type"] = "ping";
    test_message["source"] = "sdrpp_plugin_test";

    Json::StreamWriterBuilder builder;
    std::string json_string = Json::writeString(builder, test_message);

    if (send(sock, json_string.c_str(), json_string.length(), 0) < 0) {
        std::cerr << "Failed to send test message" << std::endl;
        close(sock);
        return 1;
    }

    std::cout << "Test message sent successfully" << std::endl;

    // Clean up
    close(sock);
    return 0;
}
