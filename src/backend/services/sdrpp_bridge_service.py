"""
SDR++ Bridge Service for TCP Communication

Provides TCP server for communication between ground SDR++ desktop
and drone PISAD systems enabling dual-SDR coordination.

PRD References:
- NFR1: Communication reliability (<1% packet loss)
- NFR2: Signal processing latency (<100ms)
- FR9: Enhanced telemetry streaming with dual-SDR coordination
"""

import asyncio
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from src.backend.utils.logging import get_logger

logger = get_logger(__name__)


class SDRPPBridgeService:
    """TCP server service for SDR++ plugin communication."""

    def __init__(self) -> None:
        """Initialize SDR++ bridge service with default configuration."""
        self.host = "0.0.0.0"  # Listen on all interfaces
        self.port = 8081  # SDR++ communication port
        self.server: asyncio.Server | None = None  # TCP server instance
        self.clients: list[Any] = []  # Connected clients list
        self.running = False  # Service running state

        # Valid message types for JSON protocol
        self.valid_message_types: set[str] = {
            "rssi_update",
            "freq_control",
            "homing_state",
            "error",
            "heartbeat",
        }

        # Signal processor dependency for RSSI data access
        self._signal_processor: Any | None = None
        self._message_sequence = 0  # Sequence counter for messages

        # Heartbeat monitoring (based on MAVLink service pattern)
        self.heartbeat_timeout = 30.0  # 30 seconds for SDR++ connections
        self.client_heartbeats: dict[tuple[str, int], float] = {}

        # [1m] Safety manager integration for communication health monitoring
        self._safety_manager: Any | None = None
        self._safety_timeout = 10.0  # <10s timeout for safety notifications
        self._last_safety_notification = 0.0
        self._communication_quality_threshold = 0.5  # 50% quality threshold

        # [1n] Communication loss detection with safety event triggers
        self._connection_lost_callbacks: list[Callable[..., Any]] = []
        self._connection_restored_callbacks: list[Callable[..., Any]] = []
        self._last_connection_check = time.time()
        self._consecutive_failures = 0
        self._failure_threshold = 3  # 3 consecutive failures trigger safety event

        # [1q] Network quality assessment metrics
        self._message_latency_history: list[float] = []
        self._message_success_count = 0
        self._message_failure_count = 0
        self._connection_start_time = 0.0

        logger.info("SDRPPBridgeService initialized with safety integration on port %s", self.port)

    async def start(self) -> None:
        """Start the TCP server for SDR++ plugin connections."""
        if self.running:
            logger.warning("SDR++ bridge service already running")
            return

        try:
            # Create async TCP server that listens for connections
            self.server = await asyncio.start_server(
                self._handle_client,  # Function to handle each client
                self.host,  # Listen on all interfaces (0.0.0.0)
                self.port,  # Port 8081 for SDR++ communication
            )

            self.running = True
            logger.info("SDR++ bridge server started on %s:%s", self.host, self.port)

        except Exception as e:
            logger.error("Failed to start SDR++ bridge server: %s", e)
            raise

    async def stop(self) -> None:
        """Stop the TCP server and clean up connections."""
        if not self.running:
            logger.warning("SDR++ bridge service not running")
            return

        try:
            self.running = False

            # Close all client connections
            for client in self.clients[:]:  # Copy list to avoid modification during iteration
                try:
                    client.close()
                    await client.wait_closed()
                except Exception as e:
                    logger.warning("Error closing client connection: %s", e)

            self.clients.clear()

            # Close the server
            if self.server:
                self.server.close()
                await self.server.wait_closed()
                self.server = None

            logger.info("SDR++ bridge server stopped")

        except Exception as e:
            logger.error("Error stopping SDR++ bridge server: %s", e)
            raise

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle incoming client connections (placeholder for now)."""
        client_addr = writer.get_extra_info("peername")
        logger.info("SDR++ client connected from %s", client_addr)

        # Add client to tracking list
        self.clients.append(writer)

        # Initialize heartbeat tracking for this client
        self.client_heartbeats[client_addr] = time.time()

        try:
            # Check if client is still connected by reading from it
            while self.running and not writer.is_closing():
                try:
                    # Try to read a small amount of data (with timeout)
                    data = await asyncio.wait_for(reader.read(1024), timeout=1.0)
                    if not data:  # Client closed connection
                        break
                    # For now, we just read and discard data
                    # In next subtasks, we'll add message processing
                except TimeoutError:
                    # Timeout is normal - just continue checking
                    continue
                except Exception:
                    # Any other error means client disconnected
                    break

        except Exception as e:
            logger.error("Error handling client %s: %s", client_addr, e)
        finally:
            # Remove client from tracking list
            if writer in self.clients:
                self.clients.remove(writer)

            # Remove from heartbeat tracking
            if client_addr in self.client_heartbeats:
                del self.client_heartbeats[client_addr]

            # Close connection
            try:
                if not writer.is_closing():
                    writer.close()
                    await writer.wait_closed()
            except Exception:
                pass  # Ignore errors during cleanup

            logger.info("SDR++ client %s disconnected", client_addr)

    def _parse_message(self, message_data: str) -> dict[str, Any] | None:
        """
        Parse incoming JSON message with error handling.

        Args:
            message_data: Raw JSON string from TCP client

        Returns:
            Parsed message dictionary or None if invalid

        PRD References:
        - NFR1: Communication reliability with error handling
        - NFR2: Fast parsing to maintain <100ms latency
        """
        try:
            # Parse JSON from string
            parsed = json.loads(message_data)

            # Validate message structure
            if not isinstance(parsed, dict):
                logger.warning("Message is not a JSON object")
                return None

            # Check required fields
            if "type" not in parsed:
                logger.warning("Message missing required 'type' field")
                return None

            # Validate message type
            message_type = parsed["type"]
            if message_type not in self.valid_message_types:
                logger.warning("Invalid message type: %s", message_type)
                return None

            logger.debug("Successfully parsed %s message", message_type)
            return parsed

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSON message: %s", e)
            return None
        except Exception as e:
            logger.error("Unexpected error parsing message: %s", e)
            return None

    def set_signal_processor(self, signal_processor: Any) -> None:
        """Set signal processor service for RSSI data access.

        Args:
            signal_processor: Signal processor service instance
        """
        self._signal_processor = signal_processor
        logger.debug("Signal processor service configured for RSSI streaming")

    def set_safety_manager(self, safety_manager: Any) -> None:
        """[1m] Set safety manager for communication health monitoring.

        Args:
            safety_manager: Safety manager service instance for health notifications
        """
        self._safety_manager = safety_manager
        logger.info("Safety manager configured for communication health monitoring")

    def add_connection_lost_callback(self, callback: Callable[..., Any]) -> None:
        """[1n] Add callback for communication loss detection.

        Args:
            callback: Function to call when communication is lost
        """
        self._connection_lost_callbacks.append(callback)
        logger.debug("Connection lost callback registered")

    def add_connection_restored_callback(self, callback: Callable[..., Any]) -> None:
        """[1n] Add callback for communication restoration.

        Args:
            callback: Function to call when communication is restored
        """
        self._connection_restored_callbacks.append(callback)
        logger.debug("Connection restored callback registered")

    async def handle_rssi_request(self) -> dict[str, Any]:
        """Handle RSSI streaming request from SDR++ client.

        Returns:
            JSON response with current RSSI data or error message

        PRD References:
        - FR9: Enhanced telemetry streaming with dual-SDR coordination
        - NFR2: Signal processing latency (<100ms per computation cycle)
        """
        try:
            # Check if signal processor is available
            if self._signal_processor is None:
                return {
                    "type": "error",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "data": {
                        "error": "signal_processor_unavailable",
                        "message": "Signal processor service not configured",
                    },
                    "sequence": self._get_next_sequence(),
                }

            # Get current RSSI from signal processor
            current_rssi = self._signal_processor.get_current_rssi()

            # Create RSSI update message
            response = {
                "type": "rssi_update",
                "timestamp": datetime.now(UTC).isoformat(),
                "data": {
                    "rssi": current_rssi,
                    "frequency": 2437000000,  # Default frequency - will be configurable
                    "source": "drone_sdr",
                },
                "sequence": self._get_next_sequence(),
            }

            logger.debug("RSSI streaming response: %.1f dBm", current_rssi)
            return response

        except Exception as e:
            logger.error("Error handling RSSI request: %s", e)
            return {
                "type": "error",
                "timestamp": datetime.now(UTC).isoformat(),
                "data": {"error": "rssi_handler_error", "message": f"Internal error: {e!s}"},
                "sequence": self._get_next_sequence(),
            }

    def _get_next_sequence(self) -> int:
        """Get next message sequence number.

        Returns:
            Incremented sequence number for message ordering
        """
        self._message_sequence += 1
        return self._message_sequence

    async def handle_frequency_control(self, message: dict[str, Any]) -> dict[str, Any]:
        """Handle frequency control request from SDR++ client.

        Args:
            message: Frequency control message with frequency in Hz

        Returns:
            JSON response with success/error status

        PRD References:
        - NFR4: Frequency control commands processed with <50ms response time
        - NFR1: Communication reliability with error handling
        - FR1: RF beacons (850 MHz - 6.5 GHz configurable)
        """
        try:
            # Check if signal processor is available
            if self._signal_processor is None:
                return {
                    "type": "error",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "data": {
                        "error": "signal_processor_unavailable",
                        "message": "Signal processor service not configured",
                    },
                    "sequence": self._get_next_sequence(),
                }

            # Extract frequency from message
            frequency = message.get("data", {}).get("frequency")
            if frequency is None:
                return {
                    "type": "error",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "data": {
                        "error": "missing_frequency",
                        "message": "Frequency field is required in data",
                    },
                    "sequence": self._get_next_sequence(),
                }

            # Validate frequency range (850 MHz - 6.5 GHz per PRD-FR1)
            min_freq = 850e6  # 850 MHz
            max_freq = 6.5e9  # 6.5 GHz

            if not (min_freq <= frequency <= max_freq):
                return {
                    "type": "error",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "data": {
                        "error": "frequency_out_of_range",
                        "message": f"Frequency {frequency/1e6:.1f} MHz outside valid range 850-6500 MHz",
                        "min_frequency": min_freq,
                        "max_frequency": max_freq,
                    },
                    "sequence": self._get_next_sequence(),
                }

            # Set frequency using signal processor
            self._signal_processor.set_frequency(frequency)

            # Create success response
            response = {
                "type": "freq_control_response",
                "timestamp": datetime.now(UTC).isoformat(),
                "data": {
                    "status": "success",
                    "frequency": frequency,
                    "message": f"Frequency updated to {frequency/1e6:.1f} MHz",
                },
                "sequence": self._get_next_sequence(),
            }

            logger.info("Frequency control: %.1f MHz", frequency / 1e6)
            return response

        except Exception as e:
            logger.error("Error handling frequency control request: %s", e)
            return {
                "type": "error",
                "timestamp": datetime.now(UTC).isoformat(),
                "data": {"error": "frequency_control_error", "message": f"Internal error: {e!s}"},
                "sequence": self._get_next_sequence(),
            }

    async def handle_heartbeat(
        self, message: dict[str, Any], client_addr: tuple[str, int]
    ) -> dict[str, Any]:
        """Handle heartbeat message from SDR++ client.

        Args:
            message: Heartbeat message from client
            client_addr: Client address tuple (host, port)

        Returns:
            JSON response with heartbeat acknowledgment

        PRD References:
        - NFR1: Communication reliability with heartbeat monitoring
        - NFR9: MTBF >10 hours with connection health tracking
        """
        try:
            # Update client heartbeat timestamp
            self.client_heartbeats[client_addr] = time.time()

            # Create heartbeat acknowledgment response
            response = {
                "type": "heartbeat_ack",
                "timestamp": datetime.now(UTC).isoformat(),
                "data": {"status": "received"},
                "sequence": self._get_next_sequence(),
            }

            logger.debug("Heartbeat received from %s", client_addr)
            return response

        except Exception as e:
            logger.error("Error handling heartbeat from %s: %s", client_addr, e)
            return {
                "type": "error",
                "timestamp": datetime.now(UTC).isoformat(),
                "data": {"error": "heartbeat_handler_error", "message": f"Internal error: {e!s}"},
                "sequence": self._get_next_sequence(),
            }

    async def _check_heartbeat_timeouts(self) -> None:
        """Check for heartbeat timeouts and disconnect stale clients.

        PRD References:
        - NFR1: Communication reliability with timeout detection
        - NFR9: MTBF >10 hours with automatic recovery
        """
        try:
            current_time = time.time()
            timed_out_clients = []

            # Check all client heartbeats for timeouts
            for client_addr, last_heartbeat in self.client_heartbeats.items():
                if current_time - last_heartbeat > self.heartbeat_timeout:
                    timed_out_clients.append(client_addr)

            # Disconnect timed out clients
            for client_addr in timed_out_clients:
                await self._disconnect_client(client_addr)
                logger.warning("Client %s disconnected due to heartbeat timeout", client_addr)

        except Exception as e:
            logger.error("Error checking heartbeat timeouts: %s", e)

    async def _disconnect_client(self, client_addr: tuple[str, int]) -> None:
        """Disconnect a client and clean up tracking.

        Args:
            client_addr: Client address tuple to disconnect
        """
        try:
            # Remove from heartbeat tracking
            if client_addr in self.client_heartbeats:
                del self.client_heartbeats[client_addr]

            # Find and close the client connection
            clients_to_remove = []
            for client in self.clients[:]:  # Copy to avoid modification during iteration
                try:
                    if client.get_extra_info("peername") == client_addr:
                        if not client.is_closing():
                            client.close()
                            await client.wait_closed()
                        clients_to_remove.append(client)
                except Exception as e:
                    logger.warning("Error closing client connection %s: %s", client_addr, e)
                    clients_to_remove.append(client)

            # Remove from client list
            for client in clients_to_remove:
                if client in self.clients:
                    self.clients.remove(client)

        except Exception as e:
            logger.error("Error disconnecting client %s: %s", client_addr, e)

    async def get_communication_health_status(self) -> dict[str, Any]:
        """[1q] Get comprehensive communication health status for safety decision matrix.

        Returns:
            Communication health metrics for safety system integration
        """
        try:
            current_time = time.time()
            connection_duration = (
                current_time - self._connection_start_time if self._connection_start_time > 0 else 0
            )

            # Calculate communication quality metrics
            total_messages = self._message_success_count + self._message_failure_count
            success_rate = (
                (self._message_success_count / total_messages) if total_messages > 0 else 0.0
            )

            # Calculate average latency
            avg_latency = (
                sum(self._message_latency_history) / len(self._message_latency_history)
                if self._message_latency_history
                else 0.0
            )

            # Determine overall health status
            is_healthy = (
                len(self.clients) > 0  # At least one client connected
                and success_rate
                >= self._communication_quality_threshold  # Success rate above threshold
                and avg_latency < 1000.0  # Latency under 1 second
                and self._consecutive_failures < self._failure_threshold  # Not in failure state
            )

            health_status = {
                "healthy": is_healthy,
                "connected_clients": len(self.clients),
                "connection_duration_seconds": connection_duration,
                "message_success_rate": success_rate,
                "average_latency_ms": avg_latency,
                "consecutive_failures": self._consecutive_failures,
                "communication_quality": success_rate,
                "last_heartbeat_age": (
                    min([current_time - hb for hb in self.client_heartbeats.values()])
                    if self.client_heartbeats
                    else float("inf")
                ),
                "total_messages_processed": total_messages,
                "quality_threshold": self._communication_quality_threshold,
            }

            return health_status

        except Exception as e:
            logger.error("Error getting communication health status: %s", e)
            return {
                "healthy": False,
                "error": str(e),
                "connected_clients": 0,
                "message_success_rate": 0.0,
            }

    async def safety_communication_loss(self, reason: str = "Communication timeout") -> None:
        """[1n] Handle communication loss with safety event triggers.

        Args:
            reason: Reason for communication loss
        """
        try:
            logger.warning("Communication loss detected: %s", reason)

            # Notify safety manager if available
            if self._safety_manager:
                try:
                    # Create safety event for communication loss
                    safety_event = {
                        "event_type": "communication_loss",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "source": "sdrpp_bridge",
                        "reason": reason,
                        "clients_affected": len(self.clients),
                        "consecutive_failures": self._consecutive_failures,
                    }

                    # Trigger safety manager notification
                    await self._safety_manager.handle_communication_loss(safety_event)
                    logger.info("Safety manager notified of communication loss")

                except Exception as e:
                    logger.error("Failed to notify safety manager of communication loss: %s", e)

            # Execute registered callbacks
            for callback in self._connection_lost_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(reason)
                    else:
                        callback(reason)
                except Exception as e:
                    logger.error("Error executing connection lost callback: %s", e)

            self._last_safety_notification = time.time()

        except Exception as e:
            logger.error("Error handling communication loss: %s", e)

    async def safety_communication_restored(self) -> None:
        """[1n] Handle communication restoration with safety notifications."""
        try:
            logger.info("Communication restored - notifying safety systems")

            # Reset failure counters
            self._consecutive_failures = 0
            self._connection_start_time = time.time()

            # Notify safety manager if available
            if self._safety_manager:
                try:
                    safety_event = {
                        "event_type": "communication_restored",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "source": "sdrpp_bridge",
                        "connected_clients": len(self.clients),
                    }

                    await self._safety_manager.handle_communication_restored(safety_event)
                    logger.info("Safety manager notified of communication restoration")

                except Exception as e:
                    logger.error(
                        "Failed to notify safety manager of communication restoration: %s", e
                    )

            # Execute registered callbacks
            for callback in self._connection_restored_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                except Exception as e:
                    logger.error("Error executing connection restored callback: %s", e)

        except Exception as e:
            logger.error("Error handling communication restoration: %s", e)

    async def check_safety_timeout(self) -> bool:
        """[1p] Check if communication safety timeout has been exceeded.

        Returns:
            True if safety timeout exceeded, False otherwise
        """
        try:
            current_time = time.time()

            # Check if we have any active connections
            if not self.clients:
                time_since_last_connection = current_time - self._last_connection_check
                if time_since_last_connection > self._safety_timeout:
                    logger.warning(
                        "Safety timeout exceeded: %.1f seconds without connection",
                        time_since_last_connection,
                    )
                    await self.safety_communication_loss(
                        f"Safety timeout exceeded: {time_since_last_connection:.1f}s"
                    )
                    return True
            else:
                self._last_connection_check = current_time

            return False

        except Exception as e:
            logger.error("Error checking safety timeout: %s", e)
            return True  # Err on the side of caution

    async def emergency_disconnect(self, reason: str = "Emergency safety disconnect") -> None:
        """[1o] Emergency disconnect for safety-triggered coordination shutdown.

        Args:
            reason: Reason for emergency disconnect
        """
        try:
            logger.critical("EMERGENCY DISCONNECT: %s", reason)

            # Immediately stop accepting new connections
            self.running = False

            # Notify all clients of emergency disconnect
            emergency_message = {
                "type": "emergency_disconnect",
                "timestamp": datetime.now(UTC).isoformat(),
                "data": {"reason": reason, "status": "emergency_shutdown"},
                "sequence": self._get_next_sequence(),
            }

            # Send emergency message to all clients before disconnecting
            for client in self.clients[:]:
                try:
                    if not client.is_closing():
                        message_data = json.dumps(emergency_message).encode("utf-8")
                        client.write(message_data + b"\n")
                        await client.drain()
                except Exception as e:
                    logger.warning("Failed to send emergency message to client: %s", e)

            # Trigger safety communication loss
            await self.safety_communication_loss(f"Emergency disconnect: {reason}")

            # Force disconnect all clients
            for client in self.clients[:]:
                try:
                    if not client.is_closing():
                        client.close()
                        await client.wait_closed()
                except Exception as e:
                    logger.warning("Error during emergency client disconnect: %s", e)

            self.clients.clear()
            self.client_heartbeats.clear()

            logger.critical("Emergency disconnect completed")

        except Exception as e:
            logger.error("Error during emergency disconnect: %s", e)
            # Still complete emergency shutdown
            self.running = False
            self.clients.clear()
            self.client_heartbeats.clear()

    async def get_safety_status_integration(self) -> dict[str, Any]:
        """[1r] Get communication status for safety status dashboard integration.

        Returns:
            Communication status formatted for safety dashboard
        """
        try:
            health_status = await self.get_communication_health_status()

            # Format for safety dashboard
            safety_status = {
                "communication_bridge": {
                    "status": "healthy" if health_status["healthy"] else "degraded",
                    "connected_clients": health_status["connected_clients"],
                    "quality_percentage": int(health_status["message_success_rate"] * 100),
                    "latency_ms": health_status["average_latency_ms"],
                    "consecutive_failures": health_status["consecutive_failures"],
                    "connection_duration": health_status["connection_duration_seconds"],
                    "last_update": datetime.now(UTC).isoformat(),
                },
                "safety_integration": {
                    "safety_manager_connected": self._safety_manager is not None,
                    "safety_timeout_threshold": self._safety_timeout,
                    "quality_threshold": self._communication_quality_threshold,
                    "callbacks_registered": len(self._connection_lost_callbacks)
                    + len(self._connection_restored_callbacks),
                },
            }

            return safety_status

        except Exception as e:
            logger.error("Error getting safety status integration: %s", e)
            return {
                "communication_bridge": {"status": "error", "error": str(e)},
                "safety_integration": {"safety_manager_connected": False},
            }

    async def shutdown(self) -> None:
        """Shutdown SDR++ bridge service following ServiceManager pattern.

        Gracefully disconnects all clients, stops the server, and cleans up resources.
        Following the same pattern as other services for ServiceManager integration.

        PRD References:
        - NFR9: MTBF >10 hours with graceful service shutdown
        - NFR1: Communication reliability with proper cleanup
        """
        logger.info("Shutting down SDR++ bridge service...")

        try:
            # Set running flag to false to stop new operations
            self.running = False

            # Disconnect all clients gracefully
            for client in self.clients[:]:  # Copy list to avoid modification during iteration
                try:
                    if not client.is_closing():
                        client.close()
                        await client.wait_closed()
                except Exception as e:
                    logger.warning("Error closing client during shutdown: %s", e)

            # Clear client tracking
            self.clients.clear()
            self.client_heartbeats.clear()

            # Stop the TCP server
            if self.server:
                self.server.close()
                await self.server.wait_closed()
                self.server = None

            logger.info("SDR++ bridge service shutdown complete")

        except Exception as e:
            logger.error("Error during SDR++ bridge service shutdown: %s", e)
            # Still complete shutdown even if errors occurred
            self.running = False
            self.clients.clear()
            self.client_heartbeats.clear()
            self.server = None
            raise
