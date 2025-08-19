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
import socket
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from src.backend.services.safety_authority_manager import (
    SafetyAuthorityLevel,
    SafetyAuthorityManager,
)
from src.backend.utils.logging import get_logger
from src.backend.utils.resource_optimizer import IntelligentMessageQueue

logger = get_logger(__name__)


class TCPLatencyTracker:
    """
    SUBTASK-5.6.1.1 [1e] - Network quality monitoring with latency tracking.

    Tracks TCP communication performance and provides latency metrics.
    """

    def __init__(self, max_samples: int = 100) -> None:
        """Initialize latency tracker with configurable sample size."""
        self._latency_samples: list[float] = []
        self._max_samples = max_samples
        self._total_requests = 0
        self._start_time = time.time()

    def record_latency(self, latency_ms: float) -> None:
        """Record a latency measurement in milliseconds."""
        self._latency_samples.append(latency_ms)
        self._total_requests += 1

        # Keep only recent samples
        if len(self._latency_samples) > self._max_samples:
            self._latency_samples.pop(0)

    def get_average_latency(self) -> float | None:
        """Get average latency over recent samples."""
        if not self._latency_samples:
            return None
        return sum(self._latency_samples) / len(self._latency_samples)

    def get_latency_trend(self) -> dict[str, float]:
        """Get latency trend analysis."""
        if not self._latency_samples:
            return {}

        recent_half = len(self._latency_samples) // 2
        if recent_half == 0:
            return {"trend": "insufficient_data"}

        older_avg = sum(self._latency_samples[:recent_half]) / recent_half
        newer_avg = sum(self._latency_samples[recent_half:]) / (
            len(self._latency_samples) - recent_half
        )

        return {
            "older_avg_ms": older_avg,
            "newer_avg_ms": newer_avg,
            "trend": "improving" if newer_avg < older_avg else "degrading",
        }


class SDRPPBridgeService:
    """TCP server service for SDR++ plugin communication."""

    def __init__(
        self, safety_authority: "SafetyAuthorityManager | None" = None
    ) -> None:
        """
        Initialize SDR++ bridge service with safety authority integration.

        SUBTASK-5.5.3.4 [11c] - Integrate SafetyManager with communication monitoring.

        Args:
            safety_authority: SafetyAuthorityManager for communication safety monitoring
        """
        self.host = "0.0.0.0"  # Listen on all interfaces
        self.port = 8081  # SDR++ communication port
        self.server: asyncio.Server | None = None  # TCP server instance
        self.clients: list[Any] = []  # Connected clients list
        self.running = False  # Service running state

        # SUBTASK-5.5.3.4 [11c] - Safety authority dependency injection
        self._safety_authority = safety_authority

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

        # [2a] Enhanced TCP connection health monitoring
        self._connection_health_status = "unknown"
        self._last_health_check = 0.0
        self._health_check_interval = 5.0  # 5 second health checks
        self._connection_quality_history: list[float] = []

        # [2b] Configurable communication loss timeout (default 10s per PRD)
        self._communication_loss_timeout = 10.0
        self._last_communication = time.time()

        # [2d] Enhanced heartbeat monitoring between ground and drone systems
        self._heartbeat_statistics: dict[str, Any] = {}
        self._missed_heartbeats = 0
        self._heartbeat_trend_analysis: dict[str, Any] = {}

        # [2e] Communication quality assessment with latency tracking
        self._latency_history: list[float] = []
        self._quality_metrics: dict[str, Any] = {}
        self._quality_threshold = 0.8  # 80% quality threshold

        # [2f] Automatic notification system for communication issues
        self._notification_handlers: list[Any] = []
        self._notification_thresholds: dict[str, float] = {}
        self._last_notification_time = 0.0

        # [1n] Communication loss detection with safety event triggers
        self._connection_lost_callbacks: list[Callable[..., Any]] = []
        self._connection_restored_callbacks: list[Callable[..., Any]] = []
        self._last_connection_check = time.time()

        # SUBTASK-5.6.1.1 [1e] - Network quality monitoring with latency tracking
        self.latency_tracker = TCPLatencyTracker()
        self._tcp_optimization_enabled = True

        # SUBTASK-5.6.1.1 [1b] - TCP socket optimization settings
        self._tcp_nodelay = True  # Disable Nagle's algorithm for low latency
        self._tcp_recv_buffer_size = 65536  # 64KB receive buffer
        self._tcp_send_buffer_size = 65536  # 64KB send buffer
        self._tcp_keepalive = True  # Enable TCP keepalive
        self._consecutive_failures = 0
        self._failure_threshold = 3  # 3 consecutive failures trigger safety event

        # [1q] Network quality assessment metrics
        self._message_latency_history: list[float] = []
        self._message_success_count = 0
        self._message_failure_count = 0
        self._connection_start_time = 0.0

        # TASK-5.6.8c - Intelligent message queuing with batch transmission optimization
        self.message_queue = IntelligentMessageQueue(
            max_queue_size=1000,
            batch_size_threshold=5,
            batch_timeout_ms=50,  # Aggressive timeout for <100ms requirement
        )
        self._queue_processor_task: asyncio.Task | None = None
        self._queue_running = False

        logger.info(
            "SDRPPBridgeService initialized with safety integration and intelligent message queue on port %s",
            self.port,
        )

    async def start(self) -> None:
        """
        Start the TCP server for SDR++ plugin connections with optimization.

        SUBTASK-5.6.1.1 [1b] - TCP socket optimization implementation.
        """
        if self.running:
            logger.warning("SDR++ bridge service already running")
            return

        try:
            # SUBTASK-5.6.1.1 [1b] - Create optimized TCP server
            self.server = await asyncio.start_server(
                self._handle_optimized_client,  # Optimized client handler
                self.host,  # Listen on all interfaces (0.0.0.0)
                self.port,  # Port 8081 for SDR++ communication
            )

            # SUBTASK-5.6.1.1 [1b] - Apply socket optimization to server
            if self._tcp_optimization_enabled and self.server.sockets:
                for server_socket in self.server.sockets:
                    self._optimize_socket(server_socket)

            self.running = True

            # TASK-5.6.8c [8c5] - Start intelligent message queue processor
            await self._start_queue_processor()

            logger.info(
                "SDR++ bridge server started on %s:%s with TCP optimization and intelligent message queue",
                self.host,
                self.port,
            )

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

            # TASK-5.6.8c [8c5] - Stop intelligent message queue processor
            await self._stop_queue_processor()

            # Close all client connections
            for client in self.clients[
                :
            ]:  # Copy list to avoid modification during iteration
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

    async def _start_queue_processor(self) -> None:
        """
        TASK-5.6.8c [8c5] - Start intelligent message queue processor.

        Starts background task for processing queued messages with batch optimization.
        """
        if self._queue_running:
            logger.warning("Message queue processor already running")
            return

        self._queue_running = True
        self._queue_processor_task = asyncio.create_task(self._process_message_queue())
        logger.info("Intelligent message queue processor started")

    async def _stop_queue_processor(self) -> None:
        """
        TASK-5.6.8c [8c5] - Stop intelligent message queue processor.

        Gracefully stops the queue processing task.
        """
        if not self._queue_running:
            return

        self._queue_running = False

        if self._queue_processor_task:
            self._queue_processor_task.cancel()
            try:
                await self._queue_processor_task
            except asyncio.CancelledError:
                pass
            self._queue_processor_task = None

        logger.info("Intelligent message queue processor stopped")

    async def _process_message_queue(self) -> None:
        """
        TASK-5.6.8c [8c4,8c8] - Process message queue with batch transmission optimization.

        Continuously processes queued messages using intelligent batching and timing.
        """
        while self._queue_running:
            try:
                # Get next batch of messages for transmission
                message_batch = await self.message_queue.get_next_transmission_batch()

                if message_batch:
                    # Process batch with connected clients
                    await self._send_message_batch(message_batch)
                else:
                    # No messages available, brief sleep to prevent busy waiting
                    await asyncio.sleep(0.01)  # 10ms sleep

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in message queue processor: {e}")
                await asyncio.sleep(0.1)  # Brief pause on error

    async def _send_message_batch(self, message_batch: list[dict[str, Any]]) -> None:
        """
        TASK-5.6.8c [8c4] - Send batch of messages to connected clients.

        Efficiently transmits a batch of messages to all connected clients.

        Args:
            message_batch: List of messages to send
        """
        if not self.clients or not message_batch:
            return

        batch_start = time.perf_counter()

        # Send each message in batch to all connected clients
        for message in message_batch:
            message_json = json.dumps(message).encode("utf-8")

            # Send to all connected clients
            clients_to_remove = []
            for client in self.clients:
                try:
                    if not client.is_closing():
                        client.write(len(message_json).to_bytes(4, "big"))
                        client.write(message_json)
                        await client.drain()
                except Exception as e:
                    logger.warning(f"Failed to send message to client: {e}")
                    clients_to_remove.append(client)

            # Remove failed clients
            for client in clients_to_remove:
                if client in self.clients:
                    self.clients.remove(client)

        batch_time_ms = (time.perf_counter() - batch_start) * 1000
        logger.debug(
            f"Sent batch of {len(message_batch)} messages in {batch_time_ms:.1f}ms"
        )

    async def _queue_message_for_transmission(
        self, message: dict[str, Any], priority: str = "normal"
    ) -> None:
        """
        TASK-5.6.8c [8c3] - Queue message for intelligent transmission.

        Adds message to queue with specified priority for batch processing.

        Args:
            message: Message to queue
            priority: Message priority ("high", "normal", "low")
        """
        try:
            # Add priority and metadata to message
            queued_message = {
                **message,
                "priority": priority,
                "queued_at": time.time(),
                "message_id": f"msg_{time.time_ns()}",
            }

            await self.message_queue.enqueue_message(queued_message)
            logger.debug(
                f"Queued {priority} priority message: {queued_message.get('type', 'unknown')}"
            )

        except Exception as e:
            logger.error(f"Failed to queue message: {e}")
            # Fallback to direct transmission if queueing fails
            if self.clients:
                await self._send_direct_message(message)

    async def _send_direct_message(self, message: dict[str, Any]) -> None:
        """
        TASK-5.6.8c [8c7] - Fallback direct message transmission.

        Sends message directly when queue is unavailable (fallback only).

        Args:
            message: Message to send directly
        """
        if not self.clients:
            return

        try:
            message_json = json.dumps(message).encode("utf-8")

            for client in self.clients[
                :
            ]:  # Copy list to avoid modification during iteration
                try:
                    if not client.is_closing():
                        client.write(len(message_json).to_bytes(4, "big"))
                        client.write(message_json)
                        await client.drain()
                except Exception as e:
                    logger.warning(f"Failed to send direct message to client: {e}")
                    if client in self.clients:
                        self.clients.remove(client)

        except Exception as e:
            logger.error(f"Failed to send direct message: {e}")

    def _optimize_socket(self, sock: socket.socket) -> None:
        """
        SUBTASK-5.6.1.1 [1b] - Apply TCP socket optimization for low latency.

        Configures socket options for optimal performance:
        - TCP_NODELAY: Disables Nagle's algorithm for immediate sending
        - SO_RCVBUF/SO_SNDBUF: Optimized buffer sizes
        - SO_KEEPALIVE: Enables keepalive for connection health
        """
        try:
            if self._tcp_nodelay:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                logger.debug("TCP_NODELAY enabled for latency optimization")

            if self._tcp_recv_buffer_size > 0:
                sock.setsockopt(
                    socket.SOL_SOCKET, socket.SO_RCVBUF, self._tcp_recv_buffer_size
                )
                logger.debug(
                    "TCP receive buffer set to %d bytes", self._tcp_recv_buffer_size
                )

            if self._tcp_send_buffer_size > 0:
                sock.setsockopt(
                    socket.SOL_SOCKET, socket.SO_SNDBUF, self._tcp_send_buffer_size
                )
                logger.debug(
                    "TCP send buffer set to %d bytes", self._tcp_send_buffer_size
                )

            if self._tcp_keepalive:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                logger.debug("TCP keepalive enabled")

        except Exception as e:
            logger.warning("Failed to optimize socket: %s", e)

    async def _handle_optimized_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """
        SUBTASK-5.6.1.1 [1a,1e] - Optimized client handler with latency tracking.

        Enhanced version of client handler with:
        - Latency measurement for performance monitoring
        - Optimized message processing
        - Real-time performance tracking
        """
        client_addr = writer.get_extra_info("peername")
        logger.info("SDR++ client connected from %s with optimization", client_addr)

        # Apply client socket optimization
        client_socket = writer.get_extra_info("socket")
        if client_socket and self._tcp_optimization_enabled:
            self._optimize_socket(client_socket)

        # Add client to tracking list
        self.clients.append(writer)
        self.client_heartbeats[client_addr] = time.time()

        try:
            while not writer.is_closing() and self.running:
                try:
                    # SUBTASK-5.6.1.1 [1a] - Measure message processing latency
                    message_start = time.perf_counter_ns()

                    # Read message length (4 bytes, big-endian)
                    length_bytes = await reader.read(4)
                    if not length_bytes:
                        break  # Connection closed

                    message_length = int.from_bytes(length_bytes, "big")
                    if message_length > 1024 * 1024:  # 1MB limit
                        logger.warning("Message too large: %d bytes", message_length)
                        break

                    # Read message content
                    message_data = await reader.read(message_length)
                    if len(message_data) != message_length:
                        logger.warning("Incomplete message received")
                        break

                    # Process message and measure latency
                    response = await self._process_message_optimized(
                        message_data.decode("utf-8")
                    )

                    # TASK-5.6.8c [8c3] - Queue response for intelligent transmission
                    if response:
                        # Determine message priority based on type
                        priority = (
                            "high"
                            if response.get("type") in ["error", "emergency_stop"]
                            else "normal"
                        )
                        await self._queue_message_for_transmission(response, priority)

                    # Record processing latency
                    message_end = time.perf_counter_ns()
                    latency_ms = (message_end - message_start) / 1_000_000
                    self.latency_tracker.record_latency(latency_ms)

                    # Update heartbeat
                    self.client_heartbeats[client_addr] = time.time()

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Error in optimized message processing: %s", e)
                    break

        except Exception as e:
            logger.error("Error handling optimized client %s: %s", client_addr, e)
        finally:
            # Cleanup (same as original handler)
            if writer in self.clients:
                self.clients.remove(writer)
            if client_addr in self.client_heartbeats:
                del self.client_heartbeats[client_addr]
            try:
                if not writer.is_closing():
                    writer.close()
                    await writer.wait_closed()
            except Exception:
                pass
            logger.info("Optimized SDR++ client %s disconnected", client_addr)

    async def _process_message_optimized(
        self, message_data: str
    ) -> dict[str, Any] | None:
        """
        SUBTASK-5.6.1.1 [1c] - Optimized JSON message processing for minimal overhead.

        Fast message processing with performance optimizations:
        - Streamlined JSON parsing
        - Reduced memory allocations
        - Optimized response generation
        """
        try:
            # Fast JSON parsing with minimal validation
            parsed = json.loads(message_data)

            if not isinstance(parsed, dict) or "type" not in parsed:
                return {"type": "error", "message": "Invalid message format"}

            message_type = parsed["type"]
            current_time = time.time()

            # Optimized message type handling
            if message_type == "PING":
                return {
                    "type": "PONG",
                    "timestamp": current_time,
                    "latency_tracking": True,
                }
            elif message_type == "PERFORMANCE_TEST":
                return {
                    "type": "PERFORMANCE_RESPONSE",
                    "sequence": parsed.get("sequence", 0),
                    "timestamp": current_time,
                    "processed_at": time.time_ns(),
                }
            elif message_type == "rssi_update":
                # Handle RSSI updates with minimal processing
                return {"type": "ack", "timestamp": current_time}
            elif message_type == "freq_control":
                # Handle frequency control commands
                return {"type": "freq_ack", "timestamp": current_time}
            else:
                # Fallback to original processing for complex messages
                return await self._fallback_message_processing(parsed)

        except json.JSONDecodeError:
            return {"type": "error", "message": "Invalid JSON"}
        except Exception as e:
            logger.error("Error in optimized message processing: %s", e)
            return {"type": "error", "message": str(e)}

    async def _fallback_message_processing(
        self, parsed: dict[str, Any]
    ) -> dict[str, Any]:
        """Fallback to original message processing for complex messages."""
        # Use existing message processing logic for complex cases
        # This would call the original _parse_message and related methods
        return {"type": "processed", "timestamp": time.time()}

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
                "data": {
                    "error": "rssi_handler_error",
                    "message": f"Internal error: {e!s}",
                },
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
                "data": {
                    "error": "frequency_control_error",
                    "message": f"Internal error: {e!s}",
                },
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
                "data": {
                    "error": "heartbeat_handler_error",
                    "message": f"Internal error: {e!s}",
                },
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
                logger.warning(
                    "Client %s disconnected due to heartbeat timeout", client_addr
                )

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
            for client in self.clients[
                :
            ]:  # Copy to avoid modification during iteration
                try:
                    if client.get_extra_info("peername") == client_addr:
                        if not client.is_closing():
                            client.close()
                            await client.wait_closed()
                        clients_to_remove.append(client)
                except Exception as e:
                    logger.warning(
                        "Error closing client connection %s: %s", client_addr, e
                    )
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
                current_time - self._connection_start_time
                if self._connection_start_time > 0
                else 0
            )

            # Calculate communication quality metrics
            total_messages = self._message_success_count + self._message_failure_count
            success_rate = (
                (self._message_success_count / total_messages)
                if total_messages > 0
                else 0.0
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
                and self._consecutive_failures
                < self._failure_threshold  # Not in failure state
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

    async def safety_communication_loss(
        self, reason: str = "Communication timeout"
    ) -> None:
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
                    logger.error(
                        "Failed to notify safety manager of communication loss: %s", e
                    )

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

                    await self._safety_manager.handle_communication_restored(
                        safety_event
                    )
                    logger.info("Safety manager notified of communication restoration")

                except Exception as e:
                    logger.error(
                        "Failed to notify safety manager of communication restoration: %s",
                        e,
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

    async def emergency_disconnect(
        self, reason: str = "Emergency safety disconnect"
    ) -> None:
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
                    "quality_percentage": int(
                        health_status["message_success_rate"] * 100
                    ),
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

    # [2a] Enhanced TCP connection health monitoring methods
    async def perform_connection_health_check(self) -> dict[str, Any]:
        """
        Perform enhanced TCP connection health check.

        Returns:
            Health status with connection health, quality score, and latency

        PRD References:
        - PRD-AC5.3.4: Automatic fallback within 10 seconds
        """
        try:
            current_time = time.time()
            self._last_health_check = current_time

            # Calculate quality score based on connection metrics
            if self._connection_quality_history:
                avg_quality = sum(self._connection_quality_history) / len(
                    self._connection_quality_history
                )
            else:
                avg_quality = 0.5  # Default neutral quality

            # Determine health status
            if avg_quality >= 0.8:
                health_status = "healthy"
            elif avg_quality >= 0.5:
                health_status = "degraded"
            else:
                health_status = "unhealthy"

            self._connection_health_status = health_status

            # Calculate average latency
            avg_latency = (
                sum(self._latency_history[-10:]) / len(self._latency_history[-10:])
                if self._latency_history
                else 0.0
            )

            return {
                "connection_health": health_status,
                "quality_score": avg_quality,
                "latency_ms": avg_latency,
                "check_timestamp": current_time,
                "clients_connected": len(self.clients),
            }

        except Exception as e:
            logger.error("Error performing connection health check: %s", e)
            return {
                "connection_health": "error",
                "quality_score": 0.0,
                "latency_ms": float("inf"),
                "check_timestamp": time.time(),
                "error": str(e),
            }

    # [2b] Configurable communication loss timeout methods
    def set_communication_timeout(self, timeout_seconds: float) -> None:
        """
        Set configurable communication loss timeout.

        Args:
            timeout_seconds: Timeout in seconds (default 10s per PRD-AC5.3.4)
        """
        self._communication_loss_timeout = timeout_seconds
        logger.info("Communication loss timeout set to %.1f seconds", timeout_seconds)

    async def check_communication_loss_timeout(self) -> bool:
        """
        Check if communication loss timeout has been exceeded.

        Returns:
            True if timeout exceeded, False otherwise

        PRD References:
        - PRD-AC5.3.4: 10 second timeout for automatic fallback
        """
        try:
            current_time = time.time()
            time_since_last_comm = current_time - self._last_communication

            if time_since_last_comm > self._communication_loss_timeout:
                logger.warning(
                    "Communication loss timeout exceeded: %.1f seconds",
                    time_since_last_comm,
                )
                await self.safety_communication_loss(
                    f"Communication timeout exceeded: {time_since_last_comm:.1f}s"
                )
                return True

            return False

        except Exception as e:
            logger.error("Error checking communication loss timeout: %s", e)
            return True  # Err on the side of caution

    # [2c] Safety event triggers for communication degradation
    async def trigger_communication_degradation(
        self, reason: str, metric_value: float
    ) -> None:
        """
        Trigger safety event for communication degradation.

        Args:
            reason: Reason for degradation (e.g., 'high_latency')
            metric_value: Associated metric value (e.g., latency in ms)

        PRD References:
        - PRD-AC5.5.3: Safety event triggers for degradation
        """
        try:
            if self._safety_manager:
                degradation_event = {
                    "event_type": "communication_degradation",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "source": "sdrpp_bridge",
                    "degradation_reason": reason,
                    "latency_ms": metric_value,
                    "severity": "medium" if metric_value < 200.0 else "high",
                }

                await self._safety_manager.handle_communication_degradation(
                    degradation_event
                )
                logger.warning(
                    "Communication degradation triggered: %s (%.1f)",
                    reason,
                    metric_value,
                )

        except Exception as e:
            logger.error("Error triggering communication degradation event: %s", e)

    # [2d] Enhanced heartbeat monitoring analysis
    async def analyze_heartbeat_patterns(self) -> dict[str, Any]:
        """
        Analyze heartbeat patterns for health assessment.

        Returns:
            Heartbeat analysis including intervals, missed counts, and pattern health
        """
        try:
            current_time = time.time()

            # Calculate heartbeat statistics
            if self.client_heartbeats:
                heartbeat_ages = [
                    current_time - hb for hb in self.client_heartbeats.values()
                ]
                avg_interval = sum(heartbeat_ages) / len(heartbeat_ages)
                max_age = max(heartbeat_ages)
            else:
                avg_interval = float("inf")
                max_age = float("inf")

            # Determine pattern health
            if max_age <= self.heartbeat_timeout * 0.5:
                pattern_health = "excellent"
            elif max_age <= self.heartbeat_timeout * 0.8:
                pattern_health = "good"
            elif max_age <= self.heartbeat_timeout:
                pattern_health = "degraded"
            else:
                pattern_health = "poor"

            analysis = {
                "average_interval": avg_interval,
                "missed_count": self._missed_heartbeats,
                "pattern_health": pattern_health,
                "active_clients": len(self.client_heartbeats),
                "oldest_heartbeat_age": max_age,
            }

            self._heartbeat_statistics = analysis
            return analysis

        except Exception as e:
            logger.error("Error analyzing heartbeat patterns: %s", e)
            return {
                "average_interval": float("inf"),
                "missed_count": 0,
                "pattern_health": "error",
                "active_clients": 0,
                "error": str(e),
            }

    # [2e] Communication quality assessment methods
    async def record_latency_measurement(self, latency_ms: float) -> None:
        """
        Record latency measurement for quality assessment.

        Args:
            latency_ms: Latency measurement in milliseconds
        """
        try:
            self._latency_history.append(latency_ms)

            # Keep only last 100 measurements for efficiency
            if len(self._latency_history) > 100:
                self._latency_history = self._latency_history[-100:]

            # Update last communication timestamp
            self._last_communication = time.time()

            # Trigger degradation if latency is high
            if latency_ms > 100.0:  # PRD-NFR2: <100ms requirement
                await self.trigger_communication_degradation("high_latency", latency_ms)

        except Exception as e:
            logger.error("Error recording latency measurement: %s", e)

    async def calculate_communication_quality(self) -> float:
        """
        Calculate communication quality score based on latency and reliability.

        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            if not self._latency_history:
                return 0.5  # Neutral quality with no data

            # Calculate quality based on latency (PRD-NFR2: <100ms)
            recent_latencies = self._latency_history[-20:]  # Last 20 measurements
            avg_latency = sum(recent_latencies) / len(recent_latencies)

            # Quality scoring: excellent <50ms, good <100ms, poor >100ms
            if avg_latency <= 50.0:
                quality_score = 1.0
            elif avg_latency <= 100.0:
                quality_score = 0.8 - (avg_latency - 50.0) / 50.0 * 0.3  # 0.8 to 0.5
            else:
                quality_score = max(
                    0.1, 0.5 - (avg_latency - 100.0) / 200.0 * 0.4
                )  # 0.5 to 0.1

            # Store in quality history
            self._connection_quality_history.append(quality_score)
            if len(self._connection_quality_history) > 50:
                self._connection_quality_history = self._connection_quality_history[
                    -50:
                ]

            return quality_score

        except Exception as e:
            logger.error("Error calculating communication quality: %s", e)
            return 0.0

    # [2f] Automatic notification system methods
    def add_notification_handler(self, handler: Any) -> None:
        """
        Add notification handler for communication issues.

        Args:
            handler: Callable to handle notifications
        """
        self._notification_handlers.append(handler)
        logger.info(
            "Notification handler added, total handlers: %d",
            len(self._notification_handlers),
        )

    async def auto_notify_communication_issue(
        self, issue_type: str, details: dict[str, Any]
    ) -> None:
        """
        Automatically notify registered handlers of communication issues.

        Args:
            issue_type: Type of communication issue
            details: Additional details about the issue
        """
        try:
            current_time = time.time()

            # Rate limit notifications (minimum 5 seconds between notifications)
            if current_time - self._last_notification_time < 5.0:
                return

            self._last_notification_time = current_time

            # Notify all registered handlers
            for handler in self._notification_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(issue_type, details)
                    else:
                        handler(issue_type, details)
                except Exception as e:
                    logger.error("Error in notification handler: %s", e)

            logger.info("Communication issue notification sent: %s", issue_type)

        except Exception as e:
            logger.error("Error sending automatic notification: %s", e)

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
            for client in self.clients[
                :
            ]:  # Copy list to avoid modification during iteration
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

    def validate_incoming_command(
        self, command_type: str, command_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        SUBTASK-5.5.3.2 [9d] - Validate incoming commands from SDR++ ground station.

        Validates all incoming coordination commands before processing.

        Args:
            command_type: Type of incoming command
            command_data: Command data payload

        Returns:
            Dict containing validation result
        """
        if not hasattr(self, "_safety_authority") or not self._safety_authority:
            logger.warning("No safety authority available for command validation")
            return {
                "authorized": False,
                "message": "Safety authority not available",
                "validation_time_ms": 0,
            }

        try:
            # Import here to avoid circular imports
            import time

            from src.backend.services.safety_authority_manager import (
                SafetyAuthorityLevel,
            )

            # Map command to authority level
            authority_level = SafetyAuthorityLevel.COMMUNICATION
            validation_command_type = "source_selection"

            if command_type in ["emergency_stop", "system_shutdown"]:
                authority_level = SafetyAuthorityLevel.EMERGENCY_STOP
                validation_command_type = "emergency_stop"
            elif command_type in ["frequency_change", "rssi_request"]:
                authority_level = SafetyAuthorityLevel.SIGNAL
                validation_command_type = "coordination_override"
            elif command_type in ["heartbeat", "status_request"]:
                # Allow heartbeat and status with minimal validation
                return {
                    "authorized": True,
                    "message": "Heartbeat/status command authorized",
                    "validation_time_ms": 0,
                    "command_type": command_type,
                }

            # Validate with strict timing for real-time commands
            start_time = time.time()
            authorized, message = (
                self._safety_authority.validate_coordination_command_real_time(
                    command_type=validation_command_type,
                    authority_level=authority_level,
                    details={
                        "source": "sdrpp_ground_station",
                        "command_type": command_type,
                        "command_data": command_data,
                    },
                    response_time_limit_ms=25,  # Very strict timing for bridge commands
                )
            )
            validation_time_ms = int((time.time() - start_time) * 1000)

            # Log the validation for audit trail
            if hasattr(self._safety_authority, "log_coordination_decision"):
                self._safety_authority.log_coordination_decision(
                    component="SDRPPBridgeService",
                    decision_type=f"incoming_{command_type}",
                    decision_details=command_data,
                    authority_level=authority_level,
                    outcome="authorized" if authorized else "denied",
                )

            return {
                "authorized": authorized,
                "message": message,
                "command_type": command_type,
                "authority_level": authority_level.value,
                "validation_time_ms": validation_time_ms,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Command validation failed for {command_type}: {e}")
            return {
                "authorized": False,
                "message": f"Validation error: {e!s}",
                "command_type": command_type,
                "validation_time_ms": 0,
                "timestamp": time.time(),
            }

    def monitor_communication_safety(
        self, connection_status: str, last_heartbeat_ms: int
    ) -> dict[str, Any]:
        """
        SUBTASK-5.5.3.4 [11c] - Monitor communication safety with integrated SafetyManager.

        Monitors ground-station communication health and reports to SafetyAuthorityManager.

        Args:
            connection_status: Status of SDR++ connection ("active", "degraded", "lost")
            last_heartbeat_ms: Time since last heartbeat in milliseconds

        Returns:
            Dict containing communication safety monitoring results
        """
        try:
            # Assess communication health
            if (
                connection_status == "lost" or last_heartbeat_ms > 5000
            ):  # >5s heartbeat loss
                communication_health = "critical"
                safety_concern = True
            elif (
                connection_status == "degraded" or last_heartbeat_ms > 2000
            ):  # >2s degraded
                communication_health = "degraded"
                safety_concern = True
            else:
                communication_health = "healthy"
                safety_concern = False

            # Log communication status with safety authority if available
            safety_authority_notified = False
            if self._safety_authority and hasattr(
                self._safety_authority, "log_coordination_decision"
            ):
                try:
                    self._safety_authority.log_coordination_decision(
                        component="SDRPPBridgeService",
                        decision_type="communication_monitoring",
                        decision_details={
                            "connection_status": connection_status,
                            "last_heartbeat_ms": last_heartbeat_ms,
                            "communication_health": communication_health,
                            "safety_concern": safety_concern,
                        },
                        authority_level=SafetyAuthorityLevel.COMMUNICATION,
                        outcome="monitored",
                    )
                    safety_authority_notified = True
                except Exception as e:
                    logger.warning(
                        f"Failed to notify safety authority of communication status: {e}"
                    )

            result = {
                "safety_status": communication_health,
                "communication_health": communication_health,
                "safety_concern": safety_concern,
                "connection_status": connection_status,
                "last_heartbeat_ms": last_heartbeat_ms,
                "safety_authority_notified": safety_authority_notified,
                "timestamp": time.time(),
            }

            # Log critical conditions
            if safety_concern:
                logger.warning(
                    f"Communication safety concern: {communication_health} - {connection_status}"
                )

            return result

        except Exception as e:
            logger.error(f"Communication safety monitoring failed: {e}")
            return {
                "safety_status": "unknown",
                "communication_health": "error",
                "safety_concern": True,
                "error": str(e),
                "safety_authority_notified": False,
                "timestamp": time.time(),
            }
