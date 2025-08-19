"""
WebSocket handler for real-time updates.
"""

import asyncio
import json
from datetime import datetime
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.backend.constants.websocket_messages import (
    WebSocketMessageTypes,
)
from src.backend.core.config import get_config
from src.backend.core.exceptions import (
    PISADException,
    SafetyInterlockError,
    SignalProcessingError,
    StateTransitionError,
)
from src.backend.services.mavlink_service import ConnectionState, MAVLinkService
from src.backend.services.signal_processor_integration import SignalProcessorIntegration
from src.backend.services.state_machine import StateMachine, SystemState
from src.backend.utils.logging import get_logger
from src.backend.utils.safety import SafetyInterlockSystem

logger = get_logger(__name__)

router = APIRouter()

# Global signal processor integration instance
signal_processor_integration: SignalProcessorIntegration | None = None

# Global MAVLink service instance
mavlink_service: MAVLinkService | None = None

# Global safety interlock system instance
safety_system: SafetyInterlockSystem | None = None

# Global state machine instance
state_machine: StateMachine | None = None

# Global reference to broadcast tasks
_rssi_broadcast_task: asyncio.Task[None] | None = None
_telemetry_broadcast_task: asyncio.Task[None] | None = None
_safety_broadcast_task: asyncio.Task[None] | None = None
_homing_broadcast_task: asyncio.Task[None] | None = None
_state_broadcast_task: asyncio.Task[None] | None = None

# Set to track active WebSocket connections
active_connections: set[WebSocket] = set()


class ConnectionManager:
    """Manages WebSocket connections and message broadcasting."""

    def __init__(self):
        self.active_connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Accept new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info(
            f"WebSocket client connected. Total connections: {len(self.active_connections)}"
        )

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove WebSocket connection."""
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.info(
            f"WebSocket client disconnected. Total connections: {len(self.active_connections)}"
        )

    async def broadcast_json(self, message: dict[str, Any]) -> None:
        """Broadcast JSON message to all connected clients."""
        if not self.active_connections:
            return

        disconnected: set[WebSocket] = set()
        message_str = json.dumps(message)

        # Send to all connections
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.debug(f"Failed to send to client: {e}")
                disconnected.add(connection)

        # Remove disconnected clients
        async with self._lock:
            self.active_connections -= disconnected

    async def broadcast_bytes(self, data: bytes) -> None:
        """Broadcast binary data to all connected clients."""
        if not self.active_connections:
            return

        disconnected: set[WebSocket] = set()

        # Send to all connections
        for connection in self.active_connections:
            try:
                await connection.send_bytes(data)
            except Exception as e:
                logger.debug(f"Failed to send to client: {e}")
                disconnected.add(connection)

        # Remove disconnected clients
        async with self._lock:
            self.active_connections -= disconnected


# Global connection manager
manager = ConnectionManager()


async def get_signal_processor() -> SignalProcessorIntegration:
    """Get or create signal processor integration instance."""
    global signal_processor_integration
    global _rssi_broadcast_task

    if signal_processor_integration is None:
        signal_processor_integration = SignalProcessorIntegration()
        # Start the integration service
        await signal_processor_integration.start()
        # Start RSSI broadcast task and store reference
        _rssi_broadcast_task = asyncio.create_task(broadcast_rssi_updates())

    return signal_processor_integration


async def get_mavlink_service() -> MAVLinkService:
    """Get or create MAVLink service instance."""
    global mavlink_service
    global _telemetry_broadcast_task

    if mavlink_service is None:
        config = get_config()
        # Determine device path based on configuration
        # Default to TCP for SITL if not configured
        device_path = getattr(config, "mavlink_device", "tcp:127.0.0.1:5760")
        baud_rate = getattr(config, "mavlink_baud", 115200)

        mavlink_service = MAVLinkService(device_path=device_path, baud_rate=baud_rate)

        # Add state change callback for connection status updates
        mavlink_service.add_state_callback(
            lambda state: asyncio.create_task(broadcast_mavlink_status(state))
        )

        # Start the MAVLink service
        await mavlink_service.start()

        # Start telemetry broadcast task
        _telemetry_broadcast_task = asyncio.create_task(broadcast_telemetry_updates())

    return mavlink_service


async def broadcast_mavlink_status(state: ConnectionState):
    """Broadcast MAVLink connection status changes."""
    message = {
        "type": "mavlink_status",
        "data": {
            "connected": state == ConnectionState.CONNECTED,
            "state": state.value,
            "timestamp": datetime.utcnow().isoformat(),
        },
    }
    await manager.broadcast_json(message)


async def broadcast_telemetry_updates():
    """Broadcast telemetry updates to all connected WebSocket clients at 2Hz."""
    try:
        service = await get_mavlink_service()

        while True:
            if service.is_connected():
                telemetry = service.get_telemetry()

                # Get safety status if available
                safety_interlocks = {}
                if safety_system:
                    safety_status = safety_system.get_safety_status()
                    safety_interlocks = {
                        name: check["is_safe"]
                        for name, check in safety_status["checks"].items()
                    }

                # Format telemetry update message
                message = {
                    "type": "telemetry",
                    "data": {
                        "position": telemetry["position"],
                        "battery": telemetry["battery"]["percentage"],
                        "flightMode": telemetry["flight_mode"],
                        "velocity": {
                            "forward": 0.0,
                            "yaw": 0.0,
                        },  # Placeholder for actual velocity
                        "gpsStatus": service.get_gps_status_string(),
                        "armed": telemetry["armed"],
                        "safetyInterlocks": safety_interlocks,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                }

                # Broadcast to all clients
                await manager.broadcast_json(message)

            # 2Hz update rate (500ms intervals)
            await asyncio.sleep(0.5)

    except PISADException as e:
        logger.error(f"Error in telemetry broadcast task: {e}")


async def get_state_machine() -> StateMachine:
    """Get or create state machine instance."""
    global state_machine
    global _state_broadcast_task

    if state_machine is None:
        state_machine = StateMachine()

        # Add state change callback for state updates
        async def state_change_callback(
            old_state: SystemState, new_state: SystemState, reason: str | None
        ):
            await broadcast_state_change(old_state, new_state, reason)

        state_machine.add_state_callback(state_change_callback)

        # Start state machine
        await state_machine.start()

        # Start state broadcast task
        _state_broadcast_task = asyncio.create_task(broadcast_state_updates())

    return state_machine


async def broadcast_state_change(
    old_state: SystemState, new_state: SystemState, reason: str | None
):
    """Broadcast state change event to all connected clients."""
    if state_machine is None:
        return

    # Get allowed transitions for the new state
    allowed_transitions = state_machine.get_allowed_transitions()

    # Get state duration (time since last transition)
    state_machine.get_statistics()

    message = {
        "type": "state_change",
        "data": {
            "from_state": old_state.value,
            "to_state": new_state.value,
            "current_state": new_state.value,
            "reason": reason,
            "allowed_transitions": [s.value for s in allowed_transitions],
            "timestamp": datetime.utcnow().isoformat(),
        },
    }

    await manager.broadcast_json(message)
    logger.info(f"Broadcast state change: {old_state.value} -> {new_state.value}")


async def broadcast_state_updates():
    """Broadcast periodic state updates with full context at 1Hz."""
    try:
        sm = await get_state_machine()

        while True:
            # Get comprehensive state information
            current_state = sm.get_current_state()
            allowed_transitions = sm.get_allowed_transitions()
            statistics = sm.get_statistics()
            history = sm.get_state_history(limit=10)
            search_status = sm.get_search_pattern_status()

            # Calculate state duration
            state_duration_ms = None
            if history and len(history) > 0:
                # Get time since last state change
                last_change = history[0]
                if last_change.get("timestamp"):
                    try:
                        last_time = datetime.fromisoformat(last_change["timestamp"])
                        duration = datetime.utcnow() - last_time.replace(tzinfo=None)
                        state_duration_ms = duration.total_seconds() * 1000
                    except Exception:
                        pass

            # Format comprehensive state update
            message = {
                "type": "state",
                "data": {
                    "current_state": current_state.value,
                    "previous_state": sm._previous_state.value,
                    "allowed_transitions": [s.value for s in allowed_transitions],
                    "state_duration_ms": state_duration_ms,
                    "history": history,
                    "search_status": search_status,
                    "homing_enabled": statistics.get("homing_enabled", False),
                    "detection_count": statistics.get("detection_count", 0),
                    "time_since_detection": statistics.get("time_since_detection"),
                    "timestamp": datetime.utcnow().isoformat(),
                },
            }

            # Broadcast to all clients
            await manager.broadcast_json(message)

            # 1Hz update rate
            await asyncio.sleep(1.0)

    except StateTransitionError as e:
        logger.error(f"Error in state broadcast task: {e}")


async def broadcast_rssi_updates():
    """Broadcast RSSI updates to all connected WebSocket clients at 10Hz."""
    config = get_config()
    update_interval = (
        config.websocket.WS_RSSI_UPDATE_INTERVAL_MS / 1000.0
    )  # Convert to seconds

    try:
        processor = await get_signal_processor()

        async for rssi_reading in processor.get_rssi_stream():
            # Format RSSI update message
            message = {
                "type": "rssi",
                "data": {
                    "rssi": rssi_reading.rssi,
                    "noiseFloor": rssi_reading.noise_floor,
                    "snr": rssi_reading.snr,
                    "confidence": rssi_reading.confidence,
                    "timestamp": (
                        rssi_reading.timestamp.isoformat()
                        if rssi_reading.timestamp
                        else datetime.utcnow().isoformat()
                    ),
                },
            }

            # Broadcast to all clients
            await manager.broadcast_json(message)

            # Rate limiting to ensure 10Hz (100ms intervals)
            await asyncio.sleep(update_interval)

    except SignalProcessingError as e:
        logger.error(f"Error in RSSI broadcast task: {e}")


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates.

    Handles:
    - RSSI updates at 10Hz
    - Detection events
    - System state changes
    """
    await manager.connect(websocket)

    try:
        # Send initial connection message
        await websocket.send_json(
            {
                "type": "connection",
                "data": {
                    "status": "connected",
                    "timestamp": datetime.utcnow().isoformat(),
                },
            }
        )

        # Ensure signal processor is started
        await get_signal_processor()

        # Ensure MAVLink service is started
        await get_mavlink_service()

        # Start homing broadcast task if not already running
        global _homing_broadcast_task
        if _homing_broadcast_task is None:
            _homing_broadcast_task = asyncio.create_task(broadcast_homing_updates())

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages from client (ping/pong, commands, etc.)
                data = await websocket.receive_text()

                # Parse incoming message
                try:
                    message = json.loads(data)

                    # Handle different message types
                    if message.get("type") == "ping":
                        await websocket.send_json(
                            {
                                "type": "pong",
                                "data": {"timestamp": datetime.utcnow().isoformat()},
                            }
                        )
                    else:
                        logger.debug(f"Received WebSocket message: {message}")

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {data}")

            except WebSocketDisconnect:
                break
            except PISADException as e:
                logger.error(f"Error handling WebSocket message: {e}")
                break

    finally:
        await manager.disconnect(websocket)


@router.get("/ws-info")
async def websocket_info():
    """Information about WebSocket endpoint."""
    config = get_config()
    return {
        "endpoint": "/ws",
        "status": "active",
        "update_rate_hz": 1000 / config.websocket.WS_RSSI_UPDATE_INTERVAL_MS,
        "active_connections": len(manager.active_connections),
        "message_types": [
            "rssi - Signal strength updates",
            "detection - Detection events",
            "state - System state changes",
            "connection - Connection status",
            "config - Configuration changes",
            "telemetry - MAVLink telemetry updates",
            "mavlink_status - MAVLink connection status",
            "safety_status - Safety interlock status updates",
            "pattern_created - Search pattern created",
            "pattern_update - Search pattern progress updates",
            "pattern_pause - Search pattern paused",
            "pattern_resume - Search pattern resumed",
            "pattern_stop - Search pattern stopped",
            "homing_status - Homing algorithm status and gradient updates",
            "sdrpp_connection - SDR++ connection status updates",
            "dual_homing_status - Dual-system homing status and authority",
            "ground_signal_quality - Ground station signal quality metrics",
            "frequency_sync - Frequency synchronization status",
            "emergency_fallback - Emergency fallback status updates",
            "conflict_resolution - Conflict resolution status and history",
        ],
    }


async def get_safety_system() -> SafetyInterlockSystem:
    """Get or create safety interlock system instance."""
    global safety_system
    global _safety_broadcast_task

    if safety_system is None:
        safety_system = SafetyInterlockSystem()

        # Start safety monitoring
        await safety_system.start_monitoring()

        # Register with MAVLink service if available
        if mavlink_service:
            # Register callbacks for safety monitoring
            mavlink_service.add_mode_callback(safety_system.update_flight_mode)
            mavlink_service.add_battery_callback(safety_system.update_battery)
            mavlink_service.add_position_callback(safety_system.update_position)

            # Set safety check callback for velocity commands
            async def safety_check():
                return await safety_system.is_safe_to_proceed()

            mavlink_service.set_safety_check_callback(
                lambda: asyncio.run(safety_check())
            )

        # Register with signal processor if available
        if (
            signal_processor_integration
            and signal_processor_integration.signal_processor
        ):
            signal_processor_integration.signal_processor.add_snr_callback(
                safety_system.update_signal_snr
            )

        # Start safety status broadcast task
        _safety_broadcast_task = asyncio.create_task(broadcast_safety_updates())

        # Also update the system routes to use this instance
        from src.backend.api.routes import system as system_routes

        system_routes.safety_system = safety_system

        logger.info("Safety interlock system initialized and integrated")

    return safety_system


async def broadcast_safety_updates():
    """Broadcast safety status updates to all connected WebSocket clients at 2Hz."""
    try:
        system = await get_safety_system()

        while True:
            # Get comprehensive safety status
            safety_status = system.get_safety_status()

            # Format safety update message
            message = {
                "type": "safety_status",
                "data": {
                    "emergency_stopped": safety_status["emergency_stopped"],
                    "checks": safety_status["checks"],
                    "timestamp": safety_status["timestamp"],
                },
            }

            await manager.broadcast_json(message)

            # Broadcast at 2Hz
            await asyncio.sleep(0.5)

    except asyncio.CancelledError:
        logger.info("Safety broadcast task cancelled")
        raise
    except SafetyInterlockError as e:
        logger.error(f"Error in safety broadcast task: {e}")


async def broadcast_homing_updates():
    """Broadcast homing algorithm status updates to all connected WebSocket clients at 5Hz."""
    try:
        # Import locally to avoid circular dependency
        from src.backend.services.homing_controller import HomingController

        controller = None

        while True:
            try:
                # Get or create homing controller instance
                if controller is None:
                    controller = HomingController()

                # Get homing status
                status = controller.get_status()

                # Format homing update message
                message = {
                    "type": "homing_status",
                    "data": {
                        "enabled": status.get("enabled", False),
                        "substage": status.get("substage", "IDLE"),
                        "gradient_confidence": status.get("gradient_confidence", 0.0),
                        "target_heading": status.get("target_heading", 0.0),
                        "velocity_command": status.get(
                            "velocity_command", {"forward": 0.0, "yaw": 0.0}
                        ),
                        "rssi_history_size": status.get("rssi_history_size", 0),
                        "last_rssi": status.get("last_rssi", None),
                        "gradient_direction": status.get("gradient_direction", None),
                        "algorithm_mode": status.get("algorithm_mode", "GRADIENT"),
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                }

                await manager.broadcast_json(message)

            except Exception as e:
                logger.debug(f"Homing controller not available: {e}")

            # Broadcast at 5Hz (200ms intervals)
            await asyncio.sleep(0.2)

    except asyncio.CancelledError:
        logger.info("Homing broadcast task cancelled")
        raise
    except PISADException as e:
        logger.error(f"Error in homing broadcast task: {e}")


async def broadcast_message(message: dict[str, Any]) -> None:
    """
    Broadcast a message to all connected WebSocket clients.

    This function is exposed for other modules to send updates.

    Args:
        message: Dictionary containing the message to broadcast
    """
    await manager.broadcast_json(message)


# Story 5.4: SDR++ Integration WebSocket Handlers


async def broadcast_sdrpp_connection_status(
    status: str, latency: float | None = None, last_seen: str | None = None
) -> None:
    """Broadcast SDR++ connection status updates to all connected WebSocket clients."""
    try:
        # Validate status parameter
        if status not in ["connected", "disconnected", "connecting"]:
            logger.warning(f"Invalid SDR++ connection status: {status}")
            return

        message = {
            "type": WebSocketMessageTypes.SDRPP_CONNECTION,
            "data": {
                "status": status,  # 'connected' | 'disconnected' | 'connecting'
                "latency": latency,
                "lastSeen": last_seen or datetime.utcnow().isoformat(),
                "timestamp": datetime.utcnow().isoformat(),
            },
        }
        await manager.broadcast_json(message)
    except Exception as e:
        logger.error(f"Error broadcasting SDR++ connection status: {e}")


async def broadcast_dual_homing_status(
    active_source: str, homing_authority: str, performance_metrics: dict[str, Any]
) -> None:
    """Broadcast dual-system homing status updates to all connected WebSocket clients."""
    try:
        # Validate parameters
        if active_source not in ["ground", "drone"]:
            logger.warning(f"Invalid active source: {active_source}")
            return
        if homing_authority not in ["ground_priority", "drone_priority", "coordinated"]:
            logger.warning(f"Invalid homing authority: {homing_authority}")
            return

        message = {
            "type": WebSocketMessageTypes.DUAL_HOMING_STATUS,
            "data": {
                "activeSource": active_source,  # 'ground' | 'drone'
                "homingAuthority": homing_authority,  # 'ground_priority' | 'drone_priority' | 'coordinated'
                "performanceMetrics": performance_metrics or {},
                "timestamp": datetime.utcnow().isoformat(),
            },
        }
        await manager.broadcast_json(message)
    except Exception as e:
        logger.error(f"Error broadcasting dual homing status: {e}")


async def broadcast_ground_signal_quality(
    rssi: float, snr: float, noise_floor: float, quality_score: float
):
    """Broadcast ground signal quality metrics to all connected WebSocket clients."""
    try:
        # Validate numeric parameters
        if not all(
            isinstance(x, (int, float)) for x in [rssi, snr, noise_floor, quality_score]
        ):
            logger.warning("Invalid signal quality parameters - must be numeric")
            return

        message = {
            "type": WebSocketMessageTypes.GROUND_SIGNAL_QUALITY,
            "data": {
                "rssi": rssi,
                "snr": snr,
                "noiseFloor": noise_floor,
                "qualityScore": quality_score,
                "timestamp": datetime.utcnow().isoformat(),
            },
        }
        await manager.broadcast_json(message)
    except Exception as e:
        logger.error(f"Error broadcasting ground signal quality: {e}")


async def broadcast_frequency_sync_status(
    ground_freq: float, drone_freq: float, sync_status: str
) -> None:
    """Broadcast frequency synchronization status to all connected WebSocket clients."""
    try:
        # Validate parameters
        if sync_status not in ["synchronized", "mismatch", "syncing"]:
            logger.warning(f"Invalid sync status: {sync_status}")
            return
        if not all(isinstance(x, (int, float)) for x in [ground_freq, drone_freq]):
            logger.warning("Invalid frequency parameters - must be numeric")
            return

        message = {
            "type": WebSocketMessageTypes.FREQUENCY_SYNC,
            "data": {
                "groundFrequency": ground_freq,
                "droneFrequency": drone_freq,
                "syncStatus": sync_status,  # 'synchronized' | 'mismatch' | 'syncing'
                "frequencyDifference": abs(ground_freq - drone_freq),
                "timestamp": datetime.utcnow().isoformat(),
            },
        }
        await manager.broadcast_json(message)
    except Exception as e:
        logger.error(f"Error broadcasting frequency sync status: {e}")


async def broadcast_emergency_fallback_status(
    fallback_active: bool, trigger_reason: str = None
):
    """Broadcast emergency fallback status updates to all connected WebSocket clients."""
    try:
        if not isinstance(fallback_active, bool):
            logger.warning("Invalid fallback_active parameter - must be boolean")
            return

        message = {
            "type": WebSocketMessageTypes.EMERGENCY_FALLBACK,
            "data": {
                "fallbackActive": fallback_active,
                "triggerReason": trigger_reason,
                "droneAutonomy": fallback_active,
                "communicationStatus": "lost" if fallback_active else "normal",
                "timestamp": datetime.utcnow().isoformat(),
            },
        }
        await manager.broadcast_json(message)
    except Exception as e:
        logger.error(f"Error broadcasting emergency fallback status: {e}")


async def broadcast_conflict_resolution_status(conflict_data: dict[str, Any]):
    """Broadcast conflict resolution status to all connected WebSocket clients."""
    try:
        if not isinstance(conflict_data, dict):
            logger.warning("Invalid conflict_data parameter - must be dict")
            return

        message = {
            "type": WebSocketMessageTypes.CONFLICT_RESOLUTION,
            "data": conflict_data,
        }
        await manager.broadcast_json(message)
    except Exception as e:
        logger.error(f"Error broadcasting conflict resolution status: {e}")
