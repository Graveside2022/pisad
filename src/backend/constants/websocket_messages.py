"""
WebSocket message type constants for Story 5.4 SDR++ Integration.

This module defines all WebSocket message types used for communication
between the backend and frontend components in Story 5.4.
"""

from typing import ClassVar, Literal


# Story 5.4 WebSocket Message Types
class WebSocketMessageTypes:
    """WebSocket message type constants for Story 5.4 SDR++ Integration."""

    # SDR++ Integration Message Types
    SDRPP_CONNECTION = "sdrpp_connection"
    DUAL_HOMING_STATUS = "dual_homing_status"
    GROUND_SIGNAL_QUALITY = "ground_signal_quality"
    FREQUENCY_SYNC = "frequency_sync"
    EMERGENCY_FALLBACK = "emergency_fallback"
    CONFLICT_RESOLUTION = "conflict_resolution"

    # All Story 5.4 message types
    ALL_STORY_5_4_TYPES: ClassVar[list[str]] = [
        SDRPP_CONNECTION,
        DUAL_HOMING_STATUS,
        GROUND_SIGNAL_QUALITY,
        FREQUENCY_SYNC,
        EMERGENCY_FALLBACK,
        CONFLICT_RESOLUTION,
    ]


# Type aliases for better type safety
SDRConnectionStatus = Literal["connected", "disconnected", "connecting"]
ActiveSource = Literal["ground", "drone"]
HomingAuthority = Literal["ground_priority", "drone_priority", "coordinated"]
SyncStatus = Literal["synchronized", "mismatch", "syncing"]
