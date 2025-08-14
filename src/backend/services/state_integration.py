"""Integration module for connecting services with the enhanced state machine."""

from src.backend.services.homing_controller import HomingController
from src.backend.services.mavlink_service import MAVLinkService
from src.backend.services.search_pattern_generator import SearchPatternGenerator
from src.backend.services.signal_processor import SignalProcessor
from src.backend.services.state_machine import StateMachine, SystemState
from src.backend.utils.logging import get_logger

logger = get_logger(__name__)


class StateIntegration:
    """Integrates system services with the enhanced state machine."""

    def __init__(
        self,
        state_machine: StateMachine,
        mavlink_service: MAVLinkService | None = None,
        signal_processor: SignalProcessor | None = None,
        homing_controller: HomingController | None = None,
        search_pattern_generator: SearchPatternGenerator | None = None,
    ):
        """Initialize state integration.

        Args:
            state_machine: The state machine to integrate with
            mavlink_service: MAVLink service for drone control
            signal_processor: Signal processor for RSSI data
            homing_controller: Homing controller for beacon tracking
            search_pattern_generator: Search pattern generator for waypoints
        """
        self.state_machine = state_machine
        self.mavlink_service = mavlink_service
        self.signal_processor = signal_processor
        self.homing_controller = homing_controller
        self.search_pattern_generator = search_pattern_generator

        # Connect services to state machine
        self._connect_services()

        # Register state callbacks
        self._register_callbacks()

        logger.info("State integration initialized")

    def _connect_services(self) -> None:
        """Connect services to the state machine."""
        if self.mavlink_service:
            self.state_machine.set_mavlink_service(self.mavlink_service)
            logger.info("Connected MAVLink service to state machine")

        if self.signal_processor:
            self.state_machine.set_signal_processor(self.signal_processor)
            logger.info("Connected signal processor to state machine")

    def _register_callbacks(self) -> None:
        """Register state change callbacks."""
        # Register state transition callback
        self.state_machine.add_state_callback(self._on_state_change)

        # Register signal processor detection callback if available
        if self.signal_processor:
            self.signal_processor.add_detection_callback(self._on_signal_detection)

    async def _on_state_change(
        self, old_state: SystemState, new_state: SystemState, reason: str | None
    ) -> None:
        """Handle state changes.

        Args:
            old_state: Previous state
            new_state: New state
            reason: Reason for transition
        """
        logger.info(f"State change: {old_state.value} -> {new_state.value} ({reason})")

        # Handle state-specific actions
        if new_state == SystemState.SEARCHING:
            await self._start_searching()
        elif new_state == SystemState.HOMING:
            await self._start_homing()
        elif new_state == SystemState.HOLDING:
            await self._start_holding()
        elif new_state == SystemState.IDLE:
            await self._stop_all_operations()

    async def _on_signal_detection(self, rssi: float, confidence: float) -> None:
        """Handle signal detection events.

        Args:
            rssi: Signal strength in dBm
            confidence: Detection confidence percentage
        """
        # Forward to state machine
        await self.state_machine.handle_detection(rssi, confidence)

        # If in DETECTING state and homing controller available, update it
        if (
            self.state_machine.get_current_state() == SystemState.DETECTING
            and self.homing_controller
        ):
            self.homing_controller.update_rssi(rssi)

    async def _start_searching(self) -> None:
        """Start search operations."""
        if not self.search_pattern_generator:
            logger.warning("No search pattern generator available")
            return

        # Check if there's an active pattern
        pattern = self.state_machine.get_search_pattern()
        if not pattern:
            logger.info("No search pattern loaded, cannot start searching")
            return

        # Start pattern execution
        try:
            await self.state_machine.start_search_pattern()

            # If MAVLink available, send first waypoint
            if self.mavlink_service:
                waypoint = self.state_machine.get_next_waypoint()
                if waypoint:
                    await self.mavlink_service.goto_waypoint(
                        waypoint.latitude, waypoint.longitude, waypoint.altitude
                    )
        except Exception as e:
            logger.error(f"Failed to start searching: {e}")

    async def _start_homing(self) -> None:
        """Start homing operations."""
        if not self.homing_controller:
            logger.warning("No homing controller available")
            return

        try:
            # Start homing algorithm
            await self.homing_controller.start_homing()
        except Exception as e:
            logger.error(f"Failed to start homing: {e}")
            # Return to searching on failure
            await self.state_machine.transition_to(SystemState.SEARCHING, "Homing start failed")

    async def _start_holding(self) -> None:
        """Start position holding."""
        if not self.mavlink_service:
            logger.warning("No MAVLink service for position hold")
            return

        try:
            # Enable position hold mode
            await self.mavlink_service.set_mode("POSHOLD")
        except Exception as e:
            logger.error(f"Failed to enable position hold: {e}")

    async def _stop_all_operations(self) -> None:
        """Stop all active operations when returning to IDLE."""
        # Stop homing if active
        if self.homing_controller:
            try:
                await self.homing_controller.stop_homing()
            except Exception as e:
                logger.error(f"Error stopping homing: {e}")

        # Stop search pattern if active
        if self.state_machine.get_search_pattern():
            try:
                await self.state_machine.stop_search_pattern()
            except Exception as e:
                logger.error(f"Error stopping search pattern: {e}")

        # Return to loiter mode if MAVLink available
        if self.mavlink_service:
            try:
                await self.mavlink_service.set_mode("LOITER")
            except Exception as e:
                logger.error(f"Error setting loiter mode: {e}")

    async def handle_waypoint_reached(self, waypoint_index: int) -> None:
        """Handle waypoint reached event from MAVLink.

        Args:
            waypoint_index: Index of reached waypoint
        """
        if self.state_machine.get_current_state() != SystemState.SEARCHING:
            return

        # Update pattern progress
        self.state_machine.update_waypoint_progress(waypoint_index + 1)

        # Get next waypoint
        next_waypoint = self.state_machine.get_next_waypoint()
        if next_waypoint and self.mavlink_service:
            await self.mavlink_service.goto_waypoint(
                next_waypoint.latitude, next_waypoint.longitude, next_waypoint.altitude
            )
        else:
            # Pattern complete, return to IDLE
            logger.info("Search pattern complete")
            await self.state_machine.transition_to(SystemState.IDLE, "Search pattern complete")

    async def handle_signal_lost(self) -> None:
        """Handle loss of signal during operations."""
        current_state = self.state_machine.get_current_state()

        if current_state in [SystemState.DETECTING, SystemState.HOMING]:
            # Signal lost during active tracking
            await self.state_machine.handle_signal_lost()

            # If homing controller active, stop it
            if self.homing_controller and current_state == SystemState.HOMING:
                try:
                    await self.homing_controller.stop_homing()
                except Exception as e:
                    logger.error(f"Error stopping homing after signal loss: {e}")

    async def emergency_stop(self, reason: str = "Emergency stop") -> None:
        """Perform emergency stop of all operations.

        Args:
            reason: Reason for emergency stop
        """
        logger.critical(f"Emergency stop initiated: {reason}")

        # Stop all operations
        await self._stop_all_operations()

        # Transition to IDLE
        await self.state_machine.emergency_stop(reason)

        # If MAVLink available, switch to manual mode for safety
        if self.mavlink_service:
            try:
                await self.mavlink_service.set_mode("MANUAL")
            except Exception as e:
                logger.error(f"Failed to set manual mode: {e}")
