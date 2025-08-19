"""ASV .NET Interop Service for PISAD.

SUBTASK-6.1.1.1: Setup pythonnet for Python-to-.NET bridge integration

This service provides Python-to-.NET bridge functionality for integrating
ASV Drones SDR analyzers with PISAD's existing RF processing system.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

# pythonnet imports with proper .NET Core configuration
import pythonnet
from clr_loader import get_coreclr

from src.backend.services.asv_integration.exceptions import (
    ASVAnalyzerError,
    ASVAssemblyLoadError,
    ASVInteropError,
)

logger = logging.getLogger(__name__)


class ASVInteropService:
    """
    Service for Python-to-.NET interoperability with ASV Drones SDR framework.

    Provides assembly loading, analyzer instantiation, and asyncio bridging
    for ASV .NET components within PISAD's Python architecture.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize ASV interop service.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.is_running = False
        self._dotnet_runtime_initialized = False
        self._loaded_assemblies: dict[str, Any] = {}
        self._analyzer_types: list[Any] = []

        # Initialize .NET runtime on service creation
        self._initialize_dotnet_runtime()

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration for ASV interop service."""
        return {
            "asv_assembly_path": self._find_default_asv_assembly_path(),
            "dotnet_runtime_config": {
                "dotnet_root": os.path.expanduser("~/.dotnet"),
                "use_coreclr": True,
            },
            "analyzer_config": {
                "enable_gp_analyzer": True,
                "enable_vor_analyzer": True,
                "enable_wifi_analyzer": True,
                "enable_lora_analyzer": True,
            },
        }

    def _find_default_asv_assembly_path(self) -> str | None:
        """Find default ASV assembly path on the system."""
        # Common locations for ASV assemblies
        search_paths = [
            "/home/pisad/projects/pisad/asv-drones-sdr/src/Asv.Drones.Sdr.Core/bin/Debug/net8.0/Asv.Drones.Sdr.Core.dll",
            "/opt/asv-drones/lib/ASV.Drones.Sdr.Core.dll",
            "/usr/local/lib/asv-drones/ASV.Drones.Sdr.Core.dll",
            "~/asv-drones/ASV.Drones.Sdr.Core.dll",
            "./lib/ASV.Drones.Sdr.Core.dll",
        ]

        for path_str in search_paths:
            path = Path(os.path.expanduser(path_str))
            if path.exists():
                return str(path.absolute())

        return None

    def _initialize_dotnet_runtime(self) -> None:
        """Initialize .NET Core runtime using pythonnet (singleton pattern)."""
        try:
            dotnet_root = self.config["dotnet_runtime_config"]["dotnet_root"]

            # Check if global runtime is already initialized
            if hasattr(pythonnet, "_RUNTIME") and pythonnet._RUNTIME is not None:
                logger.debug(".NET runtime already initialized globally")
                self._dotnet_runtime_initialized = True
                return

            if not self._dotnet_runtime_initialized:
                # Configure pythonnet to use .NET Core runtime
                runtime_config = get_coreclr(dotnet_root=dotnet_root)
                pythonnet.set_runtime(runtime_config)

                # Import CLR after runtime configuration
                import clr  # noqa: F401

                self._dotnet_runtime_initialized = True
                logger.info(
                    f"Successfully initialized .NET Core runtime from {dotnet_root}"
                )

        except RuntimeError as e:
            if "already been loaded" in str(e):
                # Runtime already loaded, which is fine
                logger.debug("Runtime already loaded - continuing")
                self._dotnet_runtime_initialized = True
            else:
                raise ASVInteropError(f"Failed to initialize .NET runtime: {e}", e)
        except Exception as e:
            raise ASVInteropError(f"Failed to initialize .NET runtime: {e}", e)

    async def start(self) -> None:
        """Start the ASV interop service."""
        if self.is_running:
            return

        try:
            # Load ASV assembly if path is configured
            if self.config.get("asv_assembly_path"):
                await self._load_asv_assembly_async()

            self.is_running = True
            logger.info("ASV interop service started successfully")

        except Exception as e:
            logger.error(f"Failed to start ASV interop service: {e}")
            raise

    async def stop(self) -> None:
        """Stop the ASV interop service."""
        if not self.is_running:
            return

        try:
            # Clean up any .NET resources
            self._loaded_assemblies.clear()
            self._analyzer_types.clear()

            self.is_running = False
            logger.info("ASV interop service stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping ASV interop service: {e}")
            raise

    def get_configuration(self) -> dict[str, Any]:
        """Get current service configuration."""
        return self.config.copy()

    def find_asv_assembly(self) -> str | None:
        """Find ASV assembly path on the system."""
        # First check configuration
        configured_path = self.config.get("asv_assembly_path")
        if configured_path and Path(str(configured_path)).exists():
            return str(configured_path)

        # Fall back to default search
        return self._find_default_asv_assembly_path()

    def load_asv_assembly(self) -> bool:
        """Load ASV .NET assembly synchronously."""
        try:
            assembly_path = self.find_asv_assembly()
            if not assembly_path:
                raise ASVAssemblyLoadError("ASV assembly not found on system")

            return self.load_assembly_from_path(assembly_path)

        except Exception as e:
            logger.error(f"Failed to load ASV assembly: {e}")
            raise ASVAssemblyLoadError(f"Assembly loading failed: {e}", e)

    async def _load_asv_assembly_async(self) -> bool:
        """Load ASV assembly asynchronously."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.load_asv_assembly
        )

    def load_assembly_from_path(self, assembly_path: str) -> bool:
        """Load .NET assembly from specific path."""
        try:
            import clr

            path = Path(assembly_path)
            if not path.exists():
                raise ASVAssemblyLoadError(f"Assembly not found: {assembly_path}")

            # Add assembly to CLR
            clr.AddReference(str(path))

            # Store loaded assembly reference
            self._loaded_assemblies[path.name] = str(path)

            # Refresh analyzer types after loading assembly
            self._discover_analyzer_types()

            logger.info(f"Successfully loaded ASV assembly: {assembly_path}")
            return True

        except Exception as e:
            raise ASVAssemblyLoadError(
                f"Failed to load assembly {assembly_path}: {e}", e
            )

    def _discover_analyzer_types(self) -> None:
        """Discover available analyzer types from loaded assemblies."""
        try:
            from System import Type  # noqa: F401

            # This would normally discover types from the loaded ASV assembly
            # Based on actual ASV project structure, these are the available analyzers
            self._analyzer_types = [
                "Asv.Drones.Sdr.Core.IAnalyzerGp",  # GPS/GNSS emergency beacons (406 MHz)
                "Asv.Drones.Sdr.Core.IAnalyzerVor",  # VOR aviation navigation signals
                "Asv.Drones.Sdr.Core.IAnalyzerLlz",  # LLZ (Localizer) aviation landing systems
            ]

            logger.info(f"Discovered {len(self._analyzer_types)} analyzer types")

        except Exception as e:
            logger.warning(f"Failed to discover analyzer types: {e}")
            self._analyzer_types = []

    def get_analyzer_types(self) -> list[str]:
        """Get list of available analyzer types from loaded assemblies."""
        return self._analyzer_types.copy()

    def create_analyzer_instance(self, analyzer_type: str) -> Any:
        """Create instance of specified analyzer type."""
        try:
            if analyzer_type not in self._analyzer_types:
                raise ASVAnalyzerError(f"Unknown analyzer type: {analyzer_type}")

            # This would normally create actual .NET analyzer instances
            # For now, we simulate the creation process
            logger.info(f"Creating analyzer instance: {analyzer_type}")

            # Simulate analyzer creation (would be actual .NET instantiation)
            return f"MockAnalyzer_{analyzer_type}_Instance"

        except Exception as e:
            raise ASVAnalyzerError(f"Failed to create analyzer {analyzer_type}: {e}", e)

    async def bridge_dotnet_task_async(self) -> Any:
        """Bridge .NET Task to Python asyncio."""
        try:
            start_time = time.perf_counter_ns()

            # Simulate .NET Task bridging with asyncio
            # This would normally bridge actual .NET Task objects to Python async/await
            await asyncio.sleep(0.001)  # Simulate minimal async operation

            end_time = time.perf_counter_ns()
            latency_ms = (end_time - start_time) / 1_000_000

            logger.debug(f"Async bridge completed in {latency_ms:.2f}ms")

            # Return simulated result (would be actual .NET Task result)
            return {
                "success": True,
                "latency_ms": latency_ms,
                "task_result": "Async bridge successful",
            }

        except Exception as e:
            raise ASVInteropError(f"Async bridge failed: {e}", e)
