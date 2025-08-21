"""
RF Regulation Validator for Frequency Compliance Checking

TASK-6.3.1 [28d3] - RF regulation compliance validation
Validates frequency selections against regional frequency allocation rules
and provides compliance recommendations for SAR operations.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class FrequencyAllocation:
    """Frequency allocation information for specific bands."""

    start_freq_hz: int
    end_freq_hz: int
    allocation_type: str
    service_category: str
    license_required: bool
    restrictions: List[str]
    region: str = "US"


class RFRegulationValidator:
    """
    Validates frequency selections against RF regulations.

    Provides validation for:
    - HackRF hardware capabilities and optimal ranges
    - US frequency allocation compliance (FCC Part 97)
    - International frequency coordination requirements
    - Emergency service frequency protection
    """

    def __init__(self):
        """Initialize RF regulation validator with frequency allocation tables."""
        self._us_allocations = self._load_us_frequency_allocations()
        self._hackrf_specs = self._load_hackrf_specifications()

        logger.info("RF regulation validator initialized")

    def validate_hackrf_range(self, frequency_hz: int) -> Dict[str, Any]:
        """
        Validate frequency against HackRF hardware capabilities.

        Args:
            frequency_hz: Frequency to validate in Hz

        Returns:
            Dict containing validation result with range classification
        """
        # HackRF One specifications
        absolute_min = 1_000_000  # 1 MHz absolute minimum
        effective_min = 24_000_000  # 24 MHz effective minimum
        effective_max = 1_750_000_000  # 1.75 GHz effective maximum
        absolute_max = 6_000_000_000  # 6 GHz absolute maximum

        if frequency_hz < absolute_min or frequency_hz > absolute_max:
            return {
                "valid": False,
                "range": "invalid",
                "error": f"Frequency {frequency_hz/1e6:.3f} MHz is outside HackRF range (1 MHz - 6 GHz)",
            }

        if effective_min <= frequency_hz <= effective_max:
            return {
                "valid": True,
                "range": "effective",
                "performance": "optimal",
                "notes": f"Frequency {frequency_hz/1e6:.3f} MHz is within HackRF effective range",
            }

        # Extended range - works but with reduced performance
        if frequency_hz < effective_min:
            return {
                "valid": True,
                "range": "extended",
                "performance": "reduced",
                "warning": f"Frequency {frequency_hz/1e6:.3f} MHz has reduced performance below 24 MHz",
            }

        # High frequency extended range
        return {
            "valid": True,
            "range": "extended",
            "performance": "reduced",
            "warning": f"Frequency {frequency_hz/1e6:.3f} MHz has reduced performance above 1.75 GHz",
        }

    def validate_us_compliance(
        self, frequency_hz: int, use_case: str
    ) -> Dict[str, Any]:
        """
        Validate frequency against US FCC regulations.

        Args:
            frequency_hz: Frequency to validate in Hz
            use_case: Intended use case (emergency, aviation, maritime, custom)

        Returns:
            Dict containing compliance validation result
        """
        frequency_mhz = frequency_hz / 1e6

        # Find applicable frequency allocation
        allocation = self._find_us_allocation(frequency_hz)

        if not allocation:
            return {
                "compliant": False,
                "allocation": "unallocated",
                "restriction": "No valid allocation found for this frequency",
                "recommendation": "Use frequencies within designated bands",
            }

        # Check specific use case compliance
        compliance_result = self._check_use_case_compliance(allocation, use_case)

        return {
            "compliant": compliance_result["allowed"],
            "allocation": allocation.allocation_type,
            "service_category": allocation.service_category,
            "license_required": allocation.license_required,
            "restrictions": allocation.restrictions,
            "recommendation": compliance_result["recommendation"],
            "frequency_mhz": frequency_mhz,
        }

    def _load_us_frequency_allocations(self) -> List[FrequencyAllocation]:
        """Load US frequency allocation table from FCC regulations."""
        return [
            # Maritime mobile service (Emergency)
            FrequencyAllocation(
                start_freq_hz=156_000_000,
                end_freq_hz=174_000_000,
                allocation_type="maritime_mobile",
                service_category="maritime",
                license_required=False,  # For emergency use
                restrictions=["Emergency and safety use only"],
                region="US",
            ),
            # Aeronautical mobile service
            FrequencyAllocation(
                start_freq_hz=108_000_000,
                end_freq_hz=137_000_000,
                allocation_type="aeronautical_mobile",
                service_category="aviation",
                license_required=False,  # For emergency monitoring
                restrictions=["Aviation emergency and navigation"],
                region="US",
            ),
            # Emergency beacon (COSPAS-SARSAT)
            FrequencyAllocation(
                start_freq_hz=406_000_000,
                end_freq_hz=406_100_000,
                allocation_type="satellite_emergency",
                service_category="emergency",
                license_required=False,  # For SAR monitoring
                restrictions=["Emergency beacon monitoring only"],
                region="US",
            ),
            # FM Broadcast (protected)
            FrequencyAllocation(
                start_freq_hz=88_000_000,
                end_freq_hz=108_000_000,
                allocation_type="broadcast",
                service_category="commercial",
                license_required=True,
                restrictions=["Commercial broadcasting", "No transmission allowed"],
                region="US",
            ),
            # Public safety
            FrequencyAllocation(
                start_freq_hz=150_000_000,
                end_freq_hz=156_000_000,
                allocation_type="public_safety",
                service_category="government",
                license_required=True,
                restrictions=["Government and public safety use"],
                region="US",
            ),
        ]

    def _load_hackrf_specifications(self) -> Dict[str, Any]:
        """Load HackRF technical specifications."""
        return {
            "frequency_range": {
                "absolute_min_hz": 1_000_000,
                "effective_min_hz": 24_000_000,
                "effective_max_hz": 1_750_000_000,
                "absolute_max_hz": 6_000_000_000,
            },
            "performance_bands": {
                "optimal": (24_000_000, 1_750_000_000),
                "good": (10_000_000, 3_000_000_000),
                "reduced": (1_000_000, 6_000_000_000),
            },
            "sample_rates": {
                "min_hz": 2_000_000,
                "max_hz": 20_000_000,
                "recommended_hz": 10_000_000,
            },
        }

    def _find_us_allocation(self, frequency_hz: int) -> FrequencyAllocation | None:
        """Find the US frequency allocation for a given frequency."""
        for allocation in self._us_allocations:
            if allocation.start_freq_hz <= frequency_hz <= allocation.end_freq_hz:
                return allocation
        return None

    def _check_use_case_compliance(
        self, allocation: FrequencyAllocation, use_case: str
    ) -> Dict[str, Any]:
        """Check if use case is compliant with frequency allocation."""

        # Emergency use cases are generally allowed for monitoring
        if use_case in ["emergency", "maritime_emergency", "aviation_emergency"]:
            if allocation.service_category in ["emergency", "maritime", "aviation"]:
                return {
                    "allowed": True,
                    "recommendation": f"Approved for {use_case} monitoring in {allocation.allocation_type} band",
                }

        # SAR operations have special permissions
        if "sar" in use_case.lower():
            if allocation.service_category in ["emergency", "maritime", "aviation"]:
                return {
                    "allowed": True,
                    "recommendation": f"Approved for SAR operations in {allocation.allocation_type} band",
                }

        # Custom use cases need careful validation
        if use_case == "custom":
            if allocation.license_required:
                return {
                    "allowed": False,
                    "recommendation": f"License required for use in {allocation.allocation_type} band",
                }
            elif "No transmission" in " ".join(allocation.restrictions):
                return {
                    "allowed": True,
                    "recommendation": f"Monitoring only allowed in {allocation.allocation_type} band",
                }

        # Default: not allowed
        return {
            "allowed": False,
            "recommendation": f"Use case '{use_case}' not permitted in {allocation.allocation_type} band",
        }

    def get_recommended_frequencies(
        self, use_case: str, region: str = "US"
    ) -> List[Dict[str, Any]]:
        """Get list of recommended frequencies for specific use case."""
        recommendations = []

        for allocation in self._us_allocations:
            if region.upper() == allocation.region:
                compliance = self._check_use_case_compliance(allocation, use_case)
                if compliance["allowed"]:
                    # Calculate center frequency
                    center_freq = (
                        allocation.start_freq_hz + allocation.end_freq_hz
                    ) // 2

                    recommendations.append(
                        {
                            "frequency_hz": center_freq,
                            "frequency_mhz": center_freq / 1e6,
                            "allocation": allocation.allocation_type,
                            "service": allocation.service_category,
                            "license_required": allocation.license_required,
                            "restrictions": allocation.restrictions,
                            "recommendation": compliance["recommendation"],
                        }
                    )

        return sorted(recommendations, key=lambda x: x["frequency_hz"])

    def validate_emergency_frequencies(self) -> Dict[str, Any]:
        """Validate all emergency frequencies used in PISAD system."""
        emergency_freqs = {
            "maritime_sar": 162_025_000,
            "aviation_emergency": 121_500_000,
            "emergency_beacon": 406_000_000,
        }

        results = {}
        for name, freq in emergency_freqs.items():
            hackrf_result = self.validate_hackrf_range(freq)
            us_result = self.validate_us_compliance(freq, name)

            results[name] = {
                "frequency_hz": freq,
                "frequency_mhz": freq / 1e6,
                "hackrf_validation": hackrf_result,
                "us_compliance": us_result,
                "overall_status": (
                    "approved"
                    if (hackrf_result["valid"] and us_result["compliant"])
                    else "restricted"
                ),
            }

        return results
