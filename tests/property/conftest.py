"""Hypothesis configuration for property-based testing.

This module configures hypothesis profiles for different environments.
"""

from hypothesis import HealthCheck, settings

# Register profiles for different environments
settings.register_profile(
    "ci",
    max_examples=100,  # Fewer examples in CI for speed
    deadline=2000,  # 2 seconds deadline
    suppress_health_check=[HealthCheck.too_slow],  # CI can be slow
    print_blob=True,  # Print failing examples
)

settings.register_profile(
    "dev",
    max_examples=50,  # Quick feedback during development
    deadline=1000,  # 1 second deadline
    print_blob=True,
)

settings.register_profile(
    "debug",
    max_examples=10,  # Minimal examples for debugging
    deadline=None,  # No deadline when debugging
    print_blob=True,
    verbosity=2,  # Verbose output
)

settings.register_profile(
    "thorough",
    max_examples=1000,  # Thorough testing
    deadline=5000,  # 5 seconds deadline
    print_blob=True,
)

# Default to dev profile, override with HYPOTHESIS_PROFILE env var
settings.load_profile("dev")
