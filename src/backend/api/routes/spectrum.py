"""
Spectrum analysis API endpoints for RF waterfall display and analysis.
Provides real-time FFT data from SDR hardware for frontend visualization.
"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Query

from src.backend.core.dependencies import get_sdr_service
from src.backend.models.schemas import SpectrumData
from src.backend.services.sdr_service import SDRService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/spectrum", tags=["spectrum"])


@router.get("/waterfall", response_model=SpectrumData)
async def get_spectrum_waterfall(
    center_freq: int = Query(
        2437000000,
        ge=850000000,
        le=6500000000,
        description="Center frequency in Hz (850 MHz - 6.5 GHz per PRD-FR1)",
    ),
    bandwidth: int = Query(
        5000000,
        ge=1000000,
        le=20000000,
        description="Bandwidth in Hz (1-20 MHz, HackRF One limitation)",
    ),
    sdr_service: SDRService = Depends(get_sdr_service),
) -> SpectrumData:
    """
    Get real-time FFT spectrum data for waterfall display.

    Returns FFT magnitude data from HackRF One SDR with 5MHz bandwidth
    around the specified center frequency. Optimized for 3Hz update rate.

    Args:
        center_freq: Center frequency in Hz (850 MHz - 6.5 GHz)
        bandwidth: Bandwidth in Hz (1-20 MHz)
        sdr_service: Injected SDR service

    Returns:
        SpectrumData: FFT data with frequencies, magnitudes, timestamp

    Raises:
        HTTPException: If frequency out of range or SDR hardware error
    """
    try:
        # Validate frequency range per PRD-FR1
        if center_freq < 850000000 or center_freq > 6500000000:
            raise HTTPException(
                status_code=422,
                detail="Center frequency must be between 850 MHz and 6.5 GHz (PRD-FR1)",
            )

        # Validate bandwidth for HackRF One capabilities
        if bandwidth > 20000000:
            raise HTTPException(
                status_code=422, detail="Bandwidth must not exceed 20 MHz (HackRF One limitation)"
            )

        # Get FFT data from SDR service
        logger.debug(f"Getting FFT data for {center_freq/1e6:.1f} MHz, BW={bandwidth/1e6:.1f} MHz")

        fft_data = await sdr_service.get_fft_data(
            center_freq=center_freq,
            bandwidth=bandwidth,
            num_samples=1024,  # Standard FFT size for waterfall
        )

        # Convert to frontend-compatible format
        spectrum_data = SpectrumData(
            frequencies=fft_data["frequencies"].tolist(),
            magnitudes=fft_data["magnitudes"].tolist(),
            timestamp=int(time.time() * 1000),  # Milliseconds
            centerFreq=center_freq,
            sampleRate=fft_data.get("sample_rate", bandwidth),
        )

        logger.debug(f"Returning spectrum data: {len(spectrum_data.frequencies)} points")
        return spectrum_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting spectrum data: {e}")
        raise HTTPException(status_code=500, detail=f"SDR hardware error: {e!s}")


@router.post("/configure")
async def configure_spectrum_settings(
    center_freq: int | None = None,
    sample_rate: int | None = None,
    gain: float | None = None,
    immediate: bool = True,
    sdr_service: SDRService = Depends(get_sdr_service),
):
    """
    Configure SDR spectrum analysis settings.

    Updates SDR configuration with immediate application for real-time control.
    Used by frequency control components for instant frequency changes.

    Args:
        center_freq: New center frequency in Hz
        sample_rate: New sample rate in Hz
        gain: New gain setting in dB
        immediate: Apply changes immediately (default: True)
        sdr_service: Injected SDR service

    Returns:
        dict: Configuration status
    """
    try:
        config_updates = {}

        if center_freq is not None:
            # Validate frequency range per PRD-FR1
            if center_freq < 850000000 or center_freq > 6500000000:
                raise HTTPException(
                    status_code=422,
                    detail="Center frequency must be between 850 MHz and 6.5 GHz (PRD-FR1)",
                )
            config_updates["center_freq"] = center_freq

        if sample_rate is not None:
            config_updates["sample_rate"] = sample_rate

        if gain is not None:
            config_updates["gain"] = gain

        # Apply configuration changes
        if config_updates:
            await sdr_service.update_config(config_updates, immediate=immediate)
            logger.info(f"Updated SDR config: {config_updates}")

        return {"success": True, "updates": config_updates, "immediate": immediate}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating SDR config: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration error: {e!s}")
