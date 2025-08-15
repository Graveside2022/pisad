"""Configuration management API routes."""

import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from src.backend.api.websocket import broadcast_message
from src.backend.core.exceptions import (
    ConfigurationError,
    SDRError,
)
from src.backend.models.database import ConfigProfileDB
from src.backend.models.schemas import (
    ConfigProfile,
    HomingConfig,
    SDRConfig,
    SignalConfig,
)
from src.backend.services.config_service import ConfigService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/config", tags=["Configuration"])

# Initialize services
config_service = ConfigService()
profile_db = ConfigProfileDB()


async def _get_profile_by_id(profile_id: str) -> ConfigProfile | None:
    """Helper function to get a profile by ID from YAML or database.

    Args:
        profile_id: ID of the profile to retrieve

    Returns:
        ConfigProfile if found, None otherwise
    """
    # Check YAML files first
    for name in config_service.list_profiles():
        profile = config_service.load_profile(name)
        if profile and profile.id == profile_id:
            return profile

    # Check database
    db_profile = profile_db.get_profile(profile_id)
    if db_profile:
        # Convert database profile to ConfigProfile
        return ConfigProfile(
            id=db_profile["id"],
            name=db_profile["name"],
            description=db_profile["description"],
            sdrConfig=SDRConfig(**db_profile["sdrConfig"]) if db_profile["sdrConfig"] else None,
            signalConfig=(
                SignalConfig(**db_profile["signalConfig"]) if db_profile["signalConfig"] else None
            ),
            homingConfig=(
                HomingConfig(**db_profile["homingConfig"]) if db_profile["homingConfig"] else None
            ),
            isDefault=db_profile["isDefault"],
        )

    return None


class ProfileCreateRequest(BaseModel):
    """Request model for creating a configuration profile."""

    name: str
    description: str = ""
    sdrConfig: dict[str, Any]
    signalConfig: dict[str, Any]
    homingConfig: dict[str, Any]
    isDefault: bool = False


class ProfileUpdateRequest(BaseModel):
    """Request model for updating a configuration profile."""

    name: str
    description: str = ""
    sdrConfig: dict[str, Any]
    signalConfig: dict[str, Any]
    homingConfig: dict[str, Any]
    isDefault: bool = False


class ProfileResponse(BaseModel):
    """Response model for configuration profile."""

    id: str
    name: str
    description: str
    sdrConfig: dict[str, Any]
    signalConfig: dict[str, Any]
    homingConfig: dict[str, Any]
    isDefault: bool
    createdAt: str
    updatedAt: str


@router.get("/profiles", response_model=list[ProfileResponse])
async def list_profiles():
    """List all available configuration profiles.

    Returns:
        List of configuration profiles
    """
    try:
        logger.info("Listing configuration profiles")

        # Get profiles from YAML files
        profile_names = config_service.list_profiles()
        profiles = []

        for name in profile_names:
            profile = config_service.load_profile(name)
            if profile:
                profiles.append(
                    {
                        "id": profile.id,
                        "name": profile.name,
                        "description": profile.description,
                        "sdrConfig": profile.sdrConfig.__dict__ if profile.sdrConfig else {},
                        "signalConfig": (
                            profile.signalConfig.__dict__ if profile.signalConfig else {}
                        ),
                        "homingConfig": (
                            profile.homingConfig.__dict__ if profile.homingConfig else {}
                        ),
                        "isDefault": profile.isDefault,
                        "createdAt": profile.createdAt.isoformat(),
                        "updatedAt": profile.updatedAt.isoformat(),
                    }
                )

        # Also get profiles from database
        db_profiles = profile_db.list_profiles()
        for db_profile in db_profiles:
            # Check if not already in YAML profiles
            if not any(p["name"] == db_profile["name"] for p in profiles):
                profiles.append(db_profile)

        logger.info(f"Found {len(profiles)} configuration profiles")
        return profiles

    except ConfigurationError as e:
        logger.error(f"Error listing profiles: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list profiles: {e!s}",
        )


@router.post("/profiles", response_model=ProfileResponse, status_code=status.HTTP_201_CREATED)
async def create_profile(request: ProfileCreateRequest):
    """Create a new configuration profile.

    Args:
        request: Profile creation request

    Returns:
        Created configuration profile
    """
    try:
        logger.info(f"Creating configuration profile: {request.name}")

        # Create ConfigProfile object
        profile = ConfigProfile(
            id=str(uuid4()),
            name=request.name,
            description=request.description,
            sdrConfig=SDRConfig(**request.sdrConfig),
            signalConfig=SignalConfig(**request.signalConfig),
            homingConfig=HomingConfig(**request.homingConfig),
            isDefault=request.isDefault,
        )

        # Validate profile
        validation = config_service.validate_profile(profile)
        if not validation["valid"]:
            logger.warning(f"Invalid profile configuration: {validation['errors']}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid profile configuration: {', '.join(validation['errors'])}",
            )

        # Save to YAML file
        if not config_service.save_profile(profile):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save profile"
            )

        # Save to database
        profile_data = {
            "id": profile.id,
            "name": profile.name,
            "description": profile.description,
            "sdrConfig": profile.sdrConfig.__dict__,
            "signalConfig": profile.signalConfig.__dict__,
            "homingConfig": profile.homingConfig.__dict__,
            "isDefault": profile.isDefault,
            "createdAt": profile.createdAt.isoformat(),
            "updatedAt": profile.updatedAt.isoformat(),
        }
        profile_db.insert_profile(profile_data)

        logger.info(f"Successfully created profile: {profile.name}")

        return ProfileResponse(**profile_data)

    except HTTPException:
        raise
    except ConfigurationError as e:
        logger.error(f"Error creating profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create profile: {e!s}",
        )


@router.put("/profiles/{profile_id}", response_model=ProfileResponse)
async def update_profile(profile_id: str, request: ProfileUpdateRequest):
    """Update an existing configuration profile.

    Args:
        profile_id: ID of the profile to update
        request: Profile update request

    Returns:
        Updated configuration profile
    """
    try:
        logger.info(f"Updating configuration profile: {profile_id}")

        # Load existing profile using helper function
        existing_profile = await _get_profile_by_id(profile_id)
        if not existing_profile:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")

        # Update profile
        profile = ConfigProfile(
            id=profile_id,
            name=request.name,
            description=request.description,
            sdrConfig=SDRConfig(**request.sdrConfig),
            signalConfig=SignalConfig(**request.signalConfig),
            homingConfig=HomingConfig(**request.homingConfig),
            isDefault=request.isDefault,
            createdAt=existing_profile.createdAt if existing_profile else None,
        )

        # Validate profile
        validation = config_service.validate_profile(profile)
        if not validation["valid"]:
            logger.warning(f"Invalid profile configuration: {validation['errors']}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid profile configuration: {', '.join(validation['errors'])}",
            )

        # Save to YAML file
        if not config_service.save_profile(profile):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save profile"
            )

        # Update in database
        profile_data = {
            "name": profile.name,
            "description": profile.description,
            "sdrConfig": profile.sdrConfig.__dict__,
            "signalConfig": profile.signalConfig.__dict__,
            "homingConfig": profile.homingConfig.__dict__,
            "isDefault": profile.isDefault,
        }
        profile_db.update_profile(profile_id, profile_data)

        logger.info(f"Successfully updated profile: {profile.name}")

        profile_data["id"] = profile_id
        profile_data["createdAt"] = profile.createdAt.isoformat()
        profile_data["updatedAt"] = profile.updatedAt.isoformat()

        return ProfileResponse(**profile_data)

    except HTTPException:
        raise
    except ConfigurationError as e:
        logger.error(f"Error updating profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update profile: {e!s}",
        )


@router.post("/profiles/{profile_id}/activate")
async def activate_profile(profile_id: str):
    """Apply a configuration profile to the system.

    Args:
        profile_id: ID of the profile to activate

    Returns:
        Success status
    """
    try:
        logger.info(f"Activating configuration profile: {profile_id}")

        # Load profile using helper function
        profile = await _get_profile_by_id(profile_id)
        if not profile:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")

        # Apply configuration to SDR and Signal Processing services

        # Convert profile SDR config to SDRConfigModel
        # Note: Configuration would normally be applied to the SDR service here
        # For now, we just log the configuration
        # Apply to SDR service if available
        try:
            # This would normally be injected or retrieved from app state
            # For now, we'll just log the configuration
            logger.info(f"Applying profile settings: {profile.name}")
            logger.info(
                f"SDR Config: frequency={profile.sdrConfig.frequency}, "
                f"sampleRate={profile.sdrConfig.sampleRate}, gain={profile.sdrConfig.gain}"
            )
            logger.info(
                f"Signal Config: triggerThreshold={profile.signalConfig.triggerThreshold}, "
                f"dropThreshold={profile.signalConfig.dropThreshold}"
            )
            logger.info(
                f"Homing Config: forwardVelocityMax={profile.homingConfig.forwardVelocityMax}, "
                f"yawRateMax={profile.homingConfig.yawRateMax}"
            )
        except SDRError as e:
            logger.warning(f"Could not apply config to SDR service: {e}")

        # Broadcast configuration change via WebSocket
        await broadcast_message(
            {
                "type": "config",
                "action": "profile_activated",
                "profile": {
                    "id": profile.id,
                    "name": profile.name,
                    "sdrConfig": profile.sdrConfig.__dict__,
                    "signalConfig": profile.signalConfig.__dict__,
                    "homingConfig": profile.homingConfig.__dict__,
                },
            }
        )

        # Set as default if requested
        if profile.isDefault:
            config_service.set_default_profile(profile.name)
            profile_db.set_default_profile(profile.id)

        logger.info(f"Successfully activated profile: {profile.name}")

        return {"status": "success", "message": f"Profile {profile.name} activated successfully"}

    except HTTPException:
        raise
    except ConfigurationError as e:
        logger.error(f"Error activating profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to activate profile: {e!s}",
        )


@router.delete("/profiles/{profile_id}")
async def delete_profile(profile_id: str):
    """Delete a configuration profile.

    Args:
        profile_id: ID of the profile to delete

    Returns:
        Success status
    """
    try:
        logger.info(f"Deleting configuration profile: {profile_id}")

        # Find and delete from YAML
        deleted_yaml = False
        profile = await _get_profile_by_id(profile_id)
        if profile:
            deleted_yaml = config_service.delete_profile(profile.name)

        # Delete from database
        deleted_db = profile_db.delete_profile(profile_id)

        if not deleted_yaml and not deleted_db:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")

        logger.info(f"Successfully deleted profile: {profile_id}")

        return {"status": "success", "message": "Profile deleted successfully"}

    except HTTPException:
        raise
    except ConfigurationError as e:
        logger.error(f"Error deleting profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete profile: {e!s}",
        )
