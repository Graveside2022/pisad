"""Fixed tests for analytics API routes."""

import json
import csv
from datetime import datetime
from uuid import uuid4, UUID
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from pathlib import Path
import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from src.backend.api.routes.analytics import (
    get_performance_metrics,
    get_replay_data,
    control_replay,
    export_data,
    MetricsResponse,
    ExportRequest,
    ReplayControlRequest,
)
from src.backend.services.performance_analytics import MissionPerformanceMetrics


class TestExportRequest:
    """Tests for ExportRequest model validation."""
    
    def test_export_request_valid(self):
        """Test valid export request."""
        request = ExportRequest(
            mission_id=uuid4(),
            format="csv",
            data_type="telemetry"
        )
        assert request.format == "csv"
        assert request.data_type == "telemetry"
    
    def test_export_request_invalid_format(self):
        """Test invalid export format."""
        with pytest.raises(ValidationError):
            ExportRequest(
                mission_id=uuid4(),
                format="xml",  # Invalid format
                data_type="telemetry"
            )
    
    def test_export_request_invalid_data_type(self):
        """Test invalid data type."""
        with pytest.raises(ValidationError):
            ExportRequest(
                mission_id=uuid4(),
                format="csv",
                data_type="invalid"  # Invalid data type
            )


class TestReplayControlRequest:
    """Tests for ReplayControlRequest model validation."""
    
    def test_replay_control_valid(self):
        """Test valid replay control request."""
        request = ReplayControlRequest(
            action="play",
            speed=2.0,
            position=100
        )
        assert request.action == "play"
        assert request.speed == 2.0
        assert request.position == 100
    
    def test_replay_control_invalid_action(self):
        """Test invalid replay action."""
        with pytest.raises(ValidationError):
            ReplayControlRequest(
                action="invalid",  # Invalid action
                speed=1.0
            )


class TestGetPerformanceMetrics:
    """Tests for get_performance_metrics endpoint."""
    
    @pytest.mark.asyncio
    async def test_get_metrics_success(self):
        """Test successful metrics retrieval."""
        mission_id = uuid4()
        
        # Mock the entire file system interaction
        with patch("src.backend.api.routes.analytics.Path") as mock_path_class:
            # Create mock for the data directory path
            mock_data_dir = MagicMock()
            
            # Create mock for telemetry file
            mock_telemetry_file = MagicMock()
            mock_telemetry_file.exists.return_value = True
            
            # Create mock for detections file
            mock_detections_file = MagicMock()
            mock_detections_file.exists.return_value = True
            
            # Set up the path chain: Path("data/missions") / str(mission_id)
            mock_path_class.return_value.__truediv__.return_value.__truediv__.return_value = mock_data_dir
            
            # Set up data_dir / "telemetry.csv" and data_dir / "detections.json"
            def mock_truediv(filename):
                if filename == "telemetry.csv":
                    return mock_telemetry_file
                elif filename == "detections.json":
                    return mock_detections_file
                return MagicMock()
            
            mock_data_dir.__truediv__.side_effect = mock_truediv
            
            # Mock file reading
            telemetry_data = "timestamp,rssi,snr\n2024-01-01T00:00:00,-70,10\n"
            detections_data = '[{"id": "1", "timestamp": "2024-01-01T00:00:00"}]'
            
            def mock_open_func(file_path, *args, **kwargs):
                if str(file_path) == str(mock_telemetry_file):
                    return mock_open(read_data=telemetry_data)()
                elif str(file_path) == str(mock_detections_file):
                    return mock_open(read_data=detections_data)()
                return mock_open()()
            
            with patch("builtins.open", side_effect=mock_open_func):
                with patch("src.backend.api.routes.analytics.analytics_service") as mock_analytics:
                    mock_report = MissionPerformanceMetrics(
                        mission_id=mission_id,
                        detection_metrics={"total_detections": 5},
                        approach_metrics={"final_distance_m": 10},
                        search_metrics={"area_covered_km2": 2.5},
                        false_positive_analysis={"precision": 0.95},
                        environmental_correlation={"weather_impact_score": 0.8},
                        baseline_comparison={"time_improvement_percent": 50},
                        overall_score=85.0,
                        recommendations=["Increase search altitude"],
                    )
                    mock_analytics.generate_performance_report.return_value = mock_report
                    
                    result = await get_performance_metrics(mission_id)
                    
                    assert isinstance(result, MetricsResponse)
                    assert result.mission_id == mission_id
                    assert result.overall_score == 85.0
                    assert len(result.recommendations) == 1
    
    @pytest.mark.asyncio
    async def test_get_metrics_mission_not_found(self):
        """Test metrics retrieval for non-existent mission."""
        mission_id = uuid4()
        
        with patch("src.backend.api.routes.analytics.Path") as mock_path_class:
            # Create mock for telemetry file that doesn't exist
            mock_telemetry_file = MagicMock()
            mock_telemetry_file.exists.return_value = False
            
            # Set up the path chain
            mock_data_dir = MagicMock()
            mock_data_dir.__truediv__.return_value = mock_telemetry_file
            mock_path_class.return_value.__truediv__.return_value.__truediv__.return_value = mock_data_dir
            
            with pytest.raises(HTTPException) as exc_info:
                await get_performance_metrics(mission_id)
            
            assert exc_info.value.status_code == 404
    
    @pytest.mark.asyncio
    async def test_get_metrics_without_recommendations(self):
        """Test metrics retrieval without recommendations."""
        mission_id = uuid4()
        
        with patch("src.backend.api.routes.analytics.Path") as mock_path_class:
            # Set up mocks similar to success case
            mock_data_dir = MagicMock()
            mock_telemetry_file = MagicMock()
            mock_telemetry_file.exists.return_value = True
            mock_detections_file = MagicMock()
            mock_detections_file.exists.return_value = False  # No detections file
            
            mock_path_class.return_value.__truediv__.return_value.__truediv__.return_value = mock_data_dir
            
            def mock_truediv(filename):
                if filename == "telemetry.csv":
                    return mock_telemetry_file
                elif filename == "detections.json":
                    return mock_detections_file
                return MagicMock()
            
            mock_data_dir.__truediv__.side_effect = mock_truediv
            
            telemetry_data = "timestamp,rssi,snr\n2024-01-01T00:00:00,-70,10\n"
            
            with patch("builtins.open", mock_open(read_data=telemetry_data)):
                with patch("src.backend.api.routes.analytics.analytics_service") as mock_analytics:
                    mock_report = MissionPerformanceMetrics(
                        mission_id=mission_id,
                        detection_metrics={"total_detections": 0},
                        approach_metrics={"final_distance_m": 100},
                        search_metrics={"area_covered_km2": 2.0},
                        false_positive_analysis={"precision": 0.0},
                        environmental_correlation={"weather_impact_score": 0.0},
                        baseline_comparison={"time_improvement_percent": 0},
                        overall_score=50.0,
                        recommendations=[],
                    )
                    mock_analytics.generate_performance_report.return_value = mock_report
                    
                    result = await get_performance_metrics(mission_id, include_recommendations=False)
                    
                    assert isinstance(result, MetricsResponse)
                    assert result.mission_id == mission_id
                    assert len(result.recommendations) == 0


class TestGetReplayData:
    """Tests for get_replay_data endpoint."""
    
    @pytest.mark.asyncio
    async def test_get_replay_data_success(self):
        """Test successful replay data retrieval."""
        mission_id = uuid4()
        
        with patch("src.backend.api.routes.analytics.replay_service") as mock_replay:
            mock_replay.load_mission.return_value = True
            mock_replay.get_telemetry.return_value = [
                {"timestamp": "2024-01-01T00:00:00", "rssi": -70}
            ]
            mock_replay.get_detection_events.return_value = [
                {"id": "1", "timestamp": "2024-01-01T00:00:00"}
            ]
            
            result = await get_replay_data(mission_id)
            
            assert result is not None
            assert "telemetry" in result
            assert "detections" in result
    
    @pytest.mark.asyncio
    async def test_get_replay_data_already_loaded(self):
        """Test replay data retrieval when already loaded."""
        mission_id = uuid4()
        
        with patch("src.backend.api.routes.analytics.replay_service") as mock_replay:
            mock_replay.load_mission.return_value = False  # Already loaded
            mock_replay.get_telemetry.return_value = []
            mock_replay.get_detection_events.return_value = []
            
            result = await get_replay_data(mission_id)
            
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_get_replay_data_load_failure(self):
        """Test replay data retrieval failure."""
        mission_id = uuid4()
        
        with patch("src.backend.api.routes.analytics.replay_service") as mock_replay:
            mock_replay.load_mission.side_effect = Exception("Load failed")
            
            with pytest.raises(HTTPException) as exc_info:
                await get_replay_data(mission_id)
            
            assert exc_info.value.status_code == 500


class TestControlReplay:
    """Tests for control_replay endpoint."""
    
    @pytest.mark.asyncio
    async def test_control_replay_play(self):
        """Test replay play control."""
        mission_id = uuid4()
        request = ReplayControlRequest(action="play")
        
        with patch("src.backend.api.routes.analytics.replay_service") as mock_replay:
            mock_replay.play.return_value = {"status": "playing", "position": 100}
            
            result = await control_replay(mission_id, request)
            
            assert result is not None
            mock_replay.play.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_control_replay_pause(self):
        """Test replay pause control."""
        mission_id = uuid4()
        request = ReplayControlRequest(action="pause")
        
        with patch("src.backend.api.routes.analytics.replay_service") as mock_replay:
            mock_replay.pause.return_value = {"status": "paused", "position": 200}
            
            result = await control_replay(mission_id, request)
            
            assert result is not None
            mock_replay.pause.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_control_replay_seek(self):
        """Test replay seek control."""
        mission_id = uuid4()
        request = ReplayControlRequest(action="seek", position=500)
        
        with patch("src.backend.api.routes.analytics.replay_service") as mock_replay:
            mock_replay.seek.return_value = {"status": "paused", "position": 500}
            
            result = await control_replay(mission_id, request)
            
            assert result is not None
            mock_replay.seek.assert_called_once_with(mission_id, 500)
    
    @pytest.mark.asyncio
    async def test_control_replay_mission_not_loaded(self):
        """Test replay control when mission not loaded."""
        mission_id = uuid4()
        request = ReplayControlRequest(action="play")
        
        with patch("src.backend.api.routes.analytics.replay_service") as mock_replay:
            mock_replay.play.side_effect = ValueError("Mission not loaded")
            
            with pytest.raises(ValueError):
                await control_replay(mission_id, request)


class TestExportData:
    """Tests for export_data endpoint."""
    
    @pytest.mark.asyncio
    async def test_export_telemetry_csv(self):
        """Test telemetry CSV export."""
        mission_id = uuid4()
        request = ExportRequest(mission_id=mission_id, format="csv", data_type="telemetry")
        
        with patch("src.backend.api.routes.analytics.Path") as mock_path_class:
            # Set up path mocks
            mock_telemetry_file = MagicMock()
            mock_telemetry_file.exists.return_value = True
            
            mock_data_dir = MagicMock()
            mock_data_dir.__truediv__.return_value = mock_telemetry_file
            mock_path_class.return_value.__truediv__.return_value.__truediv__.return_value = mock_data_dir
            
            # Mock file reading
            telemetry_data = "timestamp,rssi,snr\n2024-01-01T00:00:00,-70,10\n"
            
            with patch("builtins.open", mock_open(read_data=telemetry_data)):
                result = await export_data(request)
                
                assert result is not None