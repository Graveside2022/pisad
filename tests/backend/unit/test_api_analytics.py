"""Unit tests for analytics API routes."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.backend.api.routes.analytics import (
    ExportRequest,
    MetricsResponse,
    ReplayControlRequest,
    _filter_by_time_range,
    _sanitize_data,
    control_replay,
    export_data,
    get_mission_report,
    get_performance_metrics,
    get_replay_data,
    get_system_recommendations,
)
from src.backend.services.mission_replay_service import PlaybackSpeed
from src.backend.services.performance_analytics import MissionPerformanceMetrics

# Mark all tests in this module as unit tests
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class TestExportRequest:
    """Test cases for ExportRequest model validation."""

    def test_export_request_valid(self):
        """Test valid export request."""
        request = ExportRequest(
            mission_id=uuid4(),
            format="csv",
            data_type="telemetry",
            include_sensitive=False,
        )
        assert request.format == "csv"
        assert request.data_type == "telemetry"

    def test_export_request_invalid_format(self):
        """Test invalid export format."""
        with pytest.raises(ValueError):
            ExportRequest(
                mission_id=uuid4(),
                format="xml",  # Invalid format
                data_type="telemetry",
            )

    def test_export_request_invalid_data_type(self):
        """Test invalid data type."""
        with pytest.raises(ValueError):
            ExportRequest(
                mission_id=uuid4(),
                format="csv",
                data_type="invalid",  # Invalid data type
            )


class TestReplayControlRequest:
    """Test cases for ReplayControlRequest model validation."""

    def test_replay_control_valid(self):
        """Test valid replay control request."""
        request = ReplayControlRequest(
            action="play",
            speed=2.0,
        )
        assert request.action == "play"
        assert request.speed == 2.0

    def test_replay_control_invalid_action(self):
        """Test invalid replay action."""
        with pytest.raises(ValueError):
            ReplayControlRequest(action="rewind")  # Invalid action


class TestGetPerformanceMetrics:
    """Test cases for get_performance_metrics endpoint."""

    @pytest.mark.asyncio
    async def test_get_metrics_success(self, tmp_path):
        """Test successful metrics retrieval."""
        mission_id = uuid4()
        
        # Create test data
        data_dir = tmp_path / "data" / "missions" / str(mission_id)
        data_dir.mkdir(parents=True)
        
        telemetry_file = data_dir / "telemetry.csv"
        telemetry_file.write_text("timestamp,rssi,snr\n2024-01-01T00:00:00,-70,10\n")
        
        detections_file = data_dir / "detections.json"
        detections_file.write_text('[{"id": "1", "timestamp": "2024-01-01T00:00:00"}]')
        
        with patch("src.backend.api.routes.analytics.Path") as mock_path:
            mock_path.return_value = tmp_path / "data" / "missions" / str(mission_id)
            
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
        
        with patch("src.backend.api.routes.analytics.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            
            with pytest.raises(HTTPException) as exc_info:
                await get_performance_metrics(mission_id)
            
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_metrics_without_recommendations(self, tmp_path):
        """Test metrics retrieval without recommendations."""
        mission_id = uuid4()
        
        # Create test data
        data_dir = tmp_path / "data" / "missions" / str(mission_id)
        data_dir.mkdir(parents=True)
        telemetry_file = data_dir / "telemetry.csv"
        telemetry_file.write_text("timestamp,rssi\n2024-01-01T00:00:00,-70\n")
        
        with patch("src.backend.api.routes.analytics.Path") as mock_path:
            mock_path.return_value = tmp_path / "data" / "missions" / str(mission_id)
            
            with patch("src.backend.api.routes.analytics.analytics_service") as mock_analytics:
                mock_report = MissionPerformanceMetrics(
                    mission_id=mission_id,
                    detection_metrics={},
                    approach_metrics={},
                    search_metrics={},
                    false_positive_analysis={},
                    environmental_correlation={},
                    baseline_comparison={},
                    overall_score=75.0,
                    recommendations=["Recommendation 1", "Recommendation 2"],
                )
                mock_analytics.generate_performance_report.return_value = mock_report
                
                result = await get_performance_metrics(mission_id, include_recommendations=False)
                
                assert result.recommendations == []


class TestGetReplayData:
    """Test cases for get_replay_data endpoint."""

    @pytest.mark.asyncio
    async def test_get_replay_data_success(self):
        """Test successful replay data retrieval."""
        mission_id = uuid4()
        
        with patch("src.backend.api.routes.analytics.replay_service") as mock_replay:
            mock_replay.mission_id = None  # Not loaded
            mock_replay.load_mission_data = AsyncMock(return_value=True)
            mock_replay.get_status.return_value = {"status": "ready"}
            mock_replay.get_timeline_range.return_value = (
                datetime(2024, 1, 1, 0, 0),
                datetime(2024, 1, 1, 1, 0),
            )
            
            # Mock the Path checks
            with patch("src.backend.api.routes.analytics.Path") as mock_path:
                mock_file = MagicMock()
                mock_file.exists.return_value = True
                mock_path.return_value.__truediv__.return_value.__truediv__.return_value = mock_file
                
                result = await get_replay_data(mission_id)
                
                assert result["status"]["status"] == "ready"
                assert "timeline_range" in result
                mock_replay.load_mission_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_replay_data_already_loaded(self):
        """Test replay data when mission already loaded."""
        mission_id = uuid4()
        
        with patch("src.backend.api.routes.analytics.replay_service") as mock_replay:
            mock_replay.mission_id = mission_id  # Already loaded
            mock_replay.get_status.return_value = {"status": "playing"}
            mock_replay.get_timeline_range.return_value = None
            
            result = await get_replay_data(mission_id)
            
            assert result["status"]["status"] == "playing"
            assert result["timeline_range"]["start"] is None

    @pytest.mark.asyncio
    async def test_get_replay_data_load_failure(self):
        """Test replay data load failure."""
        mission_id = uuid4()
        
        with patch("src.backend.api.routes.analytics.replay_service") as mock_replay:
            mock_replay.mission_id = None
            mock_replay.load_mission_data = AsyncMock(return_value=False)
            
            # Mock the Path checks
            with patch("src.backend.api.routes.analytics.Path") as mock_path:
                mock_file = MagicMock()
                mock_file.exists.return_value = True
                mock_path.return_value.__truediv__.return_value.__truediv__.return_value = mock_file
                
                with pytest.raises(HTTPException) as exc_info:
                    await get_replay_data(mission_id)
                
                assert exc_info.value.status_code == 500


class TestControlReplay:
    """Test cases for control_replay endpoint."""

    @pytest.mark.asyncio
    async def test_control_replay_play(self):
        """Test replay play control."""
        mission_id = uuid4()
        request = ReplayControlRequest(action="play")
        
        with patch("src.backend.api.routes.analytics.replay_service") as mock_replay:
            mock_replay.mission_id = mission_id
            mock_replay.play = AsyncMock()
            mock_replay.get_status.return_value = {"status": "playing"}
            
            result = await control_replay(mission_id, request)
            
            assert result["status"] == "playing"
            mock_replay.play.assert_called_once()

    @pytest.mark.asyncio
    async def test_control_replay_pause(self):
        """Test replay pause control."""
        mission_id = uuid4()
        request = ReplayControlRequest(action="pause")
        
        with patch("src.backend.api.routes.analytics.replay_service") as mock_replay:
            mock_replay.mission_id = mission_id
            mock_replay.pause = AsyncMock()
            mock_replay.get_status.return_value = {"status": "paused"}
            
            result = await control_replay(mission_id, request)
            
            assert result["status"] == "paused"
            mock_replay.pause.assert_called_once()

    @pytest.mark.asyncio
    async def test_control_replay_seek(self):
        """Test replay seek control."""
        mission_id = uuid4()
        request = ReplayControlRequest(action="seek", position=100)
        
        with patch("src.backend.api.routes.analytics.replay_service") as mock_replay:
            mock_replay.mission_id = mission_id
            mock_replay.seek = AsyncMock()
            mock_replay.get_status.return_value = {"status": "ready", "position": 100}
            
            result = await control_replay(mission_id, request)
            
            mock_replay.seek.assert_called_once_with(100)

    @pytest.mark.asyncio
    async def test_control_replay_set_speed(self):
        """Test replay speed control."""
        mission_id = uuid4()
        request = ReplayControlRequest(action="play", speed=2.0)
        
        with patch("src.backend.api.routes.analytics.replay_service") as mock_replay:
            mock_replay.mission_id = mission_id
            mock_replay.play = AsyncMock()
            mock_replay.set_speed = AsyncMock()
            mock_replay.get_status.return_value = {"status": "playing", "speed": 2.0}
            
            result = await control_replay(mission_id, request)
            
            mock_replay.set_speed.assert_called_once()

    @pytest.mark.asyncio
    async def test_control_replay_mission_not_loaded(self):
        """Test replay control when mission not loaded."""
        mission_id = uuid4()
        request = ReplayControlRequest(action="play")
        
        with patch("src.backend.api.routes.analytics.replay_service") as mock_replay:
            mock_replay.mission_id = uuid4()  # Different mission
            
            with pytest.raises(HTTPException) as exc_info:
                await control_replay(mission_id, request)
            
            assert exc_info.value.status_code == 400


class TestExportData:
    """Test cases for export_data endpoint."""

    @pytest.mark.asyncio
    async def test_export_telemetry_csv(self, tmp_path):
        """Test telemetry export as CSV."""
        mission_id = uuid4()
        
        # Create test data
        data_dir = tmp_path / "data" / "missions" / str(mission_id)
        data_dir.mkdir(parents=True)
        telemetry_file = data_dir / "telemetry.csv"
        telemetry_file.write_text("timestamp,rssi,snr\n2024-01-01T00:00:00,-70,10\n2024-01-01T00:01:00,-65,12\n")
        
        request = ExportRequest(
            mission_id=mission_id,
            format="csv",
            data_type="telemetry",
        )
        
        with patch("src.backend.api.routes.analytics.Path") as mock_path:
            mock_path.return_value = tmp_path / "data" / "missions" / str(mission_id)
            
            response = await export_data(request)
            
            assert response.media_type == "text/csv"
            assert "timestamp,rssi,snr" in response.body.decode()

    @pytest.mark.asyncio
    async def test_export_all_json(self, tmp_path):
        """Test export all data as JSON."""
        mission_id = uuid4()
        
        # Create test data
        data_dir = tmp_path / "data" / "missions" / str(mission_id)
        data_dir.mkdir(parents=True)
        
        telemetry_file = data_dir / "telemetry.csv"
        telemetry_file.write_text("timestamp,rssi\n2024-01-01T00:00:00,-70\n")
        
        detections_file = data_dir / "detections.json"
        detections_file.write_text('[{"id": "1", "timestamp": "2024-01-01T00:00:00"}]')
        
        request = ExportRequest(
            mission_id=mission_id,
            format="json",
            data_type="all",
            include_sensitive=True,
        )
        
        with patch("src.backend.api.routes.analytics.Path") as mock_path:
            mock_path.return_value = tmp_path / "data" / "missions" / str(mission_id)
            
            with patch("src.backend.api.routes.analytics.get_performance_metrics") as mock_metrics:
                mock_metrics.return_value = MagicMock(
                    model_dump=MagicMock(return_value={"overall_score": 85})
                )
                
                response = await export_data(request)
                
                assert response.media_type == "application/json"
                data = json.loads(response.body.decode())
                assert "telemetry" in data
                assert "detections" in data
                assert "metrics" in data

    @pytest.mark.asyncio
    async def test_export_with_time_filter(self, tmp_path):
        """Test export with time range filter."""
        mission_id = uuid4()
        
        # Create test data
        data_dir = tmp_path / "data" / "missions" / str(mission_id)
        data_dir.mkdir(parents=True)
        telemetry_file = data_dir / "telemetry.csv"
        telemetry_file.write_text(
            "timestamp,rssi\n"
            "2024-01-01T00:00:00,-70\n"
            "2024-01-01T00:30:00,-65\n"
            "2024-01-01T01:00:00,-60\n"
        )
        
        request = ExportRequest(
            mission_id=mission_id,
            format="json",
            data_type="telemetry",
            start_time=datetime(2024, 1, 1, 0, 15),
            end_time=datetime(2024, 1, 1, 0, 45),
        )
        
        with patch("src.backend.api.routes.analytics.Path") as mock_path:
            mock_path.return_value = tmp_path / "data" / "missions" / str(mission_id)
            
            response = await export_data(request)
            
            data = json.loads(response.body.decode())
            assert len(data["telemetry"]) == 1
            assert data["telemetry"][0]["timestamp"] == "2024-01-01T00:30:00"

    @pytest.mark.asyncio
    async def test_export_sanitized_data(self, tmp_path):
        """Test export with sensitive data sanitization."""
        mission_id = uuid4()
        
        # Create test data with sensitive fields
        data_dir = tmp_path / "data" / "missions" / str(mission_id)
        data_dir.mkdir(parents=True)
        telemetry_file = data_dir / "telemetry.csv"
        telemetry_file.write_text(
            "timestamp,rssi,operator_id,api_key\n"
            "2024-01-01T00:00:00,-70,user123,secret123\n"
        )
        
        request = ExportRequest(
            mission_id=mission_id,
            format="json",
            data_type="telemetry",
            include_sensitive=False,
        )
        
        with patch("src.backend.api.routes.analytics.Path") as mock_path:
            mock_path.return_value = tmp_path / "data" / "missions" / str(mission_id)
            
            response = await export_data(request)
            
            data = json.loads(response.body.decode())
            telemetry = data["telemetry"][0]
            assert "operator_id" not in telemetry
            assert "api_key" not in telemetry
            assert "timestamp" in telemetry
            assert "rssi" in telemetry

    @pytest.mark.asyncio
    async def test_export_mission_not_found(self):
        """Test export for non-existent mission."""
        mission_id = uuid4()
        request = ExportRequest(
            mission_id=mission_id,
            format="csv",
            data_type="telemetry",
        )
        
        with patch("src.backend.api.routes.analytics.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            
            with pytest.raises(HTTPException) as exc_info:
                await export_data(request)
            
            assert exc_info.value.status_code == 404


class TestGetMissionReport:
    """Test cases for get_mission_report endpoint."""

    @pytest.mark.asyncio
    async def test_get_json_report(self):
        """Test JSON report generation."""
        mission_id = uuid4()
        
        with patch("src.backend.api.routes.analytics.get_performance_metrics") as mock_metrics:
            mock_metrics.return_value = MagicMock(
                model_dump=MagicMock(return_value={
                    "mission_id": str(mission_id),
                    "overall_score": 85,
                    "recommendations": ["Increase altitude"],
                })
            )
            
            response = await get_mission_report(mission_id, format="json")
            
            assert response.media_type == "application/json"
            data = json.loads(response.body.decode())
            assert data["mission_id"] == str(mission_id)
            assert "generated_at" in data
            assert "metrics" in data

    @pytest.mark.asyncio
    async def test_get_pdf_report_not_implemented(self):
        """Test PDF report generation (not implemented)."""
        mission_id = uuid4()
        
        with pytest.raises(HTTPException) as exc_info:
            await get_mission_report(mission_id, format="pdf")
        
        assert exc_info.value.status_code == 501


class TestGetSystemRecommendations:
    """Test cases for get_system_recommendations endpoint."""

    @pytest.mark.asyncio
    async def test_get_recommendations_with_missions(self, tmp_path):
        """Test system recommendations with existing missions."""
        # Create test mission directories
        data_dir = tmp_path / "data" / "missions"
        
        mission_id1 = uuid4()
        mission_dir1 = data_dir / str(mission_id1)
        mission_dir1.mkdir(parents=True)
        
        mission_id2 = uuid4()
        mission_dir2 = data_dir / str(mission_id2)
        mission_dir2.mkdir(parents=True)
        
        with patch("src.backend.api.routes.analytics.Path") as mock_path:
            mock_path.return_value = data_dir
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.iterdir.return_value = [mission_dir1, mission_dir2]
            
            with patch("src.backend.api.routes.analytics.get_performance_metrics") as mock_metrics:
                # Return different recommendations for each mission
                mock_metrics.side_effect = [
                    MagicMock(recommendations=["Increase altitude", "Reduce speed"]),
                    MagicMock(recommendations=["Increase altitude", "Improve SNR"]),
                ]
                
                result = await get_system_recommendations()
                
                assert result["missions_analyzed"] == 2
                assert "Increase altitude" in result["top_recommendations"]
                assert result["recommendation_frequency"]["Increase altitude"] == 2

    @pytest.mark.asyncio
    async def test_get_recommendations_no_missions(self):
        """Test system recommendations with no missions."""
        with patch("src.backend.api.routes.analytics.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            
            result = await get_system_recommendations()
            
            assert result["missions_analyzed"] == 0
            assert result["top_recommendations"] == []
            assert result["recommendation_frequency"] == {}

    @pytest.mark.asyncio
    async def test_get_recommendations_with_invalid_missions(self, tmp_path):
        """Test system recommendations with some invalid missions."""
        # Create test mission directories
        data_dir = tmp_path / "data" / "missions"
        
        mission_id1 = uuid4()
        mission_dir1 = data_dir / str(mission_id1)
        mission_dir1.mkdir(parents=True)
        
        # Create invalid mission directory (not a UUID)
        invalid_dir = data_dir / "invalid_mission"
        invalid_dir.mkdir(parents=True)
        
        with patch("src.backend.api.routes.analytics.Path") as mock_path:
            mock_path.return_value = data_dir
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.iterdir.return_value = [mission_dir1, invalid_dir]
            
            with patch("src.backend.api.routes.analytics.get_performance_metrics") as mock_metrics:
                mock_metrics.return_value = MagicMock(recommendations=["Test recommendation"])
                
                result = await get_system_recommendations()
                
                # Should only process valid mission
                assert result["missions_analyzed"] == 1


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_filter_by_time_range_no_filter(self):
        """Test filtering with no time range."""
        data = [
            {"timestamp": "2024-01-01T00:00:00", "value": 1},
            {"timestamp": "2024-01-01T01:00:00", "value": 2},
        ]
        
        result = _filter_by_time_range(data, None, None)
        assert len(result) == 2

    def test_filter_by_time_range_with_start(self):
        """Test filtering with start time."""
        data = [
            {"timestamp": "2024-01-01T00:00:00", "value": 1},
            {"timestamp": "2024-01-01T01:00:00", "value": 2},
            {"timestamp": "2024-01-01T02:00:00", "value": 3},
        ]
        
        start_time = datetime(2024, 1, 1, 1, 0)
        result = _filter_by_time_range(data, start_time, None)
        assert len(result) == 2
        assert result[0]["value"] == 2

    def test_filter_by_time_range_with_end(self):
        """Test filtering with end time."""
        data = [
            {"timestamp": "2024-01-01T00:00:00", "value": 1},
            {"timestamp": "2024-01-01T01:00:00", "value": 2},
            {"timestamp": "2024-01-01T02:00:00", "value": 3},
        ]
        
        end_time = datetime(2024, 1, 1, 1, 30)
        result = _filter_by_time_range(data, None, end_time)
        assert len(result) == 2
        assert result[-1]["value"] == 2

    def test_filter_by_time_range_with_both(self):
        """Test filtering with both start and end time."""
        data = [
            {"timestamp": "2024-01-01T00:00:00", "value": 1},
            {"timestamp": "2024-01-01T01:00:00", "value": 2},
            {"timestamp": "2024-01-01T02:00:00", "value": 3},
            {"timestamp": "2024-01-01T03:00:00", "value": 4},
        ]
        
        start_time = datetime(2024, 1, 1, 0, 30)
        end_time = datetime(2024, 1, 1, 2, 30)
        result = _filter_by_time_range(data, start_time, end_time)
        assert len(result) == 2
        assert result[0]["value"] == 2
        assert result[1]["value"] == 3

    def test_filter_by_time_range_invalid_timestamp(self):
        """Test filtering with invalid timestamps."""
        data = [
            {"timestamp": "2024-01-01T00:00:00", "value": 1},
            {"timestamp": "invalid", "value": 2},
            {"value": 3},  # No timestamp
            {"timestamp": "2024-01-01T02:00:00", "value": 4},
        ]
        
        result = _filter_by_time_range(data, None, None)
        assert len(result) == 2  # Only valid timestamps

    def test_sanitize_data_removes_sensitive(self):
        """Test sanitization removes sensitive fields."""
        data = {
            "mission_id": "123",
            "operator_id": "user123",
            "api_key": "secret",
            "telemetry": {
                "rssi": -70,
                "password": "hidden",
            },
            "tokens": ["token1", "token2"],
        }
        
        result = _sanitize_data(data)
        
        assert "mission_id" in result
        assert "operator_id" not in result
        assert "api_key" not in result
        assert "telemetry" in result
        assert "rssi" in result["telemetry"]
        assert "password" not in result["telemetry"]
        assert "tokens" not in result

    def test_sanitize_data_nested_lists(self):
        """Test sanitization of nested lists."""
        data = {
            "missions": [
                {"id": "1", "operator_name": "John"},
                {"id": "2", "secret": "hidden"},
            ]
        }
        
        result = _sanitize_data(data)
        
        assert len(result["missions"]) == 2
        assert "operator_name" not in result["missions"][0]
        assert "secret" not in result["missions"][1]
        assert "id" in result["missions"][0]
        assert "id" in result["missions"][1]

    def test_sanitize_data_case_insensitive(self):
        """Test sanitization is case insensitive."""
        data = {
            "API_KEY": "key1",
            "Api_Key": "key2",
            "apikey": "key3",
            "data": "safe",
        }
        
        result = _sanitize_data(data)
        
        assert "API_KEY" not in result
        assert "Api_Key" not in result
        assert "apikey" not in result
        assert "data" in result