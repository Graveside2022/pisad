import React, { useEffect, useState } from "react";
import {
  MapContainer,
  TileLayer,
  Polyline,
  Marker,
  Polygon,
  useMap,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { Box, Card, CardContent, Typography, Chip, Stack } from "@mui/material";
import { type SearchPatternPreview, type Waypoint } from "../../types/search";
import searchService from "../../services/search";

// Fix for default markers in react-leaflet
import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

delete (L.Icon.Default.prototype as unknown as { _getIconUrl?: () => string })
  ._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
});

interface PatternMapPreviewProps {
  patternId?: string;
  refresh?: boolean;
}

const MapBounds: React.FC<{ waypoints: Waypoint[] }> = ({ waypoints }) => {
  const map = useMap();

  useEffect(() => {
    if (waypoints.length > 0) {
      const bounds = L.latLngBounds(waypoints.map((wp) => [wp.lat, wp.lon]));
      map.fitBounds(bounds, { padding: [50, 50] });
    }
  }, [waypoints, map]);

  return null;
};

const PatternMapPreview: React.FC<PatternMapPreviewProps> = ({
  patternId,
  refresh,
}) => {
  const [preview, setPreview] = useState<SearchPatternPreview | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadPreview();
  }, [patternId, refresh]);

  const loadPreview = async () => {
    if (!patternId && !refresh) return;

    setLoading(true);
    setError(null);
    try {
      const data = await searchService.getPatternPreview();
      setPreview(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load preview");
    } finally {
      setLoading(false);
    }
  };

  const formatDistance = (meters: number): string => {
    if (meters >= 1000) {
      return `${(meters / 1000).toFixed(2)} km`;
    }
    return `${meters.toFixed(0)} m`;
  };

  const formatTime = (seconds: number): string => {
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    }
    return `${secs}s`;
  };

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Typography>Loading pattern preview...</Typography>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent>
          <Typography color="error">{error}</Typography>
        </CardContent>
      </Card>
    );
  }

  if (!preview || !preview.waypoints || preview.waypoints.length === 0) {
    return (
      <Card>
        <CardContent>
          <Typography>No pattern preview available</Typography>
        </CardContent>
      </Card>
    );
  }

  const pathCoordinates = preview.waypoints.map(
    (wp) => [wp.lat, wp.lon] as [number, number],
  );
  const centerLat =
    preview.waypoints.reduce((sum, wp) => sum + wp.lat, 0) /
    preview.waypoints.length;
  const centerLon =
    preview.waypoints.reduce((sum, wp) => sum + wp.lon, 0) /
    preview.waypoints.length;

  // Create numbered markers for waypoints
  const createNumberedIcon = (number: number) => {
    return L.divIcon({
      html: `<div style="
        background-color: #2196F3;
        color: white;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: bold;
        border: 2px solid white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
      ">${number}</div>`,
      className: "numbered-marker",
      iconSize: [24, 24],
      iconAnchor: [12, 12],
    });
  };

  return (
    <Card>
      <CardContent>
        <Stack spacing={2}>
          <Box
            sx={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <Typography variant="h6">Pattern Preview</Typography>
            <Stack direction="row" spacing={1}>
              <Chip
                label={`${preview.waypoints.length} waypoints`}
                size="small"
              />
              <Chip
                label={formatDistance(preview.total_distance)}
                size="small"
                color="primary"
              />
              <Chip
                label={formatTime(preview.estimated_time)}
                size="small"
                color="secondary"
              />
            </Stack>
          </Box>

          <Box sx={{ height: 500, width: "100%" }}>
            <MapContainer
              center={[centerLat, centerLon]}
              zoom={13}
              style={{ height: "100%", width: "100%" }}
            >
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              />

              <MapBounds waypoints={preview.waypoints} />

              {/* Draw pattern path */}
              <Polyline
                positions={pathCoordinates}
                color="#2196F3"
                weight={3}
                opacity={0.8}
              />

              {/* Draw boundary if available */}
              {preview.boundary && preview.boundary.coordinates && (
                <Polygon
                  positions={preview.boundary.coordinates[0].map(
                    (coord: number[]) => [coord[1], coord[0]],
                  )}
                  color="#FF5722"
                  weight={2}
                  opacity={0.5}
                  fillOpacity={0.1}
                />
              )}

              {/* Add numbered markers for waypoints */}
              {preview.waypoints.map((waypoint, index) => (
                <Marker
                  key={index}
                  position={[waypoint.lat, waypoint.lon]}
                  icon={createNumberedIcon(index + 1)}
                ></Marker>
              ))}

              {/* Highlight start and end points */}
              {preview.waypoints.length > 0 && (
                <>
                  <Marker
                    position={[
                      preview.waypoints[0].lat,
                      preview.waypoints[0].lon,
                    ]}
                    icon={L.divIcon({
                      html: '<div style="background-color: #4CAF50; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;">START</div>',
                      className: "start-marker",
                      iconSize: [50, 20],
                      iconAnchor: [25, 10],
                    })}
                  />
                  <Marker
                    position={[
                      preview.waypoints[preview.waypoints.length - 1].lat,
                      preview.waypoints[preview.waypoints.length - 1].lon,
                    ]}
                    icon={L.divIcon({
                      html: '<div style="background-color: #F44336; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;">END</div>',
                      className: "end-marker",
                      iconSize: [40, 20],
                      iconAnchor: [20, 10],
                    })}
                  />
                </>
              )}
            </MapContainer>
          </Box>

          <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
            <Typography variant="caption" color="text.secondary">
              Scale: 1:
              {Math.round(
                (40075016.686 *
                  Math.abs(Math.cos((centerLat * Math.PI) / 180))) /
                  Math.pow(2, 13 + 8),
              )}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              • Blue line: Flight path
            </Typography>
            <Typography variant="caption" color="text.secondary">
              • Red boundary: Search area
            </Typography>
            <Typography variant="caption" color="text.secondary">
              • Numbered markers: Waypoint sequence
            </Typography>
          </Box>
        </Stack>
      </CardContent>
    </Card>
  );
};

export default PatternMapPreview;
