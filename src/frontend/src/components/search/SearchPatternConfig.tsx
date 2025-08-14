import React, { useState } from "react";
import {
  Box,
  Card,
  CardContent,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Slider,
  Typography,
  TextField,
  Button,
  ToggleButton,
  ToggleButtonGroup,
  Stack,
  Alert,
  CircularProgress,
  Grid,
} from "@mui/material";
import {
  type PatternType,
  type BoundaryType,
  type SearchPatternRequest,
  type CenterRadiusBoundary,
  type CornerBoundary,
  type Coordinate,
} from "../../types/search";
import searchService from "../../services/search";

interface SearchPatternConfigProps {
  onPatternCreated?: (patternId: string) => void;
  onPreviewRequested?: (config: SearchPatternRequest) => void;
}

const SearchPatternConfig: React.FC<SearchPatternConfigProps> = ({
  onPatternCreated,
  onPreviewRequested,
}) => {
  const [patternType, setPatternType] =
    useState<PatternType>("expanding_square");
  const [spacing, setSpacing] = useState<number>(75);
  const [velocity, setVelocity] = useState<number>(7);
  const [boundaryType, setBoundaryType] =
    useState<BoundaryType>("center_radius");
  const [centerLat, setCenterLat] = useState<string>("");
  const [centerLon, setCenterLon] = useState<string>("");
  const [radius, setRadius] = useState<string>("500");
  const [corners, setCorners] = useState<Coordinate[]>([
    { lat: 0, lon: 0 },
    { lat: 0, lon: 0 },
    { lat: 0, lon: 0 },
    { lat: 0, lon: 0 },
  ]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validateInputs = (): boolean => {
    if (boundaryType === "center_radius") {
      if (!centerLat || !centerLon || !radius) {
        setError("Please fill in all center and radius fields");
        return false;
      }
      const lat = parseFloat(centerLat);
      const lon = parseFloat(centerLon);
      const r = parseFloat(radius);
      if (isNaN(lat) || isNaN(lon) || isNaN(r)) {
        setError("Invalid coordinate or radius values");
        return false;
      }
      if (lat < -90 || lat > 90) {
        setError("Latitude must be between -90 and 90");
        return false;
      }
      if (lon < -180 || lon > 180) {
        setError("Longitude must be between -180 and 180");
        return false;
      }
      if (r <= 0 || r > 10000) {
        setError("Radius must be between 0 and 10000 meters");
        return false;
      }
    } else {
      for (const corner of corners) {
        if (
          corner.lat < -90 ||
          corner.lat > 90 ||
          corner.lon < -180 ||
          corner.lon > 180
        ) {
          setError("Invalid corner coordinates");
          return false;
        }
      }
    }
    return true;
  };

  const buildRequest = (): SearchPatternRequest => {
    const bounds =
      boundaryType === "center_radius"
        ? ({
            type: "center_radius" as const,
            center: { lat: parseFloat(centerLat), lon: parseFloat(centerLon) },
            radius: parseFloat(radius),
          } as CenterRadiusBoundary)
        : ({
            type: "corners" as const,
            corners: corners,
          } as CornerBoundary);

    return {
      pattern: patternType,
      spacing,
      velocity,
      bounds,
    };
  };

  const handlePreview = () => {
    setError(null);
    if (!validateInputs()) return;

    const request = buildRequest();
    onPreviewRequested?.(request);
  };

  const handleCreate = async () => {
    setError(null);
    if (!validateInputs()) return;

    setLoading(true);
    try {
      const request = buildRequest();
      const response = await searchService.createPattern(request);
      onPatternCreated?.(response.pattern_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create pattern");
    } finally {
      setLoading(false);
    }
  };

  const updateCorner = (index: number, field: "lat" | "lon", value: string) => {
    const newCorners = [...corners];
    newCorners[index] = {
      ...newCorners[index],
      [field]: parseFloat(value) || 0,
    };
    setCorners(newCorners);
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Search Pattern Configuration
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        <Stack spacing={3}>
          <FormControl fullWidth>
            <InputLabel>Pattern Type</InputLabel>
            <Select
              value={patternType}
              onChange={(e) => setPatternType(e.target.value as PatternType)}
              label="Pattern Type"
            >
              <MenuItem value="expanding_square">Expanding Square</MenuItem>
              <MenuItem value="spiral">Spiral</MenuItem>
              <MenuItem value="lawnmower">Lawnmower</MenuItem>
            </Select>
          </FormControl>

          <Box>
            <Typography gutterBottom>Spacing: {spacing}m</Typography>
            <Slider
              value={spacing}
              onChange={(_, value) => setSpacing(value as number)}
              min={50}
              max={100}
              step={5}
              marks
              valueLabelDisplay="auto"
            />
          </Box>

          <Box>
            <Typography gutterBottom>Velocity: {velocity} m/s</Typography>
            <Slider
              value={velocity}
              onChange={(_, value) => setVelocity(value as number)}
              min={5}
              max={10}
              step={0.5}
              marks
              valueLabelDisplay="auto"
            />
          </Box>

          <Box>
            <Typography gutterBottom>Boundary Type</Typography>
            <ToggleButtonGroup
              value={boundaryType}
              exclusive
              onChange={(_, value) => value && setBoundaryType(value)}
              fullWidth
            >
              <ToggleButton value="center_radius">Center + Radius</ToggleButton>
              <ToggleButton value="corners">Corner Coordinates</ToggleButton>
            </ToggleButtonGroup>
          </Box>

          {boundaryType === "center_radius" ? (
            <Grid container spacing={2}>
              <Grid size={6}>
                <TextField
                  label="Center Latitude"
                  value={centerLat}
                  onChange={(e) => setCenterLat(e.target.value)}
                  type="number"
                  fullWidth
                  inputProps={{ step: 0.000001, min: -90, max: 90 }}
                />
              </Grid>
              <Grid size={6}>
                <TextField
                  label="Center Longitude"
                  value={centerLon}
                  onChange={(e) => setCenterLon(e.target.value)}
                  type="number"
                  fullWidth
                  inputProps={{ step: 0.000001, min: -180, max: 180 }}
                />
              </Grid>
              <Grid size={12}>
                <TextField
                  label="Radius (meters)"
                  value={radius}
                  onChange={(e) => setRadius(e.target.value)}
                  type="number"
                  fullWidth
                  inputProps={{ step: 10, min: 10, max: 10000 }}
                />
              </Grid>
            </Grid>
          ) : (
            <Stack spacing={2}>
              {corners.map((corner, index) => (
                <Grid container spacing={2} key={index}>
                  <Grid size={12}>
                    <Typography variant="caption">
                      Corner {index + 1}
                    </Typography>
                  </Grid>
                  <Grid size={6}>
                    <TextField
                      label="Latitude"
                      value={corner.lat}
                      onChange={(e) =>
                        updateCorner(index, "lat", e.target.value)
                      }
                      type="number"
                      fullWidth
                      size="small"
                      inputProps={{ step: 0.000001, min: -90, max: 90 }}
                    />
                  </Grid>
                  <Grid size={6}>
                    <TextField
                      label="Longitude"
                      value={corner.lon}
                      onChange={(e) =>
                        updateCorner(index, "lon", e.target.value)
                      }
                      type="number"
                      fullWidth
                      size="small"
                      inputProps={{ step: 0.000001, min: -180, max: 180 }}
                    />
                  </Grid>
                </Grid>
              ))}
            </Stack>
          )}

          <Grid container spacing={2}>
            <Grid size={6}>
              <Button
                variant="outlined"
                fullWidth
                onClick={handlePreview}
                disabled={loading}
              >
                Preview Pattern
              </Button>
            </Grid>
            <Grid size={6}>
              <Button
                variant="contained"
                fullWidth
                onClick={handleCreate}
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} /> : "Create Pattern"}
              </Button>
            </Grid>
          </Grid>
        </Stack>
      </CardContent>
    </Card>
  );
};

export default SearchPatternConfig;
