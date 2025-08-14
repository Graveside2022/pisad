import { useEffect, useState } from "react";
import {
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Grid,
} from "@mui/material";
import type { SignalDetection } from "../../types";

function DetectionLog() {
  const [detections, setDetections] = useState<SignalDetection[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch detection log from API
    const fetchDetections = async () => {
      try {
        const response = await fetch("/api/detections?limit=10");
        if (response.ok) {
          const data = await response.json();
          if (data.detections) {
            setDetections(data.detections);
          }
        }
      } catch (error) {
        console.error("Failed to fetch detections:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchDetections();
    const interval = setInterval(fetchDetections, 5000); // Refresh every 5 seconds

    return () => clearInterval(interval);
  }, []);

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
  };

  const formatFrequency = (freq: number) => {
    return `${(freq / 1000000).toFixed(3)} MHz`;
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return "success";
    if (confidence >= 50) return "warning";
    return "error";
  };

  if (loading) {
    return (
      <Paper elevation={2} sx={{ p: 2, height: "100%" }}>
        <Typography variant="h6" gutterBottom>
          Detection Log
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Loading...
        </Typography>
      </Paper>
    );
  }

  return (
    <Paper elevation={2} sx={{ p: 2, height: "100%", overflow: "auto" }}>
      <Typography variant="h6" gutterBottom>
        Detection Log
      </Typography>

      {detections.length === 0 ? (
        <Grid
          container
          justifyContent="center"
          alignItems="center"
          sx={{ py: 4 }}
        >
          <Grid>
            <Typography variant="body2" color="text.secondary">
              No detections recorded
            </Typography>
          </Grid>
        </Grid>
      ) : (
        <TableContainer>
          <Table size="small" stickyHeader>
            <TableHead>
              <TableRow>
                <TableCell>Time</TableCell>
                <TableCell>Frequency</TableCell>
                <TableCell align="right">RSSI</TableCell>
                <TableCell align="right">SNR</TableCell>
                <TableCell align="center">Confidence</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {detections.map((detection) => (
                <TableRow key={detection.id}>
                  <TableCell>{formatTimestamp(detection.timestamp)}</TableCell>
                  <TableCell>{formatFrequency(detection.frequency)}</TableCell>
                  <TableCell align="right">
                    {detection.rssi.toFixed(1)} dBm
                  </TableCell>
                  <TableCell align="right">
                    {detection.snr.toFixed(1)} dB
                  </TableCell>
                  <TableCell align="center">
                    <Chip
                      label={`${detection.confidence.toFixed(0)}%`}
                      color={getConfidenceColor(detection.confidence)}
                      size="small"
                      variant="outlined"
                    />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Paper>
  );
}

export default DetectionLog;
