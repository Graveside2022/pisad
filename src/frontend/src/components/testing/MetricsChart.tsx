import React from "react";
import { Card, CardContent, Typography, Box } from "@mui/material";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";

interface MetricsChartProps {
  metrics: {
    beacon_power_dbm: number;
    detection_range_m: number;
    approach_accuracy_m: number;
    time_to_locate_s: number;
    transition_latency_ms: number;
    max_rssi_dbm: number;
    min_rssi_dbm: number;
    avg_rssi_dbm: number;
    signal_loss_count: number;
  };
}

export const MetricsChart: React.FC<MetricsChartProps> = ({ metrics }) => {
  // Performance metrics for radar chart
  const performanceData = [
    {
      metric: "Detection Range",
      value: Math.min((metrics.detection_range_m / 750) * 100, 100),
      fullMark: 100,
    },
    {
      metric: "Accuracy",
      value: Math.max(100 - (metrics.approach_accuracy_m / 100) * 100, 0),
      fullMark: 100,
    },
    {
      metric: "Speed",
      value: Math.max(100 - (metrics.time_to_locate_s / 60) * 100, 0),
      fullMark: 100,
    },
    {
      metric: "Responsiveness",
      value: Math.max(100 - (metrics.transition_latency_ms / 5000) * 100, 0),
      fullMark: 100,
    },
    {
      metric: "Signal Quality",
      value: Math.max(((metrics.avg_rssi_dbm + 120) / 100) * 100, 0),
      fullMark: 100,
    },
    {
      metric: "Reliability",
      value: Math.max(100 - metrics.signal_loss_count * 10, 0),
      fullMark: 100,
    },
  ];

  // RSSI comparison data
  const rssiData = [
    { name: "Min RSSI", value: metrics.min_rssi_dbm },
    { name: "Avg RSSI", value: metrics.avg_rssi_dbm },
    { name: "Max RSSI", value: metrics.max_rssi_dbm },
  ];

  // Key metrics bar chart
  const keyMetricsData = [
    { name: "Range (100m)", value: metrics.detection_range_m / 100 },
    { name: "Accuracy (10m)", value: metrics.approach_accuracy_m / 10 },
    { name: "Time (10s)", value: metrics.time_to_locate_s / 10 },
    { name: "Latency (100ms)", value: metrics.transition_latency_ms / 100 },
  ];

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Performance Analysis
        </Typography>

        <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 3 }}>
          {/* Performance Radar Chart */}
          <Box>
            <Typography variant="subtitle2" gutterBottom align="center">
              Overall Performance
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={performanceData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="metric" />
                <PolarRadiusAxis angle={90} domain={[0, 100]} />
                <Radar
                  name="Performance"
                  dataKey="value"
                  stroke="#8884d8"
                  fill="#8884d8"
                  fillOpacity={0.6}
                />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </Box>

          {/* Key Metrics Bar Chart */}
          <Box>
            <Typography variant="subtitle2" gutterBottom align="center">
              Key Metrics (Normalized)
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={keyMetricsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#82ca9d" />
              </BarChart>
            </ResponsiveContainer>
          </Box>

          {/* RSSI Levels */}
          <Box>
            <Typography variant="subtitle2" gutterBottom align="center">
              RSSI Levels (dBm)
            </Typography>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={rssiData} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[-120, 0]} />
                <YAxis type="category" dataKey="name" />
                <Tooltip />
                <Bar dataKey="value" fill="#ffc658" />
              </BarChart>
            </ResponsiveContainer>
          </Box>

          {/* Summary Statistics */}
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Summary Statistics
            </Typography>
            <Box sx={{ p: 2, bgcolor: "background.default", borderRadius: 1 }}>
              <Typography variant="body2">
                <strong>Beacon Power:</strong> {metrics.beacon_power_dbm} dBm
              </Typography>
              <Typography variant="body2">
                <strong>Detection Success Rate:</strong>{" "}
                {metrics.signal_loss_count === 0
                  ? "100%"
                  : `${100 - metrics.signal_loss_count * 5}%`}
              </Typography>
              <Typography variant="body2">
                <strong>Performance Score:</strong>{" "}
                {(
                  performanceData.reduce((sum, item) => sum + item.value, 0) /
                  performanceData.length
                ).toFixed(1)}
                %
              </Typography>
              <Typography variant="body2">
                <strong>Test Conditions:</strong> Field environment
              </Typography>
            </Box>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};
