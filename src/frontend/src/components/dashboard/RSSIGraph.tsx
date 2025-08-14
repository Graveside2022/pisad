import { useEffect, useRef, useState } from "react";
import { Typography, Paper, Grid } from "@mui/material";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

interface RSSIDataPoint {
  timestamp: string;
  rssi: number;
  noiseFloor: number;
  snr: number;
}

interface RSSIGraphProps {
  currentRSSI?: number;
  currentNoiseFloor?: number;
  currentSNR?: number;
}

function RSSIGraph({
  currentRSSI,
  currentNoiseFloor,
  currentSNR,
}: RSSIGraphProps) {
  const [data, setData] = useState<RSSIDataPoint[]>([]);
  const dataRef = useRef<RSSIDataPoint[]>([]);
  const MAX_DATA_POINTS = 60; // 60 seconds of data at 1Hz

  useEffect(() => {
    if (
      currentRSSI !== undefined &&
      currentNoiseFloor !== undefined &&
      currentSNR !== undefined
    ) {
      const newPoint: RSSIDataPoint = {
        timestamp: new Date().toLocaleTimeString(),
        rssi: currentRSSI,
        noiseFloor: currentNoiseFloor,
        snr: currentSNR,
      };

      dataRef.current = [...dataRef.current, newPoint];

      // Keep only last 60 seconds of data
      if (dataRef.current.length > MAX_DATA_POINTS) {
        dataRef.current = dataRef.current.slice(-MAX_DATA_POINTS);
      }

      setData([...dataRef.current]);
    }
  }, [currentRSSI, currentNoiseFloor, currentSNR]);

  const formatYAxis = (value: number) => `${value} dBm`;

  return (
    <Paper elevation={2} sx={{ p: 2, height: "100%" }}>
      <Typography variant="h6" gutterBottom>
        RSSI History (60 seconds)
      </Typography>

      <Grid sx={{ width: "100%", height: "calc(100% - 40px)" }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#444" />
            <XAxis
              dataKey="timestamp"
              stroke="#888"
              tick={{ fontSize: 10 }}
              interval="preserveStartEnd"
            />
            <YAxis
              stroke="#888"
              tickFormatter={formatYAxis}
              domain={[-100, -30]}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#1a1f3a",
                border: "1px solid #444",
              }}
              formatter={(value: number) => `${value.toFixed(1)} dBm`}
            />
            <Legend />

            <Line
              type="monotone"
              dataKey="rssi"
              stroke="#00b4d8"
              strokeWidth={2}
              dot={false}
              name="RSSI"
              isAnimationActive={false}
            />

            <Line
              type="monotone"
              dataKey="noiseFloor"
              stroke="#ff6b6b"
              strokeWidth={1}
              strokeDasharray="5 5"
              dot={false}
              name="Noise Floor"
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </Grid>
    </Paper>
  );
}

export default RSSIGraph;
