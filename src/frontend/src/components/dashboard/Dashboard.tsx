import { useState, useEffect } from "react";
import { Typography, Grid } from "@mui/material";
import { useWebSocket } from "../../hooks/useWebSocket";
import SignalMeter from "./SignalMeter";
import RSSIGraph from "./RSSIGraph";
import SDRStatus from "./SDRStatus";
import SystemHealth from "./SystemHealth";
import DetectionLog from "./DetectionLog";
import HomingMonitor from "../homing/HomingMonitor";
import GradientVisualization from "../homing/GradientVisualization";

interface RSSIData {
  rssi: number;
  noiseFloor: number;
  snr: number;
  confidence: number;
  timestamp: string;
}

function Dashboard() {
  const { isConnected, addMessageHandler } = useWebSocket();
  const [rssiData, setRssiData] = useState<RSSIData>({
    rssi: -80,
    noiseFloor: -95,
    snr: 15,
    confidence: 0,
    timestamp: new Date().toISOString(),
  });

  useEffect(() => {
    // Handle WebSocket messages
    const cleanup = addMessageHandler((message) => {
      if (message.type === "rssi" && message.data) {
        setRssiData(message.data as RSSIData);
      }
    });

    return cleanup;
  }, [addMessageHandler]);

  return (
    <Grid container spacing={3}>
      <Grid size={12}>
        <Typography variant="h4" gutterBottom>
          Signal Monitoring Dashboard
        </Typography>
      </Grid>
      <Grid size={{ xs: 12, md: 6, lg: 4 }}>
        <SignalMeter
          rssi={rssiData.rssi}
          noiseFloor={rssiData.noiseFloor}
          snr={rssiData.snr}
          size="medium"
        />
      </Grid>

      <Grid size={{ xs: 12, md: 6, lg: 8 }}>
        <RSSIGraph
          currentRSSI={rssiData.rssi}
          currentNoiseFloor={rssiData.noiseFloor}
          currentSNR={rssiData.snr}
        />
      </Grid>

      <Grid size={{ xs: 12, md: 4 }}>
        <SDRStatus status={isConnected ? "CONNECTED" : "DISCONNECTED"} />
      </Grid>

      <Grid size={{ xs: 12, md: 4 }}>
        <SystemHealth />
      </Grid>

      <Grid size={{ xs: 12, md: 4 }}>
        <DetectionLog />
      </Grid>

      <Grid size={{ xs: 12, md: 6 }}>
        <HomingMonitor />
      </Grid>

      <Grid size={{ xs: 12, md: 6 }}>
        <GradientVisualization />
      </Grid>
    </Grid>
  );
}

export default Dashboard;
