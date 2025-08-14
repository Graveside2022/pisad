import React, { useState, useEffect } from "react";
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Button,
  LinearProgress,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  TextField,
} from "@mui/material";
import {
  PlayArrow,
  Stop,
  GetApp,
  CheckCircle,
  Error,
  Warning,
} from "@mui/icons-material";
import { useFieldTest } from "../../hooks/useFieldTest";
import { PreflightChecklist } from "./PreflightChecklist";
import { MetricsChart } from "./MetricsChart";

interface EnvironmentalConditions {
  wind_speed_mps: number;
  temperature_c: number;
  humidity_percent: number;
}

export const FieldTestDashboard: React.FC = () => {
  const {
    activeTest,
    testMetrics,
    preflightStatus,
    startTest,
    stopTest,
    exportTestData,
    getPreflightStatus,
  } = useFieldTest();

  const [testType, setTestType] = useState<string>("detection_range");
  const [testName, setTestName] = useState<string>("");
  const [beaconPower, setBeaconPower] = useState<number>(10);
  const [envConditions, setEnvConditions] = useState<EnvironmentalConditions>({
    wind_speed_mps: 0,
    temperature_c: 20,
    humidity_percent: 50,
  });
  const [isTestRunning, setIsTestRunning] = useState(false);

  useEffect(() => {
    // Get initial preflight status
    getPreflightStatus();

    // Poll for updates if test is running
    if (isTestRunning && activeTest) {
      const interval = setInterval(() => {
        // Update test status
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [isTestRunning, activeTest, getPreflightStatus]);

  const handleStartTest = async () => {
    if (!testName) {
      alert("Please enter a test name");
      return;
    }

    const config = {
      test_name: testName,
      test_type: testType,
      beacon_config: {
        frequency_hz: 433000000,
        power_dbm: beaconPower,
        modulation: "LoRa",
        spreading_factor: 7,
        bandwidth_hz: 125000,
        coding_rate: 5,
        pulse_rate_hz: 1.0,
        pulse_duration_ms: 100,
      },
      environmental_conditions: envConditions,
    };

    const success = await startTest(config);
    if (success) {
      setIsTestRunning(true);
    }
  };

  const handleStopTest = async () => {
    await stopTest();
    setIsTestRunning(false);
  };

  const handleExportData = async (format: "csv" | "json") => {
    if (activeTest) {
      await exportTestData(activeTest.test_id, format);
    }
  };

  const getPhaseColor = (phase: string) => {
    switch (phase) {
      case "setup":
        return "info";
      case "detection":
        return "warning";
      case "approach":
        return "primary";
      case "analysis":
        return "secondary";
      case "completed":
        return "success";
      case "failed":
        return "error";
      default:
        return "default";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "running":
        return <Warning color="warning" />;
      case "completed":
        return <CheckCircle color="success" />;
      case "failed":
        return <Error color="error" />;
      default:
        return null;
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Field Test Dashboard
      </Typography>

      <Grid container spacing={3}>
        {/* Preflight Checklist */}
        <Grid size={{ xs: 12, md: 6 }}>
          <PreflightChecklist
            checklistStatus={preflightStatus}
            onRefresh={getPreflightStatus}
          />
        </Grid>

        {/* Test Configuration */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Test Configuration
              </Typography>

              <Grid container spacing={2}>
                <Grid size={12}>
                  <TextField
                    fullWidth
                    label="Test Name"
                    value={testName}
                    onChange={(e) => setTestName(e.target.value)}
                    disabled={isTestRunning}
                  />
                </Grid>

                <Grid size={12}>
                  <FormControl fullWidth>
                    <InputLabel>Test Type</InputLabel>
                    <Select
                      value={testType}
                      onChange={(e) => setTestType(e.target.value)}
                      disabled={isTestRunning}
                    >
                      <MenuItem value="detection_range">
                        Detection Range
                      </MenuItem>
                      <MenuItem value="approach_accuracy">
                        Approach Accuracy
                      </MenuItem>
                      <MenuItem value="state_transition">
                        State Transition
                      </MenuItem>
                      <MenuItem value="safety_validation">
                        Safety Validation
                      </MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid size={6}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Beacon Power (dBm)"
                    value={beaconPower}
                    onChange={(e) => setBeaconPower(Number(e.target.value))}
                    disabled={isTestRunning}
                  />
                </Grid>

                <Grid size={6}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Wind Speed (m/s)"
                    value={envConditions.wind_speed_mps}
                    onChange={(e) =>
                      setEnvConditions({
                        ...envConditions,
                        wind_speed_mps: Number(e.target.value),
                      })
                    }
                    disabled={isTestRunning}
                  />
                </Grid>

                <Grid size={6}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Temperature (°C)"
                    value={envConditions.temperature_c}
                    onChange={(e) =>
                      setEnvConditions({
                        ...envConditions,
                        temperature_c: Number(e.target.value),
                      })
                    }
                    disabled={isTestRunning}
                  />
                </Grid>

                <Grid size={6}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Humidity (%)"
                    value={envConditions.humidity_percent}
                    onChange={(e) =>
                      setEnvConditions({
                        ...envConditions,
                        humidity_percent: Number(e.target.value),
                      })
                    }
                    disabled={isTestRunning}
                  />
                </Grid>

                <Grid size={12}>
                  <Box sx={{ display: "flex", gap: 2 }}>
                    {!isTestRunning ? (
                      <Button
                        variant="contained"
                        color="primary"
                        startIcon={<PlayArrow />}
                        onClick={handleStartTest}
                        disabled={!preflightStatus?.all_passed}
                        fullWidth
                      >
                        Start Test
                      </Button>
                    ) : (
                      <Button
                        variant="contained"
                        color="error"
                        startIcon={<Stop />}
                        onClick={handleStopTest}
                        fullWidth
                      >
                        Stop Test
                      </Button>
                    )}
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Test Status */}
        {activeTest && (
          <Grid size={12}>
            <Card>
              <CardContent>
                <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
                  <Typography variant="h6" sx={{ flexGrow: 1 }}>
                    Test Status
                  </Typography>
                  {getStatusIcon(activeTest.status)}
                </Box>

                <Grid container spacing={2}>
                  <Grid size={{ xs: 12, md: 3 }}>
                    <Typography variant="body2" color="textSecondary">
                      Test ID
                    </Typography>
                    <Typography
                      variant="body1"
                      sx={{ fontFamily: "monospace" }}
                    >
                      {activeTest.test_id.substring(0, 8)}...
                    </Typography>
                  </Grid>

                  <Grid size={{ xs: 12, md: 3 }}>
                    <Typography variant="body2" color="textSecondary">
                      Phase
                    </Typography>
                    <Chip
                      label={activeTest.phase}
                      color={getPhaseColor(activeTest.phase) as any}
                      size="small"
                    />
                  </Grid>

                  <Grid size={{ xs: 12, md: 3 }}>
                    <Typography variant="body2" color="textSecondary">
                      Progress
                    </Typography>
                    <Box sx={{ display: "flex", alignItems: "center" }}>
                      <Typography variant="body1">
                        {activeTest.current_iteration} /{" "}
                        {activeTest.total_iterations}
                      </Typography>
                    </Box>
                  </Grid>

                  <Grid size={{ xs: 12, md: 3 }}>
                    <Typography variant="body2" color="textSecondary">
                      Beacon Detection
                    </Typography>
                    <Chip
                      label={
                        activeTest.beacon_detected ? "Detected" : "Not Detected"
                      }
                      color={activeTest.beacon_detected ? "success" : "default"}
                      size="small"
                    />
                  </Grid>

                  <Grid size={12}>
                    <LinearProgress
                      variant="determinate"
                      value={
                        (activeTest.current_iteration /
                          activeTest.total_iterations) *
                        100
                      }
                    />
                  </Grid>

                  <Grid size={{ xs: 12, md: 6 }}>
                    <Typography variant="body2" color="textSecondary">
                      Current Distance
                    </Typography>
                    <Typography variant="h5">
                      {activeTest.current_distance_m.toFixed(1)} m
                    </Typography>
                  </Grid>

                  <Grid size={{ xs: 12, md: 6 }}>
                    <Typography variant="body2" color="textSecondary">
                      Current RSSI
                    </Typography>
                    <Typography variant="h5">
                      {activeTest.current_rssi_dbm.toFixed(1)} dBm
                    </Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Test Metrics */}
        {testMetrics && (
          <>
            <Grid size={12}>
              <Card>
                <CardContent>
                  <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
                    <Typography variant="h6" sx={{ flexGrow: 1 }}>
                      Test Metrics
                    </Typography>
                    <Button
                      size="small"
                      startIcon={<GetApp />}
                      onClick={() => handleExportData("csv")}
                    >
                      Export CSV
                    </Button>
                    <Button
                      size="small"
                      startIcon={<GetApp />}
                      onClick={() => handleExportData("json")}
                      sx={{ ml: 1 }}
                    >
                      Export JSON
                    </Button>
                  </Box>

                  <TableContainer component={Paper} variant="outlined">
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Metric</TableCell>
                          <TableCell align="right">Value</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        <TableRow>
                          <TableCell>Detection Range</TableCell>
                          <TableCell align="right">
                            {testMetrics.detection_range_m.toFixed(1)} m
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Approach Accuracy</TableCell>
                          <TableCell align="right">
                            {testMetrics.approach_accuracy_m.toFixed(1)} m
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Time to Locate</TableCell>
                          <TableCell align="right">
                            {testMetrics.time_to_locate_s.toFixed(1)} s
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Transition Latency</TableCell>
                          <TableCell align="right">
                            {testMetrics.transition_latency_ms.toFixed(0)} ms
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Max RSSI</TableCell>
                          <TableCell align="right">
                            {testMetrics.max_rssi_dbm.toFixed(1)} dBm
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Avg RSSI</TableCell>
                          <TableCell align="right">
                            {testMetrics.avg_rssi_dbm.toFixed(1)} dBm
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Signal Loss Count</TableCell>
                          <TableCell align="right">
                            {testMetrics.signal_loss_count}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Test Result</TableCell>
                          <TableCell align="right">
                            <Chip
                              label={testMetrics.success ? "PASS" : "FAIL"}
                              color={testMetrics.success ? "success" : "error"}
                              size="small"
                            />
                          </TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>

                  {testMetrics.safety_events.length > 0 && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Safety Events
                      </Typography>
                      <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
                        {testMetrics.safety_events.map((event, index) => (
                          <Chip
                            key={index}
                            label={event}
                            size="small"
                            color="warning"
                          />
                        ))}
                      </Box>
                    </Box>
                  )}
                </CardContent>
              </Card>
            </Grid>

            {/* Metrics Chart */}
            <Grid size={12}>
              <MetricsChart metrics={testMetrics} />
            </Grid>
          </>
        )}
      </Grid>
    </Box>
  );
};
