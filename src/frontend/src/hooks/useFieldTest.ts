import { useState, useCallback, useEffect } from "react";
import { testingService } from "../services/testingService";

interface FieldTestConfig {
  test_name: string;
  test_type: string;
  beacon_config: {
    frequency_hz: number;
    power_dbm: number;
    modulation: string;
    spreading_factor: number;
    bandwidth_hz: number;
    coding_rate: number;
    pulse_rate_hz: number;
    pulse_duration_ms: number;
  };
  environmental_conditions: {
    wind_speed_mps: number;
    temperature_c: number;
    humidity_percent: number;
  };
}

interface TestStatus {
  test_id: string;
  phase: string;
  status: string;
  current_iteration: number;
  total_iterations: number;
  current_distance_m: number;
  current_rssi_dbm: number;
  beacon_detected: boolean;
}

interface TestMetrics {
  beacon_power_dbm: number;
  detection_range_m: number;
  approach_accuracy_m: number;
  time_to_locate_s: number;
  transition_latency_ms: number;
  max_rssi_dbm: number;
  min_rssi_dbm: number;
  avg_rssi_dbm: number;
  signal_loss_count: number;
  safety_events: string[];
  success: boolean;
}

interface PreflightStatus {
  mavlink_connected: boolean;
  gps_fix_valid: boolean;
  battery_sufficient: boolean;
  safety_interlocks_passed: boolean;
  geofence_configured: boolean;
  emergency_stop_ready: boolean;
  signal_processor_active: boolean;
  state_machine_ready: boolean;
  all_passed: boolean;
}

export const useFieldTest = () => {
  const [activeTest, setActiveTest] = useState<TestStatus | null>(null);
  const [testMetrics, setTestMetrics] = useState<TestMetrics | null>(null);
  const [preflightStatus, setPreflightStatus] =
    useState<PreflightStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Get preflight checklist status
  const getPreflightStatus = useCallback(async () => {
    try {
      setIsLoading(true);
      const status = await testingService.getPreflightStatus();
      setPreflightStatus(status);
      setError(null);
    } catch (err) {
      setError("Failed to get preflight status");
      console.error("Preflight status error:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Start a new field test
  const startTest = useCallback(
    async (config: FieldTestConfig): Promise<boolean> => {
      try {
        setIsLoading(true);
        setError(null);

        const response = await testingService.startFieldTest(config);

        if (response.test_id) {
          // Start polling for test status
          const testId = response.test_id;
          pollTestStatus(testId);
          return true;
        }
        return false;
      } catch (err: any) {
        setError(err.message || "Failed to start test");
        console.error("Start test error:", err);
        return false;
      } finally {
        setIsLoading(false);
      }
    },
    [],
  );

  // Stop the active test
  const stopTest = useCallback(async () => {
    if (!activeTest) return;

    try {
      // In a real implementation, add a stop endpoint
      setActiveTest(null);
      setTestMetrics(null);
      setError(null);
    } catch (err) {
      setError("Failed to stop test");
      console.error("Stop test error:", err);
    }
  }, [activeTest]);

  // Poll for test status updates
  const pollTestStatus = useCallback(async (testId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const status = await testingService.getFieldTestStatus(testId);
        setActiveTest(status);

        // Check if test is complete
        if (status.status === "completed" || status.status === "failed") {
          clearInterval(pollInterval);

          // Get final metrics
          const metrics = await testingService.getFieldTestMetrics(testId);
          setTestMetrics(metrics.metrics);
        }
      } catch (err) {
        console.error("Poll status error:", err);
        clearInterval(pollInterval);
      }
    }, 1000); // Poll every second

    // Store interval ID for cleanup
    return () => clearInterval(pollInterval);
  }, []);

  // Export test data
  const exportTestData = useCallback(
    async (testId: string, format: "csv" | "json") => {
      try {
        const result = await testingService.exportFieldTestData(testId, format);

        // In a real implementation, trigger download
        console.log("Export path:", result.export_path);

        // Create download link
        const link = document.createElement("a");
        link.href = `/api/testing/download/${result.export_path}`;
        link.download = `test_${testId}.${format}`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      } catch (err) {
        setError("Failed to export test data");
        console.error("Export error:", err);
      }
    },
    [],
  );

  // Get test history
  const getTestHistory = useCallback(async (limit: number = 10) => {
    try {
      const history = await testingService.getTestResults(limit, "field");
      return history.test_runs;
    } catch (err) {
      console.error("Get history error:", err);
      return [];
    }
  }, []);

  // WebSocket connection for real-time updates
  useEffect(() => {
    const ws = testingService.connectWebSocket();

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === "field_test") {
        // Update test status from WebSocket
        if (activeTest && data.data.test_id === activeTest.test_id) {
          setActiveTest({
            ...activeTest,
            phase: data.data.phase,
            current_distance_m: data.data.current_distance_m,
            current_rssi_dbm: data.data.current_rssi_dbm,
            beacon_detected: data.data.beacon_detected,
          });

          // Update partial metrics if available
          if (data.data.metrics) {
            setTestMetrics(
              (prev) =>
                ({
                  ...prev,
                  ...data.data.metrics,
                }) as TestMetrics,
            );
          }
        }
      }
    };

    return () => {
      ws.close();
    };
  }, [activeTest]);

  return {
    activeTest,
    testMetrics,
    preflightStatus,
    isLoading,
    error,
    startTest,
    stopTest,
    exportTestData,
    getPreflightStatus,
    getTestHistory,
  };
};
