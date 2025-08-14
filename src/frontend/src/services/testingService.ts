import { API_BASE_URL, WS_BASE_URL } from "../config";

interface FieldTestConfig {
  test_name: string;
  test_type: string;
  beacon_config: any;
  environmental_conditions: any;
}

interface FieldTestStartResponse {
  test_id: string;
  status: string;
  start_time: string;
  checklist_status: string;
}

interface TestRunsResponse {
  test_runs: Array<{
    id: string;
    timestamp: string;
    test_type: string;
    passed: number;
    failed: number;
    duration_ms: number;
    configuration: any;
  }>;
}

class TestingService {
  private wsConnection: WebSocket | null = null;

  async getPreflightStatus() {
    const response = await fetch(
      `${API_BASE_URL}/testing/field-test/preflight`,
    );
    if (!response.ok) {
      throw new Error("Failed to get preflight status");
    }
    return response.json();
  }

  async startFieldTest(
    config: FieldTestConfig,
  ): Promise<FieldTestStartResponse> {
    const response = await fetch(`${API_BASE_URL}/testing/field-test/start`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(config),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to start field test");
    }

    return response.json();
  }

  async getFieldTestStatus(testId: string) {
    const response = await fetch(
      `${API_BASE_URL}/testing/field-test/${testId}/status`,
    );
    if (!response.ok) {
      throw new Error("Failed to get test status");
    }
    return response.json();
  }

  async getFieldTestMetrics(testId: string) {
    const response = await fetch(
      `${API_BASE_URL}/testing/field-test/${testId}/metrics`,
    );
    if (!response.ok) {
      throw new Error("Failed to get test metrics");
    }
    return response.json();
  }

  async exportFieldTestData(testId: string, format: "csv" | "json") {
    const response = await fetch(
      `${API_BASE_URL}/testing/field-test/${testId}/export?format=${format}`,
      {
        method: "POST",
      },
    );

    if (!response.ok) {
      throw new Error("Failed to export test data");
    }

    return response.json();
  }

  async getTestResults(
    limit: number = 10,
    testType?: string,
  ): Promise<TestRunsResponse> {
    const params = new URLSearchParams();
    params.append("limit", limit.toString());
    if (testType) {
      params.append("test_type", testType);
    }

    const response = await fetch(`${API_BASE_URL}/testing/results?${params}`);
    if (!response.ok) {
      throw new Error("Failed to get test results");
    }
    return response.json();
  }

  async getTestRunDetails(runId: string) {
    const response = await fetch(`${API_BASE_URL}/testing/results/${runId}`);
    if (!response.ok) {
      throw new Error("Failed to get test run details");
    }
    return response.json();
  }

  async getTestStatistics() {
    const response = await fetch(`${API_BASE_URL}/testing/statistics`);
    if (!response.ok) {
      throw new Error("Failed to get test statistics");
    }
    return response.json();
  }

  connectWebSocket(): WebSocket {
    if (this.wsConnection && this.wsConnection.readyState === WebSocket.OPEN) {
      return this.wsConnection;
    }

    this.wsConnection = new WebSocket(`${WS_BASE_URL}/ws`);

    this.wsConnection.onopen = () => {
      console.log("WebSocket connected for field test updates");
    };

    this.wsConnection.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    this.wsConnection.onclose = () => {
      console.log("WebSocket disconnected");
      // Attempt reconnection after 5 seconds
      setTimeout(() => {
        if (this.wsConnection?.readyState === WebSocket.CLOSED) {
          this.connectWebSocket();
        }
      }, 5000);
    };

    return this.wsConnection;
  }

  disconnectWebSocket() {
    if (this.wsConnection) {
      this.wsConnection.close();
      this.wsConnection = null;
    }
  }
}

export const testingService = new TestingService();
