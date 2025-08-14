import axios, { type AxiosInstance, AxiosError, type AxiosResponse } from "axios";

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

// Define response types for better type safety
interface HomingStateResponse {
  homingEnabled: boolean;
  message: string;
}

interface EmergencyStopResponse {
  message: string;
  safetyStatus?: unknown;
}

interface SafetyInterlockError {
  error?: string;
  blockedBy?: string[];
}

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 10000,
      headers: {
        "Content-Type": "application/json",
      },
    });

    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError<SafetyInterlockError>) => {
        if (error.response?.status === 403) {
          const data = error.response.data;
          if (data?.blockedBy) {
            const customError = new Error(
              data.error || "Safety interlock blocked",
            );
            (
              customError as { response?: AxiosResponse<SafetyInterlockError> }
            ).response = error.response;
            throw customError;
          }
        }
        throw error;
      },
    );
  }

  async getSystemStatus() {
    const response = await this.client.get("/api/system/status");
    return response.data;
  }

  async setHomingState(
    enabled: boolean,
    confirmationToken: string,
  ): Promise<HomingStateResponse> {
    const response = await this.client.post<HomingStateResponse>(
      "/api/system/homing",
      {
        enabled,
        confirmationToken,
      },
    );
    return response.data;
  }

  async triggerEmergencyStop(): Promise<EmergencyStopResponse> {
    const response = await this.client.post<EmergencyStopResponse>(
      "/api/system/emergency-stop",
    );
    return response.data;
  }

  async getSignalData() {
    const response = await this.client.get("/api/signals/current");
    return response.data;
  }

  async getSignalHistory(minutes: number = 5) {
    const response = await this.client.get("/api/signals/history", {
      params: { minutes },
    });
    return response.data;
  }

  async getConfiguration() {
    const response = await this.client.get("/api/config");
    return response.data;
  }

  async updateConfiguration(config: Record<string, unknown>) {
    const response = await this.client.put("/api/config", config);
    return response.data;
  }

  async getVelocityVectors() {
    const response = await this.client.get("/api/navigation/velocity");
    return response.data;
  }

  async getDronePosition() {
    const response = await this.client.get("/api/navigation/position");
    return response.data;
  }

  async getGeofenceBoundary() {
    const response = await this.client.get("/api/navigation/geofence");
    return response.data;
  }

  async getSafetyStatus() {
    const response = await this.client.get("/api/safety/status");
    return response.data;
  }

  async getSafetyEvents(limit: number = 100) {
    const response = await this.client.get("/api/safety/events", {
      params: { limit },
    });
    return response.data;
  }

  async testConnection() {
    const response = await this.client.get("/api/health");
    return response.data;
  }

  // Expose underlying HTTP methods for other services
  get<T = any>(url: string, config?: any) {
    return this.client.get<T>(url, config);
  }

  post<T = any>(url: string, data?: any, config?: any) {
    return this.client.post<T>(url, data, config);
  }

  put<T = any>(url: string, data?: any, config?: any) {
    return this.client.put<T>(url, data, config);
  }

  delete<T = any>(url: string, config?: any) {
    return this.client.delete<T>(url, config);
  }
}

export const api = new ApiService();
export const apiClient = api;
