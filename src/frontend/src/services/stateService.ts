import { apiClient } from "./api";

interface StateOverrideRequest {
  target_state: string;
  reason: string;
  confirmation_token: string;
  operator_id: string;
}

interface StateOverrideResponse {
  success: boolean;
  previous_state: string;
  new_state: string;
  message: string;
}

interface StateTransition {
  id?: number;
  from_state: string;
  to_state: string;
  timestamp: string;
  reason: string;
  operator_id?: string;
  action_duration_ms?: number;
}

interface CurrentStateResponse {
  state: string;
  allowed_transitions: string[];
  state_duration_ms: number;
  last_transition: StateTransition | null;
}

class StateService {
  async getCurrentState(): Promise<CurrentStateResponse> {
    const response = await apiClient.get("/api/system/state");
    return response.data;
  }

  async overrideState(
    request: StateOverrideRequest,
  ): Promise<StateOverrideResponse> {
    const response = await apiClient.post(
      "/api/system/state-override",
      request,
    );
    return response.data;
  }

  async getStateHistory(limit: number = 100): Promise<StateTransition[]> {
    const response = await apiClient.get("/api/system/state-history", {
      params: { limit },
    });
    return response.data;
  }

  async getStateMetrics(): Promise<any> {
    const response = await apiClient.get("/api/system/state-metrics");
    return response.data;
  }
}

export const stateService = new StateService();
