export type SystemStateEnum = "IDLE" | "SEARCHING" | "DETECTING" | "HOMING" | "HOLDING";

export const SystemState = {
  IDLE: "IDLE" as const,
  SEARCHING: "SEARCHING" as const,
  DETECTING: "DETECTING" as const,
  HOMING: "HOMING" as const,
  HOLDING: "HOLDING" as const,
};

export interface StateTransition {
  id?: number;
  from_state: SystemStateEnum | string;
  to_state: SystemStateEnum | string;
  timestamp: string;
  reason: string;
  operator_id?: string;
  action_duration_ms?: number;
}

export interface StateUpdate {
  type: "state";
  data: {
    current_state: string;
    previous_state: string;
    timestamp: string;
    reason: string;
    allowed_transitions: string[];
    state_duration_ms: number;
    history: StateTransition[];
  };
}

export interface StateOverrideRequest {
  target_state: SystemStateEnum | string;
  reason: string;
  confirmation_token: string;
  operator_id: string;
}

export interface StateOverrideResponse {
  success: boolean;
  previous_state: string;
  new_state: string;
  message: string;
}

export interface StateMetrics {
  total_transitions: number;
  state_durations: Record<string, number>;
  transition_frequencies: Record<string, number>;
  average_transition_time_ms: number;
  last_restart_recovery?: {
    recovered_state: string;
    recovery_time_ms: number;
    timestamp: string;
  };
}
