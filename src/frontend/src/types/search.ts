export type PatternType = "expanding_square" | "spiral" | "lawnmower";
export type PatternState = "IDLE" | "EXECUTING" | "PAUSED" | "COMPLETED";
export type BoundaryType = "center_radius" | "corners";

export interface Coordinate {
  lat: number;
  lon: number;
}

export interface Waypoint extends Coordinate {
  index: number;
  alt: number;
}

export interface CenterRadiusBoundary {
  type: "center_radius";
  center: Coordinate;
  radius: number;
}

export interface CornerBoundary {
  type: "corners";
  corners: Coordinate[];
}

export type SearchBoundary = CenterRadiusBoundary | CornerBoundary;

export interface SearchPatternRequest {
  pattern: PatternType;
  spacing: number;
  velocity: number;
  bounds: SearchBoundary;
}

export interface SearchPatternResponse {
  pattern_id: string;
  waypoint_count: number;
  estimated_duration: number;
  total_distance: number;
}

export interface SearchPatternPreview {
  waypoints: Waypoint[];
  boundary: {
    type?: string;
    center?: [number, number];
    radius?: number;
    coordinates?: number[][][];
  } | null;
  total_distance: number;
  estimated_time: number;
}

export interface SearchPatternStatus {
  pattern_id: string;
  state: PatternState;
  progress_percent: number;
  completed_waypoints: number;
  total_waypoints: number;
  current_waypoint: number;
  estimated_time_remaining: number;
}

export type PatternStatus = SearchPatternStatus;
export type PatternControlAction = "pause" | "resume" | "stop";

export interface PatternControlRequest {
  action: "pause" | "resume" | "stop";
}

export interface PatternControlResponse {
  success: boolean;
  new_state: PatternState;
}

export interface SearchPatternWebSocketMessage {
  type:
    | "pattern_created"
    | "pattern_update"
    | "pattern_pause"
    | "pattern_resume"
    | "pattern_stop";
  data: {
    pattern_id: string;
    state: PatternState;
    progress_percent: number;
    completed_waypoints: number;
    total_waypoints: number;
    estimated_time_remaining: number;
    timestamp: string;
  };
}
