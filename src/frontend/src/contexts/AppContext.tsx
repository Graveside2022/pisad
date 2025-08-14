import React, {
  createContext,
  useContext,
  useReducer,
  useEffect,
  type ReactNode,
} from "react";
import {
  configService,
  type ConfigProfile,
  type ProfileCreateRequest,
  type ProfileUpdateRequest,
} from "../services/config";
import type WebSocketService from "../services/websocket";
import type { SystemState } from "../types";

interface AppState {
  // Configuration state
  activeProfileId: string | null;
  profiles: ConfigProfile[];
  profilesLoading: boolean;
  profilesError: string | null;

  // System state
  sdrConnected: boolean;
  systemState: "IDLE" | "SEARCHING" | "DETECTING" | "HOMING" | "HOLDING";

  // Real-time data
  currentRSSI: number;
  noiseFloor: number;
  snr: number;
  detectionConfidence: number;
}

type AppAction =
  | { type: "SET_PROFILES"; payload: ConfigProfile[] }
  | { type: "SET_ACTIVE_PROFILE"; payload: string | null }
  | { type: "SET_PROFILES_LOADING"; payload: boolean }
  | { type: "SET_PROFILES_ERROR"; payload: string | null }
  | { type: "SET_SDR_CONNECTED"; payload: boolean }
  | { type: "SET_SYSTEM_STATE"; payload: AppState["systemState"] }
  | {
      type: "UPDATE_RSSI";
      payload: {
        rssi: number;
        noiseFloor: number;
        snr: number;
        confidence: number;
      };
    }
  | {
      type: "CONFIG_ACTIVATED";
      payload: { profileId: string; profile: ConfigProfile };
    };

const initialState: AppState = {
  activeProfileId: null,
  profiles: [],
  profilesLoading: false,
  profilesError: null,
  sdrConnected: false,
  systemState: "IDLE",
  currentRSSI: -100,
  noiseFloor: -90,
  snr: 0,
  detectionConfidence: 0,
};

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case "SET_PROFILES":
      return { ...state, profiles: action.payload, profilesError: null };

    case "SET_ACTIVE_PROFILE":
      return { ...state, activeProfileId: action.payload };

    case "SET_PROFILES_LOADING":
      return { ...state, profilesLoading: action.payload };

    case "SET_PROFILES_ERROR":
      return { ...state, profilesError: action.payload };

    case "SET_SDR_CONNECTED":
      return { ...state, sdrConnected: action.payload };

    case "SET_SYSTEM_STATE":
      return { ...state, systemState: action.payload };

    case "UPDATE_RSSI":
      return {
        ...state,
        currentRSSI: action.payload.rssi,
        noiseFloor: action.payload.noiseFloor,
        snr: action.payload.snr,
        detectionConfidence: action.payload.confidence,
      };

    case "CONFIG_ACTIVATED":
      return { ...state, activeProfileId: action.payload.profileId };

    default:
      return state;
  }
}

interface AppContextValue {
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
  loadProfiles: () => Promise<void>;
  activateProfile: (profileId: string) => Promise<void>;
  createProfile: (profile: ProfileCreateRequest) => Promise<void>;
  updateProfile: (
    profileId: string,
    profile: ProfileUpdateRequest,
  ) => Promise<void>;
  deleteProfile: (profileId: string) => Promise<void>;
}

const AppContext = createContext<AppContextValue | undefined>(undefined);

export function useAppContext() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error("useAppContext must be used within AppProvider");
  }
  return context;
}

interface AppProviderProps {
  children: ReactNode;
  webSocketService?: typeof WebSocketService;
}

export function AppProvider({ children, webSocketService }: AppProviderProps) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  // Load profiles on mount
  useEffect(() => {
    loadProfiles();
  }, []);

  // Subscribe to WebSocket events
  useEffect(() => {
    if (!webSocketService) return;

    // Handle RSSI updates
    const handleRSSI = (data: {
      data: {
        rssi: number;
        noiseFloor: number;
        snr: number;
        confidence?: number;
      };
    }) => {
      dispatch({
        type: "UPDATE_RSSI",
        payload: {
          rssi: data.data.rssi,
          noiseFloor: data.data.noiseFloor,
          snr: data.data.snr,
          confidence: data.data.confidence || 0,
        },
      });
    };

    // Handle configuration updates
    const handleConfig = (data: {
      action: string;
      profile?: ConfigProfile;
    }) => {
      if (data.action === "profile_activated") {
        if (data.profile) {
          dispatch({
            type: "CONFIG_ACTIVATED",
            payload: {
              profileId: data.profile.id,
              profile: data.profile,
            },
          });
        }
        // Reload profiles to get updated state
        loadProfiles();
      }
    };

    // Handle state changes
    const handleState = (data: {
      data: { state?: SystemState; sdrConnected?: boolean };
    }) => {
      if (data.data.state) {
        dispatch({ type: "SET_SYSTEM_STATE", payload: data.data.state.current_state });
      }
      if (data.data.sdrConnected !== undefined) {
        dispatch({
          type: "SET_SDR_CONNECTED",
          payload: data.data.sdrConnected,
        });
      }
    };

    webSocketService.on("rssi", handleRSSI);
    webSocketService.on("config", handleConfig);
    webSocketService.on("state", handleState);

    return () => {
      webSocketService.off("rssi", handleRSSI);
      webSocketService.off("config", handleConfig);
      webSocketService.off("state", handleState);
    };
  }, [webSocketService]);

  const loadProfiles = async () => {
    dispatch({ type: "SET_PROFILES_LOADING", payload: true });
    try {
      const profiles = await configService.getProfiles();
      dispatch({ type: "SET_PROFILES", payload: profiles });

      // Find and set active profile (if any is marked as active/default)
      const activeProfile = profiles.find((p) => p.isDefault);
      if (activeProfile) {
        dispatch({ type: "SET_ACTIVE_PROFILE", payload: activeProfile.id });
      }
    } catch (error) {
      dispatch({ type: "SET_PROFILES_ERROR", payload: String(error) });
    } finally {
      dispatch({ type: "SET_PROFILES_LOADING", payload: false });
    }
  };

  const activateProfile = async (profileId: string) => {
    try {
      await configService.activateProfile(profileId);
      dispatch({ type: "SET_ACTIVE_PROFILE", payload: profileId });
      // Reload profiles to get updated state
      await loadProfiles();
    } catch (error) {
      dispatch({ type: "SET_PROFILES_ERROR", payload: String(error) });
      throw error;
    }
  };

  const createProfile = async (profile: ProfileCreateRequest) => {
    try {
      await configService.createProfile(profile);
      await loadProfiles();
    } catch (error) {
      dispatch({ type: "SET_PROFILES_ERROR", payload: String(error) });
      throw error;
    }
  };

  const updateProfile = async (
    profileId: string,
    profile: ProfileUpdateRequest,
  ) => {
    try {
      await configService.updateProfile(profileId, profile);
      await loadProfiles();
    } catch (error) {
      dispatch({ type: "SET_PROFILES_ERROR", payload: String(error) });
      throw error;
    }
  };

  const deleteProfile = async (profileId: string) => {
    try {
      await configService.deleteProfile(profileId);
      await loadProfiles();
    } catch (error) {
      dispatch({ type: "SET_PROFILES_ERROR", payload: String(error) });
      throw error;
    }
  };

  const value: AppContextValue = {
    state,
    dispatch,
    loadProfiles,
    activateProfile,
    createProfile,
    updateProfile,
    deleteProfile,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}
