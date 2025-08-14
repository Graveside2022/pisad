import React, { createContext, useContext, useReducer, type ReactNode } from "react";

interface HomingState {
  isEnabled: boolean;
  isActivating: boolean;
  confirmationDialogOpen: boolean;
  confirmationToken: string;
  lastActivationAttempt: Date | null;
  activationHistory: ActivationAttempt[];
  error: string | null;
}

interface ActivationAttempt {
  timestamp: Date;
  success: boolean;
  error?: string;
  blockedBy?: string[];
}

type HomingAction =
  | { type: "SET_ENABLED"; payload: boolean }
  | { type: "SET_ACTIVATING"; payload: boolean }
  | { type: "OPEN_CONFIRMATION_DIALOG"; payload: string }
  | { type: "CLOSE_CONFIRMATION_DIALOG" }
  | { type: "ACTIVATION_SUCCESS" }
  | {
      type: "ACTIVATION_FAILURE";
      payload: { error: string; blockedBy?: string[] };
    }
  | { type: "CLEAR_ERROR" }
  | { type: "RESET" };

const initialState: HomingState = {
  isEnabled: false,
  isActivating: false,
  confirmationDialogOpen: false,
  confirmationToken: "",
  lastActivationAttempt: null,
  activationHistory: [],
  error: null,
};

const homingReducer = (
  state: HomingState,
  action: HomingAction,
): HomingState => {
  switch (action.type) {
    case "SET_ENABLED":
      return {
        ...state,
        isEnabled: action.payload,
        error: null,
      };

    case "SET_ACTIVATING":
      return {
        ...state,
        isActivating: action.payload,
      };

    case "OPEN_CONFIRMATION_DIALOG":
      return {
        ...state,
        confirmationDialogOpen: true,
        confirmationToken: action.payload,
      };

    case "CLOSE_CONFIRMATION_DIALOG":
      return {
        ...state,
        confirmationDialogOpen: false,
        confirmationToken: "",
      };

    case "ACTIVATION_SUCCESS": {
      const successAttempt: ActivationAttempt = {
        timestamp: new Date(),
        success: true,
      };
      return {
        ...state,
        isEnabled: true,
        isActivating: false,
        confirmationDialogOpen: false,
        confirmationToken: "",
        lastActivationAttempt: new Date(),
        activationHistory: [...state.activationHistory, successAttempt].slice(
          -10,
        ),
        error: null,
      };
    }

    case "ACTIVATION_FAILURE": {
      const failureAttempt: ActivationAttempt = {
        timestamp: new Date(),
        success: false,
        error: action.payload.error,
        blockedBy: action.payload.blockedBy,
      };
      return {
        ...state,
        isActivating: false,
        confirmationDialogOpen: false,
        confirmationToken: "",
        lastActivationAttempt: new Date(),
        activationHistory: [...state.activationHistory, failureAttempt].slice(
          -10,
        ),
        error: action.payload.error,
      };
    }

    case "CLEAR_ERROR":
      return {
        ...state,
        error: null,
      };

    case "RESET":
      return initialState;

    default:
      return state;
  }
};

interface HomingContextType {
  state: HomingState;
  dispatch: React.Dispatch<HomingAction>;
  generateConfirmationToken: () => string;
  canActivateHoming: (safetyInterlocks: SafetyInterlocks | null) => boolean;
}

interface SafetyInterlocks {
  modeCheck: boolean;
  batteryCheck: boolean;
  geofenceCheck: boolean;
  signalCheck: boolean;
  operatorCheck: boolean;
}

const HomingContext = createContext<HomingContextType | undefined>(undefined);

export const HomingProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const [state, dispatch] = useReducer(homingReducer, initialState);

  const generateConfirmationToken = (): string => {
    return `confirm-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  };

  const canActivateHoming = (
    safetyInterlocks: SafetyInterlocks | null,
  ): boolean => {
    if (!safetyInterlocks) return false;
    return Object.values(safetyInterlocks).every((check) => check === true);
  };

  return (
    <HomingContext.Provider
      value={{
        state,
        dispatch,
        generateConfirmationToken,
        canActivateHoming,
      }}
    >
      {children}
    </HomingContext.Provider>
  );
};

// eslint-disable-next-line react-refresh/only-export-components
export const useHoming = () => {
  const context = useContext(HomingContext);
  if (context === undefined) {
    throw new Error("useHoming must be used within a HomingProvider");
  }
  return context;
};
