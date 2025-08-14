import { useState, useCallback } from "react";
import { stateService } from "../services/stateService";
import { type StateOverrideRequest, type StateOverrideResponse } from "../types/state";

interface UseStateOverrideResult {
  overrideState: (
    request: StateOverrideRequest,
  ) => Promise<StateOverrideResponse>;
  isLoading: boolean;
  error: string | null;
}

export const useStateOverride = (): UseStateOverrideResult => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const overrideState = useCallback(
    async (request: StateOverrideRequest): Promise<StateOverrideResponse> => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await stateService.overrideState(request);
        return response;
      } catch (err) {
        const errorMessage = (err as Error).message;
        setError(errorMessage);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [],
  );

  return {
    overrideState,
    isLoading,
    error,
  };
};
