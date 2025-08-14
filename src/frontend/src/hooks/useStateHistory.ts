import { useState, useEffect } from "react";
import { stateService } from "../services/stateService";

interface StateTransition {
  id?: number;
  from_state: string;
  to_state: string;
  timestamp: string;
  reason: string;
  operator_id?: string;
  action_duration_ms?: number;
}

interface UseStateHistoryResult {
  history: StateTransition[];
  isLoading: boolean;
  error: string | null;
  refreshHistory: () => Promise<void>;
}

export const useStateHistory = (limit: number = 100): UseStateHistoryResult => {
  const [history, setHistory] = useState<StateTransition[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchHistory = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const data = await stateService.getStateHistory(limit);
      setHistory(data);
    } catch (err) {
      setError((err as Error).message);
      console.error("Failed to fetch state history:", err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [limit]);

  return {
    history,
    isLoading,
    error,
    refreshHistory: fetchHistory,
  };
};
