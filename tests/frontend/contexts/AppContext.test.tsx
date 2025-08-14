import React from "react";
import { render, screen, act } from "@testing-library/react";
import "@testing-library/jest-dom";
import { AppProvider, useAppContext } from "../../../src/frontend/src/contexts/AppContext";

// Test component to access context
const TestComponent = () => {
  const context = useAppContext();
  
  return (
    <div>
      <div data-testid="state">{context.state.currentMode}</div>
      <div data-testid="loading">{context.state.isLoading.toString()}</div>
      <div data-testid="error">{context.state.error || "no-error"}</div>
      <button 
        onClick={() => context.dispatch({ type: "SET_MODE", payload: "ACTIVE" })}
      >
        Set Active
      </button>
      <button
        onClick={() => context.dispatch({ type: "SET_LOADING", payload: true })}
      >
        Set Loading
      </button>
      <button
        onClick={() => context.dispatch({ type: "SET_ERROR", payload: "Test error" })}
      >
        Set Error
      </button>
    </div>
  );
};

describe("AppContext", () => {
  describe("AppProvider", () => {
    it("provides initial state to children", () => {
      render(
        <AppProvider>
          <TestComponent />
        </AppProvider>
      );

      expect(screen.getByTestId("state")).toHaveTextContent("IDLE");
      expect(screen.getByTestId("loading")).toHaveTextContent("false");
      expect(screen.getByTestId("error")).toHaveTextContent("no-error");
    });

    it("updates state when dispatch is called", () => {
      render(
        <AppProvider>
          <TestComponent />
        </AppProvider>
      );

      const setActiveButton = screen.getByText("Set Active");
      
      act(() => {
        setActiveButton.click();
      });

      expect(screen.getByTestId("state")).toHaveTextContent("ACTIVE");
    });

    it("handles loading state changes", () => {
      render(
        <AppProvider>
          <TestComponent />
        </AppProvider>
      );

      const setLoadingButton = screen.getByText("Set Loading");
      
      act(() => {
        setLoadingButton.click();
      });

      expect(screen.getByTestId("loading")).toHaveTextContent("true");
    });

    it("handles error state changes", () => {
      render(
        <AppProvider>
          <TestComponent />
        </AppProvider>
      );

      const setErrorButton = screen.getByText("Set Error");
      
      act(() => {
        setErrorButton.click();
      });

      expect(screen.getByTestId("error")).toHaveTextContent("Test error");
    });
  });

  describe("useAppContext", () => {
    it("throws error when used outside of AppProvider", () => {
      // Suppress console.error for this test
      const originalError = console.error;
      console.error = jest.fn();

      expect(() => {
        render(<TestComponent />);
      }).toThrow("useAppContext must be used within an AppProvider");

      console.error = originalError;
    });
  });

  describe("reducer", () => {
    const TestReducerComponent = () => {
      const { state, dispatch } = useAppContext();
      
      return (
        <div>
          <div data-testid="mode">{state.currentMode}</div>
          <div data-testid="connected">{state.isConnected?.toString() || "undefined"}</div>
          <button
            onClick={() => dispatch({ type: "SET_CONNECTION", payload: true })}
          >
            Connect
          </button>
          <button
            onClick={() => dispatch({ type: "SET_CONNECTION", payload: false })}
          >
            Disconnect
          </button>
          <button
            onClick={() => dispatch({ type: "RESET_STATE" })}
          >
            Reset
          </button>
          <button
            onClick={() => dispatch({ 
              type: "UPDATE_CONFIG", 
              payload: { frequency: 433950000 } 
            })}
          >
            Update Config
          </button>
        </div>
      );
    };

    it("handles connection state changes", () => {
      render(
        <AppProvider>
          <TestReducerComponent />
        </AppProvider>
      );

      expect(screen.getByTestId("connected")).toHaveTextContent("undefined");

      act(() => {
        screen.getByText("Connect").click();
      });
      expect(screen.getByTestId("connected")).toHaveTextContent("true");

      act(() => {
        screen.getByText("Disconnect").click();
      });
      expect(screen.getByTestId("connected")).toHaveTextContent("false");
    });

    it("handles state reset", () => {
      render(
        <AppProvider>
          <TestReducerComponent />
        </AppProvider>
      );

      // Change some state
      act(() => {
        screen.getByText("Connect").click();
      });
      expect(screen.getByTestId("connected")).toHaveTextContent("true");

      // Reset state
      act(() => {
        screen.getByText("Reset").click();
      });
      expect(screen.getByTestId("mode")).toHaveTextContent("IDLE");
      expect(screen.getByTestId("connected")).toHaveTextContent("undefined");
    });

    it("handles config updates", () => {
      const TestConfigComponent = () => {
        const { state, dispatch } = useAppContext();
        
        return (
          <div>
            <div data-testid="config">
              {JSON.stringify(state.config || {})}
            </div>
            <button
              onClick={() => dispatch({ 
                type: "UPDATE_CONFIG", 
                payload: { frequency: 433950000, sampleRate: 2400000 } 
              })}
            >
              Update
            </button>
          </div>
        );
      };

      render(
        <AppProvider>
          <TestConfigComponent />
        </AppProvider>
      );

      act(() => {
        screen.getByText("Update").click();
      });

      const configData = screen.getByTestId("config").textContent;
      expect(configData).toContain("433950000");
      expect(configData).toContain("2400000");
    });
  });

  describe("multiple consumers", () => {
    const Consumer1 = () => {
      const { state, dispatch } = useAppContext();
      return (
        <div>
          <div data-testid="consumer1-mode">{state.currentMode}</div>
          <button onClick={() => dispatch({ type: "SET_MODE", payload: "SEARCHING" })}>
            Consumer1 Action
          </button>
        </div>
      );
    };

    const Consumer2 = () => {
      const { state } = useAppContext();
      return (
        <div data-testid="consumer2-mode">{state.currentMode}</div>
      );
    };

    it("shares state between multiple consumers", () => {
      render(
        <AppProvider>
          <Consumer1 />
          <Consumer2 />
        </AppProvider>
      );

      expect(screen.getByTestId("consumer1-mode")).toHaveTextContent("IDLE");
      expect(screen.getByTestId("consumer2-mode")).toHaveTextContent("IDLE");

      act(() => {
        screen.getByText("Consumer1 Action").click();
      });

      expect(screen.getByTestId("consumer1-mode")).toHaveTextContent("SEARCHING");
      expect(screen.getByTestId("consumer2-mode")).toHaveTextContent("SEARCHING");
    });
  });

  describe("complex state updates", () => {
    const ComplexComponent = () => {
      const { state, dispatch } = useAppContext();
      
      const handleComplexUpdate = () => {
        dispatch({ type: "SET_LOADING", payload: true });
        dispatch({ type: "SET_MODE", payload: "HOMING" });
        dispatch({ type: "UPDATE_CONFIG", payload: { homingEnabled: true } });
        dispatch({ type: "SET_LOADING", payload: false });
      };
      
      return (
        <div>
          <div data-testid="complex-state">
            {JSON.stringify({
              mode: state.currentMode,
              loading: state.isLoading,
              config: state.config,
            })}
          </div>
          <button onClick={handleComplexUpdate}>Complex Update</button>
        </div>
      );
    };

    it("handles multiple sequential dispatches", () => {
      render(
        <AppProvider>
          <ComplexComponent />
        </AppProvider>
      );

      act(() => {
        screen.getByText("Complex Update").click();
      });

      const state = JSON.parse(screen.getByTestId("complex-state").textContent!);
      expect(state.mode).toBe("HOMING");
      expect(state.loading).toBe(false);
      expect(state.config?.homingEnabled).toBe(true);
    });
  });
});