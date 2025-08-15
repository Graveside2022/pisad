import axios, { AxiosError } from "axios";
import { api } from "../../../src/frontend/src/services/api";

jest.mock("axios");
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe("ApiService", () => {
  let mockClient: any;

  beforeEach(() => {
    mockClient = {
      get: jest.fn(),
      post: jest.fn(),
      put: jest.fn(),
      delete: jest.fn(),
      interceptors: {
        response: {
          use: jest.fn(),
        },
      },
    };
    mockedAxios.create.mockReturnValue(mockClient);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe("initialization", () => {
    it("creates axios client with correct configuration", () => {
      expect(mockedAxios.create).toHaveBeenCalledWith({
        baseURL: "http://localhost:8000",
        timeout: 10000,
        headers: {
          "Content-Type": "application/json",
        },
      });
    });

    it("sets up response interceptor", () => {
      expect(mockClient.interceptors.response.use).toHaveBeenCalled();
    });
  });

  describe("getSystemStatus", () => {
    it("fetches system status successfully", async () => {
      const mockStatus = {
        state: "IDLE",
        homingEnabled: false,
        batteryPercent: 85,
        gpsStatus: "3D_FIX",
      };
      mockClient.get.mockResolvedValue({ data: mockStatus });

      const result = await api.getSystemStatus();

      expect(mockClient.get).toHaveBeenCalledWith("/api/system/status");
      expect(result).toEqual(mockStatus);
    });

    it("handles error when fetching system status", async () => {
      const error = new Error("Network error");
      mockClient.get.mockRejectedValue(error);

      await expect(api.getSystemStatus()).rejects.toThrow("Network error");
    });
  });

  describe("setHomingState", () => {
    it("enables homing with confirmation token", async () => {
      const mockResponse = {
        homingEnabled: true,
        message: "Homing enabled successfully",
      };
      mockClient.post.mockResolvedValue({ data: mockResponse });

      const result = await api.setHomingState(true, "confirm-123");

      expect(mockClient.post).toHaveBeenCalledWith("/api/system/homing", {
        enabled: true,
        confirmationToken: "confirm-123",
      });
      expect(result).toEqual(mockResponse);
    });

    it("disables homing without confirmation token", async () => {
      const mockResponse = {
        homingEnabled: false,
        message: "Homing disabled successfully",
      };
      mockClient.post.mockResolvedValue({ data: mockResponse });

      const result = await api.setHomingState(false, "");

      expect(mockClient.post).toHaveBeenCalledWith("/api/system/homing", {
        enabled: false,
        confirmationToken: "",
      });
      expect(result).toEqual(mockResponse);
    });

    it("handles safety interlock errors", async () => {
      const interceptor = mockClient.interceptors.response.use.mock.calls[0][1];
      const error: AxiosError = {
        response: {
          status: 403,
          data: {
            error: "Safety interlock blocked",
            blockedBy: ["modeCheck", "batteryCheck"],
          },
        },
      } as any;

      await expect(interceptor(error)).rejects.toThrow("Safety interlock blocked");
    });
  });

  describe("triggerEmergencyStop", () => {
    it("triggers emergency stop successfully", async () => {
      const mockResponse = {
        message: "Emergency stop activated",
        safetyStatus: { allSystemsSafe: true },
      };
      mockClient.post.mockResolvedValue({ data: mockResponse });

      const result = await api.triggerEmergencyStop();

      expect(mockClient.post).toHaveBeenCalledWith("/api/system/emergency-stop");
      expect(result).toEqual(mockResponse);
    });
  });

  describe("getSignalData", () => {
    it("fetches current signal data", async () => {
      const mockSignal = {
        rssi: -55,
        noiseFloor: -90,
        snr: 35,
        confidence: 0.95,
      };
      mockClient.get.mockResolvedValue({ data: mockSignal });

      const result = await api.getSignalData();

      expect(mockClient.get).toHaveBeenCalledWith("/api/signals/current");
      expect(result).toEqual(mockSignal);
    });
  });

  describe("getSignalHistory", () => {
    it("fetches signal history with default duration", async () => {
      const mockHistory = [
        { timestamp: "2025-08-12T10:00:00Z", rssi: -60 },
        { timestamp: "2025-08-12T10:00:01Z", rssi: -58 },
      ];
      mockClient.get.mockResolvedValue({ data: mockHistory });

      const result = await api.getSignalHistory();

      expect(mockClient.get).toHaveBeenCalledWith("/api/signals/history", {
        params: { minutes: 5 },
      });
      expect(result).toEqual(mockHistory);
    });

    it("fetches signal history with custom duration", async () => {
      const mockHistory = [
        { timestamp: "2025-08-12T10:00:00Z", rssi: -60 },
      ];
      mockClient.get.mockResolvedValue({ data: mockHistory });

      const result = await api.getSignalHistory(10);

      expect(mockClient.get).toHaveBeenCalledWith("/api/signals/history", {
        params: { minutes: 10 },
      });
      expect(result).toEqual(mockHistory);
    });
  });

  describe("getConfiguration", () => {
    it("fetches configuration successfully", async () => {
      const mockConfig = {
        sdr: { frequency: 433920000, sampleRate: 2400000 },
        beacon: { pattern: "short_long" },
      };
      mockClient.get.mockResolvedValue({ data: mockConfig });

      const result = await api.getConfiguration();

      expect(mockClient.get).toHaveBeenCalledWith("/api/config");
      expect(result).toEqual(mockConfig);
    });
  });

  describe("updateConfiguration", () => {
    it("updates configuration successfully", async () => {
      const newConfig = { sdr: { frequency: 433950000 } };
      const mockResponse = {
        message: "Configuration updated",
        config: newConfig,
      };
      mockClient.put.mockResolvedValue({ data: mockResponse });

      const result = await api.updateConfiguration(newConfig);

      expect(mockClient.put).toHaveBeenCalledWith("/api/config", newConfig);
      expect(result).toEqual(mockResponse);
    });
  });

  describe("navigation methods", () => {
    it("fetches velocity vectors", async () => {
      const mockVectors = { vx: 1.5, vy: 0.5, vz: -0.2 };
      mockClient.get.mockResolvedValue({ data: mockVectors });

      const result = await api.getVelocityVectors();

      expect(mockClient.get).toHaveBeenCalledWith("/api/navigation/velocity");
      expect(result).toEqual(mockVectors);
    });

    it("fetches drone position", async () => {
      const mockPosition = { lat: 42.3601, lon: -71.0589, alt: 50 };
      mockClient.get.mockResolvedValue({ data: mockPosition });

      const result = await api.getDronePosition();

      expect(mockClient.get).toHaveBeenCalledWith("/api/navigation/position");
      expect(result).toEqual(mockPosition);
    });

    it("fetches geofence boundary", async () => {
      const mockGeofence = {
        center: { lat: 42.3601, lon: -71.0589 },
        radius: 100,
      };
      mockClient.get.mockResolvedValue({ data: mockGeofence });

      const result = await api.getGeofenceBoundary();

      expect(mockClient.get).toHaveBeenCalledWith("/api/navigation/geofence");
      expect(result).toEqual(mockGeofence);
    });
  });

  describe("safety methods", () => {
    it("fetches safety status", async () => {
      const mockSafety = {
        interlocks: {
          modeCheck: true,
          batteryCheck: true,
          geofenceCheck: true,
        },
      };
      mockClient.get.mockResolvedValue({ data: mockSafety });

      const result = await api.getSafetyStatus();

      expect(mockClient.get).toHaveBeenCalledWith("/api/safety/status");
      expect(result).toEqual(mockSafety);
    });

    it("fetches safety events with default limit", async () => {
      const mockEvents = [
        { id: 1, type: "MODE_CHANGE", timestamp: "2025-08-12T10:00:00Z" },
        { id: 2, type: "LOW_BATTERY", timestamp: "2025-08-12T10:05:00Z" },
      ];
      mockClient.get.mockResolvedValue({ data: mockEvents });

      const result = await api.getSafetyEvents();

      expect(mockClient.get).toHaveBeenCalledWith("/api/safety/events", {
        params: { limit: 100 },
      });
      expect(result).toEqual(mockEvents);
    });

    it("fetches safety events with custom limit", async () => {
      const mockEvents = [
        { id: 1, type: "MODE_CHANGE", timestamp: "2025-08-12T10:00:00Z" },
      ];
      mockClient.get.mockResolvedValue({ data: mockEvents });

      const result = await api.getSafetyEvents(50);

      expect(mockClient.get).toHaveBeenCalledWith("/api/safety/events", {
        params: { limit: 50 },
      });
      expect(result).toEqual(mockEvents);
    });
  });

  describe("testConnection", () => {
    it("tests connection successfully", async () => {
      const mockHealth = { status: "healthy", uptime: 3600 };
      mockClient.get.mockResolvedValue({ data: mockHealth });

      const result = await api.testConnection();

      expect(mockClient.get).toHaveBeenCalledWith("/api/health");
      expect(result).toEqual(mockHealth);
    });
  });

  describe("HTTP method proxies", () => {
    it("proxies GET requests", async () => {
      const mockData = { test: "data" };
      mockClient.get.mockResolvedValue({ data: mockData });

      const result = await api.get("/test", { params: { id: 1 } });

      expect(mockClient.get).toHaveBeenCalledWith("/test", { params: { id: 1 } });
      expect(result.data).toEqual(mockData);
    });

    it("proxies POST requests", async () => {
      const postData = { name: "test" };
      const mockResponse = { id: 1, ...postData };
      mockClient.post.mockResolvedValue({ data: mockResponse });

      const result = await api.post("/test", postData);

      expect(mockClient.post).toHaveBeenCalledWith("/test", postData);
      expect(result.data).toEqual(mockResponse);
    });

    it("proxies PUT requests", async () => {
      const putData = { name: "updated" };
      const mockResponse = { id: 1, ...putData };
      mockClient.put.mockResolvedValue({ data: mockResponse });

      const result = await api.put("/test/1", putData);

      expect(mockClient.put).toHaveBeenCalledWith("/test/1", putData);
      expect(result.data).toEqual(mockResponse);
    });

    it("proxies DELETE requests", async () => {
      const mockResponse = { message: "deleted" };
      mockClient.delete.mockResolvedValue({ data: mockResponse });

      const result = await api.delete("/test/1");

      expect(mockClient.delete).toHaveBeenCalledWith("/test/1");
      expect(result.data).toEqual(mockResponse);
    });
  });

  describe("error handling", () => {
    it("handles non-403 errors normally", async () => {
      const interceptor = mockClient.interceptors.response.use.mock.calls[0][1];
      const error = {
        response: {
          status: 500,
          data: { error: "Internal server error" },
        },
      };

      await expect(interceptor(error)).rejects.toEqual(error);
    });

    it("handles 403 errors without blockedBy field", async () => {
      const interceptor = mockClient.interceptors.response.use.mock.calls[0][1];
      const error = {
        response: {
          status: 403,
          data: { error: "Forbidden" },
        },
      };

      await expect(interceptor(error)).rejects.toEqual(error);
    });

    it("handles network errors", async () => {
      const interceptor = mockClient.interceptors.response.use.mock.calls[0][1];
      const error = new Error("Network error");

      await expect(interceptor(error)).rejects.toEqual(error);
    });
  });
});
