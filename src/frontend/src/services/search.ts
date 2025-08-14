import {
  type SearchPatternRequest,
  type SearchPatternResponse,
  type SearchPatternPreview,
  type SearchPatternStatus,
  type PatternControlResponse,
} from "../types/search";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

class SearchService {
  async createPattern(
    request: SearchPatternRequest,
  ): Promise<SearchPatternResponse> {
    const response = await fetch(`${API_BASE_URL}/api/search/pattern`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(
        `Failed to create search pattern: ${response.statusText}`,
      );
    }

    return response.json();
  }

  async getPatternPreview(): Promise<SearchPatternPreview> {
    const response = await fetch(`${API_BASE_URL}/api/search/pattern/preview`);

    if (!response.ok) {
      throw new Error(`Failed to get pattern preview: ${response.statusText}`);
    }

    return response.json();
  }

  async getPatternStatus(patternId?: string): Promise<SearchPatternStatus> {
    const url = patternId
      ? `${API_BASE_URL}/api/search/pattern/status?pattern_id=${patternId}`
      : `${API_BASE_URL}/api/search/pattern/status`;
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Failed to get pattern status: ${response.statusText}`);
    }

    return response.json();
  }

  async controlPattern(
    action: string,
    patternId?: string,
  ): Promise<PatternControlResponse> {
    const url = patternId
      ? `${API_BASE_URL}/api/search/pattern/control?pattern_id=${patternId}`
      : `${API_BASE_URL}/api/search/pattern/control`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ action }),
    });

    if (!response.ok) {
      throw new Error(`Failed to control pattern: ${response.statusText}`);
    }

    return response.json();
  }

  async exportPattern(
    patternId?: string,
    format: "qgc" | "mission_planner" | "kml" = "qgc",
  ): Promise<string> {
    const url = patternId
      ? `${API_BASE_URL}/api/search/pattern/export?pattern_id=${patternId}&format=${format}`
      : `${API_BASE_URL}/api/search/pattern/export?format=${format}`;
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Failed to export pattern: ${response.statusText}`);
    }

    return response.text();
  }
}

export default new SearchService();
