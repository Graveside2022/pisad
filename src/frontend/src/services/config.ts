/**
 * Configuration service for managing SDR and system configuration profiles
 */

import { apiClient } from "./api";

export interface SDRConfig {
  frequency: number;
  sampleRate: number;
  gain: number | string;
  bandwidth: number;
}

export interface SignalConfig {
  fftSize: number;
  ewmaAlpha: number;
  triggerThreshold: number;
  dropThreshold: number;
}

export interface HomingConfig {
  forwardVelocityMax: number;
  yawRateMax: number;
  approachVelocity: number;
  signalLossTimeout: number;
}

export interface ConfigProfile {
  id: string;
  name: string;
  description: string;
  sdrConfig: SDRConfig;
  signalConfig: SignalConfig;
  homingConfig: HomingConfig;
  isDefault: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface ProfileCreateRequest {
  name: string;
  description: string;
  sdrConfig: SDRConfig;
  signalConfig: SignalConfig;
  homingConfig: HomingConfig;
  isDefault?: boolean;
}

export interface ProfileUpdateRequest {
  name: string;
  description: string;
  sdrConfig: SDRConfig;
  signalConfig: SignalConfig;
  homingConfig: HomingConfig;
  isDefault?: boolean;
}

class ConfigService {
  private baseUrl = "/config";

  /**
   * Get all configuration profiles
   */
  async getProfiles(): Promise<ConfigProfile[]> {
    const response = await apiClient.get<ConfigProfile[]>(
      `${this.baseUrl}/profiles`,
    );
    return response.data;
  }

  /**
   * Get a specific configuration profile by ID
   */
  async getProfile(profileId: string): Promise<ConfigProfile> {
    const response = await apiClient.get<ConfigProfile>(
      `${this.baseUrl}/profiles/${profileId}`,
    );
    return response.data;
  }

  /**
   * Create a new configuration profile
   */
  async createProfile(profile: ProfileCreateRequest): Promise<ConfigProfile> {
    const response = await apiClient.post<ConfigProfile>(
      `${this.baseUrl}/profiles`,
      profile,
    );
    return response.data;
  }

  /**
   * Update an existing configuration profile
   */
  async updateProfile(
    profileId: string,
    profile: ProfileUpdateRequest,
  ): Promise<ConfigProfile> {
    const response = await apiClient.put<ConfigProfile>(
      `${this.baseUrl}/profiles/${profileId}`,
      profile,
    );
    return response.data;
  }

  /**
   * Delete a configuration profile
   */
  async deleteProfile(profileId: string): Promise<void> {
    await apiClient.delete(`${this.baseUrl}/profiles/${profileId}`);
  }

  /**
   * Activate a configuration profile (apply to system)
   */
  async activateProfile(
    profileId: string,
  ): Promise<{ status: string; message: string }> {
    const response = await apiClient.post<{ status: string; message: string }>(
      `${this.baseUrl}/profiles/${profileId}/activate`,
    );
    return response.data;
  }

  /**
   * Validate a configuration profile without saving
   */
  validateProfile(profile: Partial<ConfigProfile>): {
    valid: boolean;
    errors: string[];
  } {
    const errors: string[] = [];

    // Validate SDR configuration
    if (profile.sdrConfig) {
      if (
        profile.sdrConfig.frequency < 1e6 ||
        profile.sdrConfig.frequency > 6e9
      ) {
        errors.push("Frequency must be between 1 MHz and 6 GHz");
      }
      if (
        profile.sdrConfig.sampleRate < 0.25e6 ||
        profile.sdrConfig.sampleRate > 20e6
      ) {
        errors.push("Sample rate must be between 0.25 Msps and 20 Msps");
      }
      if (typeof profile.sdrConfig.gain === "number") {
        if (profile.sdrConfig.gain < -10 || profile.sdrConfig.gain > 70) {
          errors.push("Gain must be between -10 dB and 70 dB");
        }
      }
    }

    // Validate signal configuration
    if (profile.signalConfig) {
      if (
        profile.signalConfig.ewmaAlpha <= 0 ||
        profile.signalConfig.ewmaAlpha > 1
      ) {
        errors.push("EWMA alpha must be between 0 and 1");
      }
      if (
        profile.signalConfig.dropThreshold >=
        profile.signalConfig.triggerThreshold
      ) {
        errors.push("Drop threshold must be less than trigger threshold");
      }
    }

    // Validate homing configuration
    if (profile.homingConfig) {
      if (profile.homingConfig.forwardVelocityMax <= 0) {
        errors.push("Forward velocity max must be positive");
      }
      if (profile.homingConfig.yawRateMax <= 0) {
        errors.push("Yaw rate max must be positive");
      }
      if (profile.homingConfig.signalLossTimeout <= 0) {
        errors.push("Signal loss timeout must be positive");
      }
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  }

  /**
   * Format frequency for display
   */
  formatFrequency(frequency: number): string {
    if (frequency >= 1e9) {
      return `${(frequency / 1e9).toFixed(3)} GHz`;
    } else if (frequency >= 1e6) {
      return `${(frequency / 1e6).toFixed(3)} MHz`;
    } else {
      return `${(frequency / 1e3).toFixed(3)} kHz`;
    }
  }

  /**
   * Format sample rate for display
   */
  formatSampleRate(sampleRate: number): string {
    if (sampleRate >= 1e6) {
      return `${(sampleRate / 1e6).toFixed(1)} Msps`;
    } else {
      return `${(sampleRate / 1e3).toFixed(1)} ksps`;
    }
  }

  /**
   * Get default profile values
   */
  getDefaultProfile(): Partial<ConfigProfile> {
    return {
      name: "",
      description: "",
      sdrConfig: {
        frequency: 2437000000, // 2.437 GHz (WiFi Channel 6)
        sampleRate: 2000000, // 2 Msps
        gain: 40,
        bandwidth: 2000000,
      },
      signalConfig: {
        fftSize: 1024,
        ewmaAlpha: 0.1,
        triggerThreshold: -60,
        dropThreshold: -70,
      },
      homingConfig: {
        forwardVelocityMax: 5.0,
        yawRateMax: 1.0,
        approachVelocity: 2.0,
        signalLossTimeout: 5.0,
      },
      isDefault: false,
    };
  }
}

// Export singleton instance
export const configService = new ConfigService();
