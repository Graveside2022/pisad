import React, { useState, useEffect, useCallback } from "react";
import { Box, Paper, Typography, Tabs, Tab, Button } from "@mui/material";
import { Add as AddIcon } from "@mui/icons-material";
import { ProfileManager } from "./ProfileManager";
import { SDRSettings } from "./SDRSettings";

interface ConfigProfile {
  id?: string;
  name: string;
  description: string;
  sdrConfig: {
    frequency: number;
    sampleRate: number;
    gain: number | string;
    bandwidth: number;
  };
  signalConfig: {
    fftSize: number;
    ewmaAlpha: number;
    triggerThreshold: number;
    dropThreshold: number;
  };
  homingConfig: {
    forwardVelocityMax: number;
    yawRateMax: number;
    approachVelocity: number;
    signalLossTimeout: number;
  };
  isDefault: boolean;
  createdAt?: string;
  updatedAt?: string;
}

interface ConfigPanelProps {
  configService: typeof import("../../services/config").configService;
  webSocketService?: typeof import("../../services/websocket").websocketService;
  activeProfileId?: string;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`config-tabpanel-${index}`}
      aria-labelledby={`config-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  );
}

export const ConfigPanel: React.FC<ConfigPanelProps> = ({
  configService,
  webSocketService,
  activeProfileId,
}) => {
  const [tabValue, setTabValue] = useState(0);
  const [profiles, setProfiles] = useState<ConfigProfile[]>([]);
  const [editingProfile, setEditingProfile] = useState<ConfigProfile | null>(
    null,
  );
  const [showSettings, setShowSettings] = useState(false);

  const loadProfiles = useCallback(async () => {
    try {
      const profileList = await configService.getProfiles();
      setProfiles(profileList);
    } catch (error) {
      console.error("Failed to load profiles:", error);
    }
  }, [configService]);

  const handleConfigUpdate = useCallback(
    (data: { action: string; profileId?: string }) => {
      if (data.action === "profile_activated") {
        // Reload profiles to reflect changes
        loadProfiles();
      }
    },
    [loadProfiles],
  );

  useEffect(() => {
    loadProfiles();

    // Listen for WebSocket configuration updates if available
    if (webSocketService) {
      webSocketService.on("config", handleConfigUpdate);

      return () => {
        webSocketService.off("config", handleConfigUpdate);
      };
    }
  }, [webSocketService, handleConfigUpdate, loadProfiles]);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleActivateProfile = async (profileId: string) => {
    try {
      await configService.activateProfile(profileId);
      // Reload profiles to reflect the active state
      await loadProfiles();
    } catch (error) {
      console.error("Failed to activate profile:", error);
      throw error;
    }
  };

  const handleSaveProfile = async (profile: ConfigProfile) => {
    try {
      if (profile.id) {
        // Update existing profile
        await configService.updateProfile(profile.id, profile);
      } else {
        // Create new profile
        await configService.createProfile(profile);
      }

      setShowSettings(false);
      setEditingProfile(null);
      await loadProfiles();
    } catch (error) {
      console.error("Failed to save profile:", error);
      throw error;
    }
  };

  const handleDeleteProfile = async (profileId: string) => {
    try {
      await configService.deleteProfile(profileId);
      await loadProfiles();
    } catch (error) {
      console.error("Failed to delete profile:", error);
      throw error;
    }
  };

  const handleEditProfile = (profile: ConfigProfile) => {
    setEditingProfile(profile);
    setShowSettings(true);
    setTabValue(1); // Switch to settings tab
  };

  const handleNewProfile = () => {
    setEditingProfile(null);
    setShowSettings(true);
    setTabValue(1); // Switch to settings tab
  };

  const handleCancelEdit = () => {
    setShowSettings(false);
    setEditingProfile(null);
    setTabValue(0); // Switch back to profiles tab
  };

  return (
    <Paper sx={{ width: "100%", p: 2 }}>
      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          mb: 2,
        }}
      >
        <Typography variant="h5">Configuration Management</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={handleNewProfile}
        >
          New Profile
        </Button>
      </Box>

      <Box sx={{ borderBottom: 1, borderColor: "divider" }}>
        <Tabs
          value={tabValue}
          onChange={handleTabChange}
          aria-label="configuration tabs"
        >
          <Tab label="Profiles" />
          <Tab label="Settings" disabled={!showSettings} />
        </Tabs>
      </Box>

      <TabPanel value={tabValue} index={0}>
        <ProfileManager
          profiles={profiles}
          activeProfileId={activeProfileId}
          onLoadProfiles={loadProfiles}
          onActivateProfile={handleActivateProfile}
          onSaveProfile={handleSaveProfile}
          onDeleteProfile={handleDeleteProfile}
          onEditProfile={handleEditProfile}
        />
      </TabPanel>

      <TabPanel value={tabValue} index={1}>
        {showSettings && (
          <SDRSettings
            profile={editingProfile}
            onSave={(profile) => { void handleSaveProfile(profile); }}
            onCancel={handleCancelEdit}
          />
        )}
      </TabPanel>
    </Paper>
  );
};
