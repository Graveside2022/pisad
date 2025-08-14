import React, { useState, useEffect } from "react";
import {
  Box,
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Button,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  CircularProgress,
} from "@mui/material";
import {
  Edit as EditIcon,
  Delete as DeleteIcon,
  PlayArrow as ActivateIcon,
  Star as DefaultIcon,
} from "@mui/icons-material";

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

interface ProfileManagerProps {
  profiles: ConfigProfile[];
  activeProfileId?: string;
  onLoadProfiles: () => void;
  onActivateProfile: (profileId: string) => void;
  onSaveProfile: (profile: ConfigProfile) => void;
  onDeleteProfile: (profileId: string) => void;
  onEditProfile: (profile: ConfigProfile) => void;
}

export const ProfileManager: React.FC<ProfileManagerProps> = ({
  profiles,
  activeProfileId,
  onLoadProfiles,
  onActivateProfile,
  onDeleteProfile,
  onEditProfile,
}) => {
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [profileToDelete, setProfileToDelete] = useState<ConfigProfile | null>(
    null,
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    onLoadProfiles();
  }, [onLoadProfiles]);

  const handleActivate = async (profile: ConfigProfile) => {
    setLoading(true);
    setError(null);
    try {
      if (profile.id) {
        await onActivateProfile(profile.id);
      } else {
        throw new Error("Cannot activate profile without ID");
      }
      // Profile activated successfully
    } catch (err) {
      setError(`Failed to activate profile: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  const handleEdit = (profile: ConfigProfile) => {
    onEditProfile(profile);
  };

  const handleDeleteClick = (profile: ConfigProfile) => {
    setProfileToDelete(profile);
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = async () => {
    if (profileToDelete) {
      setLoading(true);
      setError(null);
      try {
        if (profileToDelete.id) {
          await onDeleteProfile(profileToDelete.id);
        } else {
          throw new Error("Cannot delete profile without ID");
        }
        setDeleteDialogOpen(false);
        setProfileToDelete(null);
        onLoadProfiles();
      } catch (err) {
        setError(`Failed to delete profile: ${err}`);
      } finally {
        setLoading(false);
      }
    }
  };

  const handleDeleteCancel = () => {
    setDeleteDialogOpen(false);
    setProfileToDelete(null);
  };

  const getFrequencyDisplay = (frequency: number): string => {
    if (frequency >= 1e9) {
      return `${(frequency / 1e9).toFixed(3)} GHz`;
    } else if (frequency >= 1e6) {
      return `${(frequency / 1e6).toFixed(3)} MHz`;
    } else {
      return `${(frequency / 1e3).toFixed(3)} kHz`;
    }
  };

  const getSampleRateDisplay = (sampleRate: number): string => {
    if (sampleRate >= 1e6) {
      return `${(sampleRate / 1e6).toFixed(1)} Msps`;
    } else {
      return `${(sampleRate / 1e3).toFixed(1)} ksps`;
    }
  };

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Configuration Profiles
      </Typography>

      {error && (
        <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {loading && (
        <Box sx={{ display: "flex", justifyContent: "center", p: 2 }}>
          <CircularProgress />
        </Box>
      )}

      <List>
        {profiles.map((profile) => (
          <ListItem
            key={profile.id}
            sx={{
              border: 1,
              borderColor: "divider",
              borderRadius: 1,
              mb: 1,
              bgcolor:
                activeProfileId === profile.id
                  ? "action.selected"
                  : "background.paper",
            }}
          >
            <ListItemText
              primary={
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Typography variant="subtitle1">{profile.name}</Typography>
                  {profile.isDefault && (
                    <Chip
                      icon={<DefaultIcon />}
                      label="Default"
                      size="small"
                      color="primary"
                      variant="outlined"
                    />
                  )}
                  {activeProfileId === profile.id && (
                    <Chip label="Active" size="small" color="success" />
                  )}
                </Box>
              }
              secondary={
                <Box>
                  <Typography variant="body2" color="text.secondary">
                    {profile.description}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Frequency:{" "}
                    {getFrequencyDisplay(profile.sdrConfig.frequency)} | Sample
                    Rate: {getSampleRateDisplay(profile.sdrConfig.sampleRate)} |
                    Gain: {profile.sdrConfig.gain} dB
                  </Typography>
                </Box>
              }
            />
            <ListItemSecondaryAction>
              <IconButton
                edge="end"
                aria-label="activate"
                onClick={() => handleActivate(profile)}
                disabled={loading || activeProfileId === profile.id}
                color="primary"
              >
                <ActivateIcon />
              </IconButton>
              <IconButton
                edge="end"
                aria-label="edit"
                onClick={() => handleEdit(profile)}
                disabled={loading}
              >
                <EditIcon />
              </IconButton>
              <IconButton
                edge="end"
                aria-label="delete"
                onClick={() => handleDeleteClick(profile)}
                disabled={loading || profile.isDefault}
                color="error"
              >
                <DeleteIcon />
              </IconButton>
            </ListItemSecondaryAction>
          </ListItem>
        ))}
      </List>

      <Dialog open={deleteDialogOpen} onClose={handleDeleteCancel}>
        <DialogTitle>Delete Profile</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete the profile "{profileToDelete?.name}
            "? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDeleteCancel}>Cancel</Button>
          <Button
            onClick={handleDeleteConfirm}
            color="error"
            variant="contained"
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Paper>
  );
};
