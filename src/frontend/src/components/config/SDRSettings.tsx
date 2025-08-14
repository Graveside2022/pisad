import React, { useState, useEffect } from "react";
import {
  Box,
  Paper,
  Typography,
  TextField,
  Grid,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  InputAdornment,
  Alert,
  Divider,
  CircularProgress,
} from "@mui/material";
import { Save as SaveIcon, Cancel as CancelIcon } from "@mui/icons-material";

interface SDRConfig {
  frequency: number;
  sampleRate: number;
  gain: number | string;
  bandwidth: number;
}

interface SignalConfig {
  fftSize: number;
  ewmaAlpha: number;
  triggerThreshold: number;
  dropThreshold: number;
}

interface HomingConfig {
  forwardVelocityMax: number;
  yawRateMax: number;
  approachVelocity: number;
  signalLossTimeout: number;
}

interface ConfigProfile {
  id?: string;
  name: string;
  description: string;
  sdrConfig: SDRConfig;
  signalConfig: SignalConfig;
  homingConfig: HomingConfig;
  isDefault: boolean;
  createdAt?: string;
  updatedAt?: string;
}

interface SDRSettingsProps {
  profile: ConfigProfile | null;
  onSave: (profile: ConfigProfile) => void;
  onCancel: () => void;
}

export const SDRSettings: React.FC<SDRSettingsProps> = ({
  profile,
  onSave,
  onCancel,
}) => {
  const [editedProfile, setEditedProfile] = useState<ConfigProfile | null>(
    null,
  );
  const [errors, setErrors] = useState<string[]>([]);

  useEffect(() => {
    if (profile) {
      setEditedProfile({ ...profile });
    } else {
      // Create new profile with defaults
      setEditedProfile({
        name: "",
        description: "",
        sdrConfig: {
          frequency: 2437000000,
          sampleRate: 2000000,
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
      });
    }
  }, [profile]);

  const validateProfile = (): boolean => {
    const newErrors: string[] = [];

    if (!editedProfile) return false;

    // Validate name
    if (!editedProfile.name.trim()) {
      newErrors.push("Profile name is required");
    }

    // Validate SDR configuration
    if (
      editedProfile.sdrConfig.frequency < 1e6 ||
      editedProfile.sdrConfig.frequency > 6e9
    ) {
      newErrors.push("Frequency must be between 1 MHz and 6 GHz");
    }
    if (
      editedProfile.sdrConfig.sampleRate < 0.25e6 ||
      editedProfile.sdrConfig.sampleRate > 20e6
    ) {
      newErrors.push("Sample rate must be between 0.25 Msps and 20 Msps");
    }
    if (typeof editedProfile.sdrConfig.gain === "number") {
      if (
        editedProfile.sdrConfig.gain < -10 ||
        editedProfile.sdrConfig.gain > 70
      ) {
        newErrors.push("Gain must be between -10 dB and 70 dB");
      }
    }

    // Validate signal configuration
    if (
      editedProfile.signalConfig.ewmaAlpha <= 0 ||
      editedProfile.signalConfig.ewmaAlpha > 1
    ) {
      newErrors.push("EWMA alpha must be between 0 and 1");
    }
    if (
      editedProfile.signalConfig.dropThreshold >=
      editedProfile.signalConfig.triggerThreshold
    ) {
      newErrors.push("Drop threshold must be less than trigger threshold");
    }

    // Validate homing configuration
    if (editedProfile.homingConfig.forwardVelocityMax <= 0) {
      newErrors.push("Forward velocity max must be positive");
    }
    if (editedProfile.homingConfig.yawRateMax <= 0) {
      newErrors.push("Yaw rate max must be positive");
    }
    if (editedProfile.homingConfig.signalLossTimeout <= 0) {
      newErrors.push("Signal loss timeout must be positive");
    }

    setErrors(newErrors);
    return newErrors.length === 0;
  };

  const handleSave = () => {
    if (validateProfile() && editedProfile) {
      onSave(editedProfile);
    }
  };

  const handleChange = (
    section: keyof ConfigProfile,
    field: string,
    value: string | number | boolean,
  ) => {
    if (!editedProfile) return;

    setEditedProfile({
      ...editedProfile,
      [section]: {
        ...(editedProfile[section] as unknown as Record<string, unknown>),
        [field]: value,
      },
    });
  };

  const handleBasicChange = (
    field: keyof ConfigProfile,
    value: string | boolean,
  ) => {
    if (!editedProfile) return;

    setEditedProfile({
      ...editedProfile,
      [field]: value,
    });
  };

  if (!editedProfile) {
    return <CircularProgress />;
  }

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        {profile ? "Edit Configuration Profile" : "New Configuration Profile"}
      </Typography>

      {errors.length > 0 && (
        <Alert severity="error" sx={{ mb: 2 }}>
          <ul style={{ margin: 0, paddingLeft: 20 }}>
            {errors.map((error, index) => (
              <li key={index}>{error}</li>
            ))}
          </ul>
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Basic Information */}
        <Grid size={12}>
          <Typography variant="subtitle1" gutterBottom>
            Basic Information
          </Typography>
        </Grid>

        <Grid size={6}>
          <TextField
            fullWidth
            label="Profile Name"
            value={editedProfile.name}
            onChange={(e) => handleBasicChange("name", e.target.value)}
            required
          />
        </Grid>

        <Grid size={6}>
          <TextField
            fullWidth
            label="Description"
            value={editedProfile.description}
            onChange={(e) => handleBasicChange("description", e.target.value)}
          />
        </Grid>

        <Grid size={12}>
          <Divider sx={{ my: 2 }} />
        </Grid>

        {/* SDR Configuration */}
        <Grid size={12}>
          <Typography variant="subtitle1" gutterBottom>
            SDR Configuration
          </Typography>
        </Grid>

        <Grid size={6}>
          <TextField
            fullWidth
            label="Frequency"
            type="number"
            value={editedProfile.sdrConfig.frequency}
            onChange={(e) =>
              handleChange("sdrConfig", "frequency", parseFloat(e.target.value))
            }
            InputProps={{
              endAdornment: <InputAdornment position="end">Hz</InputAdornment>,
            }}
            helperText="1 MHz - 6 GHz"
          />
        </Grid>

        <Grid size={6}>
          <TextField
            fullWidth
            label="Sample Rate"
            type="number"
            value={editedProfile.sdrConfig.sampleRate}
            onChange={(e) =>
              handleChange(
                "sdrConfig",
                "sampleRate",
                parseFloat(e.target.value),
              )
            }
            InputProps={{
              endAdornment: <InputAdornment position="end">sps</InputAdornment>,
            }}
            helperText="0.25 - 20 Msps"
          />
        </Grid>

        <Grid size={6}>
          <FormControl fullWidth>
            <InputLabel>Gain</InputLabel>
            <Select
              value={editedProfile.sdrConfig.gain}
              onChange={(e) =>
                handleChange(
                  "sdrConfig",
                  "gain",
                  e.target.value === "AUTO"
                    ? "AUTO"
                    : parseFloat(e.target.value as string),
                )
              }
              label="Gain"
            >
              <MenuItem value="AUTO">AUTO</MenuItem>
              <MenuItem value={0}>0 dB</MenuItem>
              <MenuItem value={10}>10 dB</MenuItem>
              <MenuItem value={20}>20 dB</MenuItem>
              <MenuItem value={30}>30 dB</MenuItem>
              <MenuItem value={40}>40 dB</MenuItem>
              <MenuItem value={50}>50 dB</MenuItem>
              <MenuItem value={60}>60 dB</MenuItem>
            </Select>
          </FormControl>
        </Grid>

        <Grid size={6}>
          <TextField
            fullWidth
            label="Bandwidth"
            type="number"
            value={editedProfile.sdrConfig.bandwidth}
            onChange={(e) =>
              handleChange("sdrConfig", "bandwidth", parseFloat(e.target.value))
            }
            InputProps={{
              endAdornment: <InputAdornment position="end">Hz</InputAdornment>,
            }}
          />
        </Grid>

        <Grid size={12}>
          <Divider sx={{ my: 2 }} />
        </Grid>

        {/* Signal Processing Configuration */}
        <Grid size={12}>
          <Typography variant="subtitle1" gutterBottom>
            Signal Processing Configuration
          </Typography>
        </Grid>

        <Grid size={6}>
          <FormControl fullWidth>
            <InputLabel>FFT Size</InputLabel>
            <Select
              value={editedProfile.signalConfig.fftSize}
              onChange={(e) =>
                handleChange("signalConfig", "fftSize", e.target.value)
              }
              label="FFT Size"
            >
              <MenuItem value={256}>256</MenuItem>
              <MenuItem value={512}>512</MenuItem>
              <MenuItem value={1024}>1024</MenuItem>
              <MenuItem value={2048}>2048</MenuItem>
              <MenuItem value={4096}>4096</MenuItem>
            </Select>
          </FormControl>
        </Grid>

        <Grid size={6}>
          <Box>
            <Typography gutterBottom>
              EWMA Alpha: {editedProfile.signalConfig.ewmaAlpha}
            </Typography>
            <Slider
              value={editedProfile.signalConfig.ewmaAlpha}
              onChange={(_e, value) =>
                handleChange("signalConfig", "ewmaAlpha", value)
              }
              min={0.01}
              max={1}
              step={0.01}
              valueLabelDisplay="auto"
            />
          </Box>
        </Grid>

        <Grid size={6}>
          <TextField
            fullWidth
            label="Trigger Threshold"
            type="number"
            value={editedProfile.signalConfig.triggerThreshold}
            onChange={(e) =>
              handleChange(
                "signalConfig",
                "triggerThreshold",
                parseFloat(e.target.value),
              )
            }
            InputProps={{
              endAdornment: <InputAdornment position="end">dBm</InputAdornment>,
            }}
          />
        </Grid>

        <Grid size={6}>
          <TextField
            fullWidth
            label="Drop Threshold"
            type="number"
            value={editedProfile.signalConfig.dropThreshold}
            onChange={(e) =>
              handleChange(
                "signalConfig",
                "dropThreshold",
                parseFloat(e.target.value),
              )
            }
            InputProps={{
              endAdornment: <InputAdornment position="end">dBm</InputAdornment>,
            }}
          />
        </Grid>

        <Grid size={12}>
          <Divider sx={{ my: 2 }} />
        </Grid>

        {/* Homing Configuration */}
        <Grid size={12}>
          <Typography variant="subtitle1" gutterBottom>
            Homing Configuration
          </Typography>
        </Grid>

        <Grid size={6}>
          <TextField
            fullWidth
            label="Max Forward Velocity"
            type="number"
            value={editedProfile.homingConfig.forwardVelocityMax}
            onChange={(e) =>
              handleChange(
                "homingConfig",
                "forwardVelocityMax",
                parseFloat(e.target.value),
              )
            }
            InputProps={{
              endAdornment: <InputAdornment position="end">m/s</InputAdornment>,
            }}
          />
        </Grid>

        <Grid size={6}>
          <TextField
            fullWidth
            label="Max Yaw Rate"
            type="number"
            value={editedProfile.homingConfig.yawRateMax}
            onChange={(e) =>
              handleChange(
                "homingConfig",
                "yawRateMax",
                parseFloat(e.target.value),
              )
            }
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">rad/s</InputAdornment>
              ),
            }}
          />
        </Grid>

        <Grid size={6}>
          <TextField
            fullWidth
            label="Approach Velocity"
            type="number"
            value={editedProfile.homingConfig.approachVelocity}
            onChange={(e) =>
              handleChange(
                "homingConfig",
                "approachVelocity",
                parseFloat(e.target.value),
              )
            }
            InputProps={{
              endAdornment: <InputAdornment position="end">m/s</InputAdornment>,
            }}
          />
        </Grid>

        <Grid size={6}>
          <TextField
            fullWidth
            label="Signal Loss Timeout"
            type="number"
            value={editedProfile.homingConfig.signalLossTimeout}
            onChange={(e) =>
              handleChange(
                "homingConfig",
                "signalLossTimeout",
                parseFloat(e.target.value),
              )
            }
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">seconds</InputAdornment>
              ),
            }}
          />
        </Grid>

        <Grid size={12}>
          <Box
            sx={{ display: "flex", gap: 2, justifyContent: "flex-end", mt: 3 }}
          >
            <Button
              variant="outlined"
              startIcon={<CancelIcon />}
              onClick={onCancel}
            >
              Cancel
            </Button>
            <Button
              variant="contained"
              startIcon={<SaveIcon />}
              onClick={handleSave}
            >
              Save Profile
            </Button>
          </Box>
        </Grid>
      </Grid>
    </Paper>
  );
};
