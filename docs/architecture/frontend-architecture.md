# Frontend Architecture

## Component Architecture

### Component Organization
```
src/
├── components/
│   ├── common/
│   │   ├── SafetyToggle.tsx
│   │   ├── StatusBadge.tsx
│   │   └── AlertBanner.tsx
│   ├── dashboard/
│   │   ├── SignalMeter.tsx
│   │   ├── RSSIGraph.tsx
│   │   ├── DetectionLog.tsx
│   │   └── Dashboard.tsx
│   ├── homing/
│   │   ├── HomingControl.tsx
│   │   ├── SafetyInterlocks.tsx
│   │   └── VelocityVisualizer.tsx
│   └── config/
│       ├── ProfileManager.tsx
│       ├── SDRSettings.tsx
│       └── ConfigPanel.tsx
├── hooks/
│   ├── useWebSocket.ts
│   ├── useSystemState.ts
│   └── useRSSIData.ts
├── services/
│   ├── api.ts
│   ├── websocket.ts
│   └── config.ts
└── types/
    └── index.ts
```

### Component Template
```typescript
// Example: SignalMeter.tsx
import React, { memo } from 'react';
import { Box, Typography, LinearProgress } from '@mui/material';
import { SignalCellularAlt } from '@mui/icons-material';

interface SignalMeterProps {
  rssi: number;
  noiseFloor: number;
  snr: number;
  size?: 'small' | 'medium' | 'large';
}

export const SignalMeter = memo<SignalMeterProps>(({ 
  rssi, 
  noiseFloor, 
  snr, 
  size = 'medium' 
}) => {
  const signalPercent = Math.max(0, Math.min(100, (rssi + 100) * 2));
  const color = snr > 12 ? 'success' : snr > 6 ? 'warning' : 'error';
  
  return (
    <Box sx={{ 
      p: 2, 
      border: 1, 
      borderColor: `${color}.main`,
      borderRadius: 1 
    }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <SignalCellularAlt color={color} />
        <Typography variant="h4" sx={{ fontFamily: 'Roboto Mono' }}>
          {rssi.toFixed(1)} dBm
        </Typography>
      </Box>
      <LinearProgress 
        variant="determinate" 
        value={signalPercent} 
        color={color}
        sx={{ mt: 1, height: 8 }}
      />
      <Typography variant="caption" sx={{ color: 'text.secondary' }}>
        SNR: {snr.toFixed(1)} dB | Noise: {noiseFloor.toFixed(1)} dBm
      </Typography>
    </Box>
  );
});
```

## State Management Architecture

### State Structure
```typescript
// Global application state
interface AppState {
  system: SystemState;
  rssi: {
    current: RSSIReading;
    history: RSSIReading[];
    isStreaming: boolean;
  };
  detections: SignalDetection[];
  config: {
    profiles: ConfigProfile[];
    activeProfile: ConfigProfile | null;
  };
  ui: {
    homingConfirmation: boolean;
    alertQueue: Alert[];
    connectionStatus: 'connected' | 'connecting' | 'disconnected';
  };
}

// Context provider
export const AppContext = React.createContext<{
  state: AppState;
  dispatch: React.Dispatch<Action>;
}>({} as any);
```

### State Management Patterns
- Use React Context for global state (system status, active profile)
- Use local component state for UI-only concerns (form inputs, modals)
- Use useReducer for complex state updates (homing activation flow)
- Implement optimistic updates with rollback for better UX
- Memoize expensive computations (gradient calculations)

## Routing Architecture

### Route Organization
```
/                    # Dashboard (default)
/homing             # Homing control panel
/config             # Configuration management
/config/profiles    # Profile management
/config/sdr        # SDR settings
/missions          # Mission history
/missions/:id      # Mission detail/replay
```

### Protected Route Pattern
```typescript
// No authentication needed - all routes accessible locally
// But enforce safety checks for certain operations
import { Navigate, Outlet } from 'react-router-dom';
import { useSystemState } from '../hooks/useSystemState';

export const SafetyRoute: React.FC = () => {
  const { isConnected, isSafeMode } = useSystemState();
  
  if (!isConnected) {
    return <Navigate to="/" replace />;
  }
  
  return (
    <>
      {isSafeMode && (
        <Alert severity="warning">
          System in safe mode - some functions disabled
        </Alert>
      )}
      <Outlet />
    </>
  );
};
```

## Frontend Services Layer

### API Client Setup
```typescript
// services/api.ts
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8080/api';

export const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 5000,
  headers: {
    'Content-Type': 'application/json',
  }
});

// Response interceptor for error handling
apiClient.interceptors.response.use(
  response => response,
  error => {
    if (error.code === 'ECONNABORTED') {
      console.error('Request timeout');
    }
    return Promise.reject(error);
  }
);
```

### Service Example
```typescript
// services/config.service.ts
import { apiClient } from './api';
import { ConfigProfile } from '../types';

export const configService = {
  async getProfiles(): Promise<ConfigProfile[]> {
    const { data } = await apiClient.get('/config/profiles');
    return data;
  },
  
  async activateProfile(id: string): Promise<void> {
    await apiClient.post(`/config/profiles/${id}/activate`);
  },
  
  async saveProfile(profile: Partial<ConfigProfile>): Promise<ConfigProfile> {
    const { data } = await apiClient.post('/config/profiles', profile);
    return data;
  }
};
```
