# Testing Strategy

## Testing Pyramid
```
          E2E Tests
         /        \
    Integration Tests
       /            \
  Frontend Unit  Backend Unit
```

## Test Organization

### Frontend Tests
```
tests/frontend/
├── components/
│   ├── SignalMeter.test.tsx
│   ├── HomingControl.test.tsx
│   └── SafetyToggle.test.tsx
├── hooks/
│   └── useWebSocket.test.ts
└── services/
    └── api.test.ts
```

### Backend Tests
```
tests/backend/
├── unit/
│   ├── test_signal_processor.py
│   ├── test_state_machine.py
│   └── test_homing_controller.py
├── integration/
│   ├── test_sdr_service.py
│   ├── test_mavlink_service.py
│   └── test_api_routes.py
└── conftest.py
```

### E2E Tests
```
tests/e2e/
├── homing_activation.spec.ts
├── profile_management.spec.ts
└── signal_detection.spec.ts
```

## Test Examples

### Frontend Component Test
```typescript
// SignalMeter.test.tsx
import { render, screen } from '@testing-library/react';
import { SignalMeter } from '../src/components/dashboard/SignalMeter';

describe('SignalMeter', () => {
  it('displays correct RSSI value', () => {
    render(<SignalMeter rssi={-75} noiseFloor={-95} snr={20} />);
    expect(screen.getByText('-75.0 dBm')).toBeInTheDocument();
  });
  
  it('shows success color for high SNR', () => {
    const { container } = render(
      <SignalMeter rssi={-60} noiseFloor={-95} snr={35} />
    );
    expect(container.querySelector('[color="success"]')).toBeTruthy();
  });
});
```

### Backend API Test
```python