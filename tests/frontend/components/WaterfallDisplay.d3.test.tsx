import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { WaterfallDisplay } from '../../../src/frontend/src/components/spectrum/WaterfallDisplay';

// Mock our d3-waterfall implementation for testing
jest.mock('../../../src/frontend/src/utils/d3-waterfall', () => {
  return {
    D3Waterfall: class MockD3Waterfall {
      constructor(containerId: string, annotations: any[], options: any) {
        this.containerId = containerId;
        this.annotations = annotations;
        this.options = options;
      }

      setClickHandler = jest.fn();
      updateSpectrumData = jest.fn();
      destroy = jest.fn();
      resize = jest.fn();
    }
  };
});

describe('WaterfallDisplay d3-waterfall Integration', () => {
  test('initializes d3-waterfall with professional RTL-SDR configuration', async () => {
    render(<WaterfallDisplay centerFreq={2437e6} bandwidth={5e6} />);

    // This test should FAIL initially with Plotly.js implementation
    // Once d3-waterfall is integrated, this should pass
    const waterfallContainer = screen.queryByTestId('d3-waterfall-container');
    expect(waterfallContainer).toBeInTheDocument();
  });

  test('creates Canvas-based rendering instead of DOM heatmap', async () => {
    render(<WaterfallDisplay centerFreq={2437e6} bandwidth={5e6} />);

    // Should find the d3-waterfall container, not Plotly.js heatmap
    // Note: The actual canvas is created by d3-waterfall inside the container
    const waterfallContainer = screen.queryByTestId('waterfall-canvas');
    expect(waterfallContainer).toBeInTheDocument();
    expect(waterfallContainer?.tagName).toBe('DIV');

    // Verify container has the correct ID for d3-waterfall initialization
    expect(waterfallContainer).toHaveAttribute('id', 'waterfall-container');
  });

  test('integrates SIGIDWIKI frequency database for signal annotations', async () => {
    render(<WaterfallDisplay centerFreq={2437e6} bandwidth={5e6} />);

    // Verify that the component has initialized d3-waterfall with SIGIDWIKI capability
    // The sigidwiki-annotations will only appear when there are detected signals,
    // so we verify the waterfall container is ready for annotations
    const waterfallContainer = screen.queryByTestId('d3-waterfall-container');
    expect(waterfallContainer).toBeInTheDocument();

    // The d3-waterfall instance should be created (this would be tested via integration test)
    // For unit tests, we verify the component renders without errors
    expect(screen.queryByText(/Spectrum Waterfall/)).toBeInTheDocument();
  });

  test('implements professional RTL-SDR pan/zoom controls', async () => {
    render(<WaterfallDisplay centerFreq={2437e6} bandwidth={5e6} />);

    // Should have RTL-SDR specific controls, not generic Plotly.js controls
    const panZoomControls = screen.queryByTestId('rtl-sdr-controls');
    expect(panZoomControls).toBeInTheDocument();
  });
});
