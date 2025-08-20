import * as d3 from 'd3';

export interface SpectrumData {
  frequencies: Float32Array;
  magnitudes: Float32Array;
  timestamp: number;
  centerFreq: number;
  sampleRate: number;
}

export interface RTLSDRRow {
  timestamp: number;
  frequencies: number[];
  magnitudes: number[];
  confidence?: number;
}

export interface SignalAnnotation {
  freqStart: number;
  freqStop: number;
  description: string;
  url: string;
  signalType?: string;
  // SIGIDWIKI specific fields
  sigidwiki?: {
    id: string;
    modulation: string;
    bandwidth: number;
    confidence: number; // 0-1 confidence score
    location?: string;
    usage?: string;
    source: 'sigidwiki' | 'user' | 'detected';
    lastUpdated: number;
  };
}

export interface WaterfallOptions {
  isAnimatable: boolean;
  isSelectable: boolean;
  isZoomable: boolean;
  margin?: number;
  maxRows?: number;
  // RTL-SDR specific options
  sampleRate?: number;
  gainSettings?: number;
  centerFreq?: number;
  bandwidth?: number;
  deviceMetadata?: RTLSDRDeviceInfo;
}

export interface RTLSDRDeviceInfo {
  deviceName: string;
  serialNumber?: string;
  hardwareVersion?: string;
  firmwareVersion?: string;
  sampleRate: number;
  gainSettings: number;
  centerFreq: number;
  bandwidth: number;
  frequencyRange: {
    min: number; // 850MHz minimum
    max: number; // 6.5GHz maximum
  };
  supportedBandwidths: number[]; // [2.5MHz, 5MHz, 8MHz, 10MHz]
  gainRange: {
    min: number;
    max: number;
    step: number;
  };
}

// RTL-SDR Native Feature Classes
class FrequencyValidator {
  private static readonly MIN_FREQ = 850e6; // 850 MHz
  private static readonly MAX_FREQ = 6500e6; // 6.5 GHz
  private static readonly HACKRF_MIN = 1e6; // 1 MHz
  private static readonly HACKRF_MAX = 6000e6; // 6 GHz

  static validateFrequency(freq: number, deviceName: string = 'generic'): boolean {
    if (deviceName.toLowerCase().includes('hackrf')) {
      return freq >= this.HACKRF_MIN && freq <= this.HACKRF_MAX;
    }
    return freq >= this.MIN_FREQ && freq <= this.MAX_FREQ;
  }

  static getValidFrequencyRange(deviceName: string = 'generic'): { min: number; max: number } {
    if (deviceName.toLowerCase().includes('hackrf')) {
      return { min: this.HACKRF_MIN, max: this.HACKRF_MAX };
    }
    return { min: this.MIN_FREQ, max: this.MAX_FREQ };
  }

  static formatFrequency(freq: number): string {
    if (freq >= 1e9) {
      return `${(freq / 1e9).toFixed(3)} GHz`;
    } else if (freq >= 1e6) {
      return `${(freq / 1e6).toFixed(1)} MHz`;
    } else if (freq >= 1e3) {
      return `${(freq / 1e3).toFixed(1)} kHz`;
    }
    return `${freq.toFixed(0)} Hz`;
  }
}

class BandwidthOptimizer {
  private static readonly OPTIMAL_5MHZ_SETTINGS = {
    sampleRate: 5e6,
    binCount: 2048,
    updateRate: 3, // 3Hz for smooth real-time
    colorRange: [-120, -20] // dBm for RTL-SDR
  };

  static optimizeFor5MHz(spectrumData: SpectrumData): SpectrumData {
    const { frequencies, magnitudes, timestamp, centerFreq, sampleRate } = spectrumData;

    // Ensure 5MHz bandwidth is optimally displayed
    if (Math.abs(sampleRate - 5e6) < 1e5) { // Within 100kHz of 5MHz
      // Apply RTL-SDR specific optimizations
      const optimizedMagnitudes = new Float32Array(magnitudes.length);

      for (let i = 0; i < magnitudes.length; i++) {
        // Apply RTL-SDR calibration and noise floor adjustment
        optimizedMagnitudes[i] = magnitudes[i] + this.getRTLSDRCalibration(frequencies[i]);
      }

      return {
        frequencies,
        magnitudes: optimizedMagnitudes,
        timestamp,
        centerFreq,
        sampleRate
      };
    }

    return spectrumData;
  }

  private static getRTLSDRCalibration(frequency: number): number {
    // RTL-SDR frequency-dependent gain correction
    // Based on typical RTL-SDR dongles frequency response
    if (frequency < 100e6) return -3; // VHF rolloff
    if (frequency > 1700e6) return -2; // UHF rolloff
    return 0; // Flat response in core range
  }

  static getOptimalBinCount(bandwidth: number): number {
    // Calculate optimal FFT bin count for bandwidth
    if (bandwidth <= 2.5e6) return 1024;
    if (bandwidth <= 5e6) return 2048;
    if (bandwidth <= 10e6) return 4096;
    return 8192;
  }
}

class RTLSDRDeviceManager {
  static createHackRFDeviceInfo(): RTLSDRDeviceInfo {
    return {
      deviceName: 'HackRF One',
      serialNumber: 'auto-detected',
      hardwareVersion: '1.0',
      firmwareVersion: '2023.01.1',
      sampleRate: 5e6,
      gainSettings: 20, // Default 20dB
      centerFreq: 2437e6, // Default 2.437 GHz
      bandwidth: 5e6,
      frequencyRange: {
        min: 1e6, // 1 MHz
        max: 6000e6 // 6 GHz
      },
      supportedBandwidths: [2.5e6, 5e6, 8e6, 10e6, 20e6],
      gainRange: {
        min: 0,
        max: 62,
        step: 2
      }
    };
  }

  static createGenericRTLSDRInfo(): RTLSDRDeviceInfo {
    return {
      deviceName: 'Generic RTL-SDR',
      sampleRate: 2.5e6,
      gainSettings: 20,
      centerFreq: 2437e6,
      bandwidth: 2.5e6,
      frequencyRange: {
        min: 850e6, // 850 MHz
        max: 6500e6 // 6.5 GHz
      },
      supportedBandwidths: [1e6, 2.5e6, 5e6],
      gainRange: {
        min: 0,
        max: 49.6,
        step: 0.1
      }
    };
  }
}

// SIGIDWIKI Integration Classes
class SigidWikiIntegration {
  private static knownSignals: SignalAnnotation[] = [
    // ISM Band signals (2.4 GHz)
    {
      freqStart: 2400e6,
      freqStop: 2450e6,
      description: 'WiFi 802.11b/g/n Channel 1-11',
      url: 'https://www.sigidwiki.com/wiki/802.11_(WiFi)',
      signalType: 'Digital',
      sigidwiki: {
        id: 'wifi-2.4ghz',
        modulation: 'OFDM/DSSS',
        bandwidth: 20e6,
        confidence: 0.95,
        location: 'Global',
        usage: 'Wireless LAN',
        source: 'sigidwiki',
        lastUpdated: Date.now()
      }
    },
    {
      freqStart: 2437e6,
      freqStop: 2437e6 + 22e6,
      description: 'WiFi Channel 6 (2.437 GHz)',
      url: 'https://www.sigidwiki.com/wiki/802.11_(WiFi)',
      signalType: 'Digital',
      sigidwiki: {
        id: 'wifi-ch6',
        modulation: 'OFDM',
        bandwidth: 22e6,
        confidence: 0.98,
        location: 'Global',
        usage: 'WiFi Channel 6',
        source: 'sigidwiki',
        lastUpdated: Date.now()
      }
    },
    {
      freqStart: 2400e6,
      freqStop: 2485e6,
      description: 'Bluetooth/BLE',
      url: 'https://www.sigidwiki.com/wiki/Bluetooth',
      signalType: 'Digital',
      sigidwiki: {
        id: 'bluetooth',
        modulation: 'FHSS',
        bandwidth: 1e6,
        confidence: 0.90,
        location: 'Global',
        usage: 'Personal Area Network',
        source: 'sigidwiki',
        lastUpdated: Date.now()
      }
    }
  ];

  static findSignalsInRange(startFreq: number, stopFreq: number): SignalAnnotation[] {
    return this.knownSignals.filter(signal => {
      // Check for frequency overlap
      return !(signal.freqStop < startFreq || signal.freqStart > stopFreq);
    });
  }

  static findExactSignal(frequency: number, tolerance: number = 1e6): SignalAnnotation | null {
    for (const signal of this.knownSignals) {
      if (frequency >= (signal.freqStart - tolerance) && frequency <= (signal.freqStop + tolerance)) {
        return signal;
      }
    }
    return null;
  }

  static createUserAnnotation(frequency: number, description: string): SignalAnnotation {
    return {
      freqStart: frequency - 100e3, // ±100kHz default bandwidth
      freqStop: frequency + 100e3,
      description: description,
      url: `#user-annotation-${Date.now()}`,
      signalType: 'User Defined',
      sigidwiki: {
        id: `user-${Date.now()}`,
        modulation: 'Unknown',
        bandwidth: 200e3,
        confidence: 0.5,
        usage: 'User annotation',
        source: 'user',
        lastUpdated: Date.now()
      }
    };
  }

  static getSignalColor(annotation: SignalAnnotation): string {
    if (!annotation.sigidwiki) return '#cccccc';

    switch (annotation.sigidwiki.source) {
      case 'sigidwiki': return '#4CAF50'; // Green for verified signals
      case 'detected': return '#FF9800'; // Orange for auto-detected
      case 'user': return '#2196F3'; // Blue for user annotations
      default: return '#cccccc';
    }
  }
}

class SignalDetector {
  private static readonly SIGNAL_THRESHOLD = -70; // dBm threshold for signal detection
  private static readonly MIN_BANDWIDTH = 10e3; // 10kHz minimum signal bandwidth

  static detectSignals(spectrumData: SpectrumData): SignalAnnotation[] {
    const { frequencies, magnitudes } = spectrumData;
    const detectedSignals: SignalAnnotation[] = [];

    let signalStart = -1;
    let signalPeak = -1;
    let peakMagnitude = -Infinity;

    for (let i = 0; i < magnitudes.length; i++) {
      const magnitude = magnitudes[i];

      if (magnitude > this.SIGNAL_THRESHOLD) {
        // Signal detected
        if (signalStart === -1) {
          signalStart = i; // Start of new signal
          signalPeak = i;
          peakMagnitude = magnitude;
        } else {
          // Continue existing signal, check for new peak
          if (magnitude > peakMagnitude) {
            signalPeak = i;
            peakMagnitude = magnitude;
          }
        }
      } else {
        // End of signal
        if (signalStart !== -1) {
          const signalBandwidth = frequencies[i - 1] - frequencies[signalStart];

          if (signalBandwidth >= this.MIN_BANDWIDTH) {
            // Create detected signal annotation
            const annotation: SignalAnnotation = {
              freqStart: frequencies[signalStart],
              freqStop: frequencies[i - 1],
              description: `Detected Signal (${FrequencyValidator.formatFrequency(frequencies[signalPeak])})`,
              url: '#auto-detected',
              signalType: 'Auto-Detected',
              sigidwiki: {
                id: `detected-${frequencies[signalPeak]}`,
                modulation: 'Unknown',
                bandwidth: signalBandwidth,
                confidence: Math.min(1.0, (peakMagnitude - this.SIGNAL_THRESHOLD) / 30), // 0-1 based on SNR
                usage: 'Auto-detected signal',
                source: 'detected',
                lastUpdated: Date.now()
              }
            };

            detectedSignals.push(annotation);
          }

          // Reset for next signal
          signalStart = -1;
          signalPeak = -1;
          peakMagnitude = -Infinity;
        }
      }
    }

    return detectedSignals;
  }
}

export class D3Waterfall {
  private container: d3.Selection<HTMLElement, unknown, HTMLElement, any>;
  private canvas: d3.Selection<HTMLCanvasElement, unknown, HTMLElement, any>;
  private svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, any>;
  private svgGroup: d3.Selection<SVGGElement, unknown, HTMLElement, any>;
  private context: CanvasRenderingContext2D;
  private width: number = 0;
  private height: number = 0;
  private margin: number;
  private maxRows: number;
  private waterfallData: RTLSDRRow[] = [];
  private annotations: SignalAnnotation[] = [];
  private xScale: d3.ScaleLinear<number, number, never> | null = null;
  private yScale: d3.ScaleLinear<number, number, never> | null = null;
  private colorScale: d3.ScaleSequential<string, never> | null = null;
  private zoom: d3.ZoomBehavior<Element, unknown> | null = null;
  private animationId: number | null = null;
  private options: WaterfallOptions;
  private clickHandler: ((frequency: number, confidence: number) => void) | null = null;
  private imageBitmap: ImageBitmap | null = null;

  // RTL-SDR specific properties
  private deviceInfo: RTLSDRDeviceInfo | null = null;
  private frequencyValidator: FrequencyValidator;
  private bandwidthOptimizer: BandwidthOptimizer;

  // SIGIDWIKI Integration properties
  private sigidwikiIntegration: SigidWikiIntegration;
  private signalDetector: SignalDetector;
  private autoDetectSignals: boolean = true;
  private detectedSignals: SignalAnnotation[] = [];

  constructor(
    containerId: string,
    annotations: SignalAnnotation[] = [],
    options: WaterfallOptions
  ) {
    this.options = options;
    this.margin = options.margin || 50;
    this.maxRows = options.maxRows || 100;
    this.annotations = annotations;

    // Initialize RTL-SDR native features
    this.frequencyValidator = FrequencyValidator;
    this.bandwidthOptimizer = BandwidthOptimizer;

    // Setup device info from options or create default HackRF
    this.deviceInfo = options.deviceMetadata || RTLSDRDeviceManager.createHackRFDeviceInfo();

    // Initialize SIGIDWIKI integration
    this.sigidwikiIntegration = SigidWikiIntegration;
    this.signalDetector = SignalDetector;

    // Initialize D3 selections
    this.container = d3.select(containerId);
    if (this.container.empty()) {
      throw new Error(`Container ${containerId} not found`);
    }

    // Create canvas for waterfall rendering
    this.canvas = this.container
      .append('canvas')
      .attr('data-testid', 'waterfall-canvas')
      .style('position', 'absolute')
      .style('pointer-events', 'none');

    // Create SVG overlay for axes and annotations
    this.svg = this.container
      .append('svg')
      .attr('data-testid', 'waterfall-svg')
      .style('position', 'absolute')
      .style('pointer-events', 'all');

    this.svgGroup = this.svg.append('g');

    // Get canvas context
    const canvasNode = this.canvas.node();
    if (!canvasNode) {
      throw new Error('Failed to create canvas element');
    }

    const context = canvasNode.getContext('2d');
    if (!context) {
      throw new Error('Failed to get 2D canvas context');
    }
    this.context = context;

    // Initialize scales
    this.initializeScales();

    // Setup zoom behavior if enabled
    if (this.options.isZoomable) {
      this.setupZoom();
    }

    // Setup click handlers
    this.setupClickHandlers();

    // Resize to fit container
    this.resize();
  }

  private initializeScales(): void {
    this.xScale = d3.scaleLinear();
    this.yScale = d3.scaleLinear();
    this.colorScale = d3.scaleSequential(d3.interpolateViridis);
  }

  private setupZoom(): void {
    if (!this.options.isZoomable) return;

    this.zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 10])
      .on('zoom', (event) => {
        this.handleZoom(event.transform);
      });

    this.svg.call(this.zoom);
  }

  private setupClickHandlers(): void {
    this.svg.on('click', (event) => {
      if (!this.clickHandler || !this.xScale) return;

      const [mouseX] = d3.pointer(event);
      const frequency = this.xScale.invert(mouseX - this.margin);
      const confidence = 1.0; // Default confidence, will be enhanced with ASV integration

      this.clickHandler(frequency, confidence);
    });
  }

  private handleZoom(transform: d3.ZoomTransform): void {
    if (!this.xScale || !this.yScale) return;

    // Update axes with zoom transform
    const newXScale = transform.rescaleX(this.xScale);
    const newYScale = transform.rescaleY(this.yScale);

    // Redraw axes
    this.drawAxes(newXScale, newYScale);

    // Redraw waterfall with zoom transform
    this.redrawWithTransform(transform);
  }

  private redrawWithTransform(transform: d3.ZoomTransform): void {
    if (!this.imageBitmap) return;

    this.context.clearRect(0, 0, this.width, this.height);
    this.context.save();
    this.context.translate(transform.x, transform.y);
    this.context.scale(transform.k, transform.k);
    this.context.drawImage(this.imageBitmap, this.margin, this.margin);
    this.context.restore();
  }

  public resize(): void {
    const containerNode = this.container.node() as HTMLElement;
    if (!containerNode) return;

    this.width = containerNode.clientWidth || 800;
    this.height = containerNode.clientHeight || 400;

    // Update canvas size
    this.canvas
      .attr('width', this.width)
      .attr('height', this.height);

    // Update SVG size
    this.svg
      .attr('width', this.width)
      .attr('height', this.height);

    // Update SVG group position
    this.svgGroup.attr('transform', `translate(${this.margin}, ${this.margin})`);

    // Update scales range
    if (this.xScale && this.yScale) {
      this.xScale.range([0, this.width - 2 * this.margin]);
      this.yScale.range([0, this.height - 2 * this.margin]);
    }
  }

  public updateSpectrumData(data: SpectrumData): void {
    // Apply RTL-SDR native optimizations
    const optimizedData = this.deviceInfo && this.deviceInfo.bandwidth === 5e6
      ? BandwidthOptimizer.optimizeFor5MHz(data)
      : data;

    // Validate frequency range for current device
    const deviceName = this.deviceInfo?.deviceName || 'generic';
    const isValidFreq = FrequencyValidator.validateFrequency(optimizedData.centerFreq, deviceName);

    if (!isValidFreq) {
      console.warn(`Frequency ${FrequencyValidator.formatFrequency(optimizedData.centerFreq)} out of range for ${deviceName}`);
      return;
    }

    // Convert SpectrumData to RTLSDRRow format with RTL-SDR metadata
    const row: RTLSDRRow = {
      timestamp: optimizedData.timestamp,
      frequencies: Array.from(optimizedData.frequencies),
      magnitudes: Array.from(optimizedData.magnitudes),
      confidence: this.calculateRTLSDRConfidence(optimizedData)
    };

    // Add to waterfall data
    this.waterfallData.push(row);

    // Limit to max rows for performance
    if (this.waterfallData.length > this.maxRows) {
      this.waterfallData.shift();
    }

    // SIGIDWIKI Integration: Auto-detect and annotate signals
    if (this.autoDetectSignals) {
      const newlyDetected = SignalDetector.detectSignals(optimizedData);
      this.detectedSignals = [...this.detectedSignals, ...newlyDetected];

      // Merge with SIGIDWIKI known signals in current frequency range
      const freqStart = Math.min(...Array.from(optimizedData.frequencies));
      const freqStop = Math.max(...Array.from(optimizedData.frequencies));
      const knownSignals = SigidWikiIntegration.findSignalsInRange(freqStart, freqStop);

      // Update annotations with both detected and known signals
      this.annotations = [...this.annotations, ...knownSignals, ...newlyDetected];

      // Remove duplicates based on frequency overlap
      this.annotations = this.removeDuplicateAnnotations(this.annotations);
    }

    // Update scales domain
    this.updateScalesDomain();

    // Render waterfall with annotations
    this.render();
  }

  private calculateRTLSDRConfidence(data: SpectrumData): number {
    // Calculate signal confidence based on RTL-SDR characteristics
    const avgMagnitude = data.magnitudes.reduce((sum, mag) => sum + mag, 0) / data.magnitudes.length;
    const noiseFloor = -100; // Typical RTL-SDR noise floor in dBm

    // Higher confidence for signals well above noise floor
    const snr = avgMagnitude - noiseFloor;
    return Math.min(1.0, Math.max(0.1, snr / 40)); // 0.1 to 1.0 confidence range
  }

  private removeDuplicateAnnotations(annotations: SignalAnnotation[]): SignalAnnotation[] {
    const uniqueAnnotations: SignalAnnotation[] = [];

    for (const annotation of annotations) {
      // Check if this annotation overlaps with any existing one
      const hasOverlap = uniqueAnnotations.some(existing => {
        return !(annotation.freqStop < existing.freqStart || annotation.freqStart > existing.freqStop);
      });

      if (!hasOverlap) {
        uniqueAnnotations.push(annotation);
      }
    }

    return uniqueAnnotations;
  }

  private updateScalesDomain(): void {
    if (!this.xScale || !this.yScale || !this.colorScale) return;
    if (this.waterfallData.length === 0) return;

    // Calculate frequency domain
    const allFreqs = this.waterfallData[0].frequencies;
    const freqExtent = d3.extent(allFreqs) as [number, number];
    this.xScale.domain(freqExtent);

    // Calculate time domain
    const timeExtent = d3.extent(this.waterfallData, d => d.timestamp) as [number, number];
    this.yScale.domain(timeExtent);

    // Calculate magnitude domain for color scale
    const allMagnitudes = this.waterfallData.flatMap(row => row.magnitudes);
    const magExtent = d3.extent(allMagnitudes) as [number, number];
    this.colorScale.domain(magExtent);
  }

  private render(): void {
    if (!this.xScale || !this.yScale || !this.colorScale) return;

    if (this.options.isAnimatable && window.requestAnimationFrame) {
      this.renderAnimated();
    } else {
      this.renderStatic();
    }

    // Update axes
    this.drawAxes(this.xScale, this.yScale);

    // Draw SIGIDWIKI signal annotations
    this.drawSignalAnnotations();

    // Cache image for zoom performance
    if (this.options.isZoomable) {
      this.cacheCanvasImage();
    }
  }

  private renderAnimated(): void {
    let rowIndex = 0;
    const renderRow = () => {
      if (rowIndex < this.waterfallData.length) {
        this.drawRow(this.waterfallData[rowIndex]);
        rowIndex++;
        this.animationId = requestAnimationFrame(renderRow);
      }
    };

    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
    this.animationId = requestAnimationFrame(renderRow);
  }

  private renderStatic(): void {
    this.context.clearRect(0, 0, this.width, this.height);
    this.waterfallData.forEach(row => this.drawRow(row));
  }

  private drawRow(row: RTLSDRRow): void {
    if (!this.xScale || !this.yScale || !this.colorScale) return;

    const rowHeight = this.waterfallData.length > 1 ?
      Math.abs(this.yScale(this.waterfallData[1].timestamp) - this.yScale(this.waterfallData[0].timestamp)) :
      2;

    row.frequencies.forEach((freq, i) => {
      const magnitude = row.magnitudes[i];
      const x = this.xScale!(freq) + this.margin;
      const y = this.yScale!(row.timestamp) + this.margin;
      const width = this.xScale!(row.frequencies[1] || freq + 1000) - this.xScale!(freq);

      this.context.fillStyle = this.colorScale!(magnitude);
      this.context.fillRect(x, y, width, rowHeight);
    });
  }

  private drawAxes(xScale: d3.ScaleLinear<number, number, never>, yScale: d3.ScaleLinear<number, number, never>): void {
    // Clear existing axes
    this.svgGroup.selectAll('.axis').remove();

    // X-axis (frequency)
    const xAxis = d3.axisBottom(xScale)
      .ticks(10)
      .tickFormat((d) => `${(d as number / 1e6).toFixed(1)}M`);

    this.svgGroup.append('g')
      .attr('class', 'axis x-axis')
      .attr('transform', `translate(0, ${this.height - 2 * this.margin})`)
      .call(xAxis);

    // Y-axis (time)
    const yAxis = d3.axisLeft(yScale)
      .ticks(5)
      .tickFormat(d => new Date(d as number).toLocaleTimeString());

    this.svgGroup.append('g')
      .attr('class', 'axis y-axis')
      .call(yAxis);

    // Axis labels
    this.svgGroup.append('text')
      .attr('class', 'axis-label')
      .attr('text-anchor', 'middle')
      .attr('x', (this.width - 2 * this.margin) / 2)
      .attr('y', this.height - this.margin / 2)
      .text('Frequency');

    this.svgGroup.append('text')
      .attr('class', 'axis-label')
      .attr('text-anchor', 'middle')
      .attr('transform', `translate(${-this.margin / 2}, ${(this.height - 2 * this.margin) / 2}) rotate(-90)`)
      .text('Time');
  }

  private drawSignalAnnotations(): void {
    if (!this.xScale || this.annotations.length === 0) return;

    // Clear existing annotations
    this.svgGroup.selectAll('.signal-annotation').remove();

    const annotationGroup = this.svgGroup.append('g')
      .attr('class', 'signal-annotations')
      .attr('data-testid', 'sigidwiki-annotations');

    // Draw signal annotation rectangles
    this.annotations.forEach((annotation, index) => {
      if (!this.xScale) return;

      const x1 = this.xScale(annotation.freqStart);
      const x2 = this.xScale(annotation.freqStop);
      const width = x2 - x1;

      if (width < 1) return; // Skip very narrow signals

      const color = SigidWikiIntegration.getSignalColor(annotation);
      const opacity = annotation.sigidwiki?.confidence || 0.3;

      // Draw signal boundary rectangle
      const signalRect = annotationGroup.append('rect')
        .attr('class', 'signal-annotation')
        .attr('x', x1)
        .attr('y', 0)
        .attr('width', width)
        .attr('height', this.height - 2 * this.margin)
        .attr('fill', color)
        .attr('fill-opacity', 0.2)
        .attr('stroke', color)
        .attr('stroke-width', 2)
        .attr('stroke-opacity', opacity);

      // Add signal label for wider signals
      if (width > 50) {
        const labelText = annotation.sigidwiki?.modulation || annotation.signalType || 'Unknown';

        annotationGroup.append('text')
          .attr('class', 'signal-label')
          .attr('x', x1 + width / 2)
          .attr('y', 15)
          .attr('text-anchor', 'middle')
          .attr('font-size', '10px')
          .attr('font-weight', 'bold')
          .attr('fill', color)
          .text(labelText);
      }

      // Add tooltip functionality
      signalRect
        .on('mouseover', (event) => {
          this.showSignalTooltip(event, annotation);
        })
        .on('mouseout', () => {
          this.hideSignalTooltip();
        });
    });
  }

  private showSignalTooltip(event: MouseEvent, annotation: SignalAnnotation): void {
    // Create or update tooltip
    let tooltip = this.container.select('.signal-tooltip');

    if (tooltip.empty()) {
      tooltip = this.container.append('div')
        .attr('class', 'signal-tooltip')
        .style('position', 'absolute')
        .style('background', 'rgba(0, 0, 0, 0.8)')
        .style('color', 'white')
        .style('padding', '8px')
        .style('border-radius', '4px')
        .style('font-size', '12px')
        .style('pointer-events', 'none')
        .style('z-index', '1000');
    }

    const freqStart = FrequencyValidator.formatFrequency(annotation.freqStart);
    const freqStop = FrequencyValidator.formatFrequency(annotation.freqStop);
    const bandwidth = FrequencyValidator.formatFrequency(annotation.freqStop - annotation.freqStart);

    let tooltipContent = `<strong>${annotation.description}</strong><br/>`;
    tooltipContent += `Frequency: ${freqStart} - ${freqStop}<br/>`;
    tooltipContent += `Bandwidth: ${bandwidth}<br/>`;

    if (annotation.sigidwiki) {
      tooltipContent += `Modulation: ${annotation.sigidwiki.modulation}<br/>`;
      tooltipContent += `Confidence: ${(annotation.sigidwiki.confidence * 100).toFixed(1)}%<br/>`;
      tooltipContent += `Source: ${annotation.sigidwiki.source}`;
      if (annotation.sigidwiki.location) {
        tooltipContent += `<br/>Location: ${annotation.sigidwiki.location}`;
      }
    }

    tooltip
      .html(tooltipContent)
      .style('left', `${event.pageX + 10}px`)
      .style('top', `${event.pageY - 10}px`)
      .style('opacity', 1);
  }

  private hideSignalTooltip(): void {
    this.container.select('.signal-tooltip')
      .style('opacity', 0);
  }

  private cacheCanvasImage(): void {
    if (!this.options.isZoomable) return;

    // Create ImageBitmap for optimized zoom/pan performance
    const imageData = this.context.getImageData(0, 0, this.width, this.height);
    createImageBitmap(imageData).then(bitmap => {
      this.imageBitmap = bitmap;
    }).catch(error => {
      console.warn('Failed to create ImageBitmap:', error);
    });
  }

  public setClickHandler(handler: (frequency: number, confidence: number) => void): void {
    this.clickHandler = handler;
  }

  // RTL-SDR Native Feature Methods
  public getDeviceInfo(): RTLSDRDeviceInfo | null {
    return this.deviceInfo;
  }

  public updateDeviceInfo(deviceInfo: RTLSDRDeviceInfo): void {
    this.deviceInfo = deviceInfo;

    // Update waterfall display with new device capabilities
    if (this.deviceInfo) {
      this.options.sampleRate = deviceInfo.sampleRate;
      this.options.centerFreq = deviceInfo.centerFreq;
      this.options.bandwidth = deviceInfo.bandwidth;
      this.options.gainSettings = deviceInfo.gainSettings;

      // Re-render with new device settings
      this.render();
    }
  }

  public validateFrequencyForDevice(frequency: number): boolean {
    if (!this.deviceInfo) return true;
    return FrequencyValidator.validateFrequency(frequency, this.deviceInfo.deviceName);
  }

  public getFormattedDeviceInfo(): string {
    if (!this.deviceInfo) return 'No device connected';

    const { deviceName, centerFreq, bandwidth, gainSettings, sampleRate } = this.deviceInfo;
    const freqStr = FrequencyValidator.formatFrequency(centerFreq);
    const bwStr = FrequencyValidator.formatFrequency(bandwidth);
    const srStr = FrequencyValidator.formatFrequency(sampleRate);

    return `${deviceName} | ${freqStr} | BW: ${bwStr} | SR: ${srStr} | Gain: ${gainSettings}dB`;
  }

  // SIGIDWIKI Integration Methods
  public enableAutoSignalDetection(enable: boolean = true): void {
    this.autoDetectSignals = enable;
  }

  public addUserAnnotation(frequency: number, description: string): void {
    const annotation = SigidWikiIntegration.createUserAnnotation(frequency, description);
    this.annotations.push(annotation);
    this.render();
  }

  public getSignalsInCurrentView(): SignalAnnotation[] {
    if (!this.xScale) return [];

    const [freqStart, freqStop] = this.xScale.domain();
    return SigidWikiIntegration.findSignalsInRange(freqStart, freqStop);
  }

  public getDetectedSignals(): SignalAnnotation[] {
    return this.detectedSignals.slice(); // Return copy
  }

  public clearDetectedSignals(): void {
    this.detectedSignals = [];
    this.annotations = this.annotations.filter(a => a.sigidwiki?.source !== 'detected');
    this.render();
  }

  public identifySignalAtFrequency(frequency: number): SignalAnnotation | null {
    return SigidWikiIntegration.findExactSignal(frequency);
  }

  public destroy(): void {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
    if (this.imageBitmap) {
      this.imageBitmap.close();
    }
    this.container.selectAll('*').remove();
  }
}
