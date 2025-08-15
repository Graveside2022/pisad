import { test, expect, Page } from "@playwright/test";

test.describe("Signal Detection Workflow", () => {
  let page: Page;

  test.beforeEach(async ({ page: testPage }) => {
    page = testPage;
    await page.goto("http://localhost:3000");
    await page.waitForLoadState("networkidle");
  });

  test.describe("Signal Visualization", () => {
    test("should display real-time RSSI values", async () => {
      const signalMeter = page.locator('[data-testid="signal-meter"]');
      await expect(signalMeter).toBeVisible();

      // Check RSSI display
      const rssiValue = signalMeter.locator('[data-testid="rssi-value"]');
      await expect(rssiValue).toBeVisible();
      await expect(rssiValue).toContainText('dBm');

      // Check noise floor
      const noiseFloor = signalMeter.locator('[data-testid="noise-floor"]');
      await expect(noiseFloor).toBeVisible();
      await expect(noiseFloor).toContainText('dBm');

      // Check SNR
      const snr = signalMeter.locator('[data-testid="snr-value"]');
      await expect(snr).toBeVisible();
      await expect(snr).toContainText('dB');
    });

    test("should update signal strength indicator colors", async () => {
      // Simulate strong signal
      await page.evaluate(() => {
        window.postMessage({
          type: "rssi",
          data: {
            rssi: -45,
            noiseFloor: -90,
            snr: 45,
            confidence: 0.95
          }
        }, "*");
      });

      await page.waitForTimeout(500);

      const signalIndicator = page.locator('[data-testid="signal-strength-indicator"]');
      await expect(signalIndicator).toHaveClass(/signal-strong/);

      // Simulate weak signal
      await page.evaluate(() => {
        window.postMessage({
          type: "rssi",
          data: {
            rssi: -85,
            noiseFloor: -90,
            snr: 5,
            confidence: 0.3
          }
        }, "*");
      });

      await page.waitForTimeout(500);
      await expect(signalIndicator).toHaveClass(/signal-weak/);
    });

    test("should display signal history graph", async () => {
      const rssiGraph = page.locator('[data-testid="rssi-graph"]');
      await expect(rssiGraph).toBeVisible();

      // Check if canvas is rendered
      const canvas = rssiGraph.locator('canvas');
      await expect(canvas).toBeVisible();

      // Verify graph updates with new data
      for (let i = 0; i < 5; i++) {
        await page.evaluate((rssi) => {
          window.postMessage({
            type: "rssi",
            data: {
              rssi: rssi,
              noiseFloor: -90,
              snr: rssi + 90,
              confidence: 0.8
            }
          }, "*");
        }, -60 + i * 5);

        await page.waitForTimeout(100);
      }

      // Check if graph has data points
      const dataPoints = await page.evaluate(() => {
        const canvas = document.querySelector('[data-testid="rssi-graph"] canvas') as HTMLCanvasElement;
        return canvas?.getContext('2d')?.getImageData(0, 0, canvas.width, canvas.height).data.some(pixel => pixel > 0);
      });

      expect(dataPoints).toBeTruthy();
    });
  });

  test.describe("Detection Events", () => {
    test("should log detection events", async () => {
      const detectionLog = page.locator('[data-testid="detection-log"]');
      await expect(detectionLog).toBeVisible();

      // Simulate detection event
      await page.evaluate(() => {
        window.postMessage({
          type: "detection",
          data: {
            id: "det-001",
            timestamp: new Date().toISOString(),
            frequency: 433.92e6,
            rssi: -40,
            snr: 35,
            confidence: 0.92,
            location: { lat: 42.3601, lon: -71.0589 }
          }
        }, "*");
      });

      await page.waitForTimeout(500);

      const logEntry = detectionLog.locator('[data-testid="detection-entry"]').first();
      await expect(logEntry).toBeVisible();
      await expect(logEntry).toContainText('-40 dBm');
      await expect(logEntry).toContainText('433.92 MHz');
      await expect(logEntry).toContainText('92%');
    });

    test("should highlight strong detections", async () => {
      // Send strong detection
      await page.evaluate(() => {
        window.postMessage({
          type: "detection",
          data: {
            id: "det-strong",
            timestamp: new Date().toISOString(),
            frequency: 433.92e6,
            rssi: -35,
            snr: 50,
            confidence: 0.98
          }
        }, "*");
      });

      await page.waitForTimeout(500);

      const strongDetection = page.locator('[data-testid="detection-entry"]').first();
      await expect(strongDetection).toHaveClass(/detection-strong/);
      await expect(strongDetection.locator('[data-testid="strong-signal-badge"]')).toBeVisible();
    });

    test("should filter detection log by confidence", async () => {
      // Send multiple detections with varying confidence
      const detections = [
        { id: "det-1", confidence: 0.95 },
        { id: "det-2", confidence: 0.75 },
        { id: "det-3", confidence: 0.55 },
        { id: "det-4", confidence: 0.35 }
      ];

      for (const det of detections) {
        await page.evaluate((detection) => {
          window.postMessage({
            type: "detection",
            data: {
              id: detection.id,
              timestamp: new Date().toISOString(),
              frequency: 433.92e6,
              rssi: -60,
              snr: 20,
              confidence: detection.confidence
            }
          }, "*");
        }, det);
      }

      await page.waitForTimeout(500);

      // Set confidence filter
      await page.fill('[data-testid="confidence-filter"]', '0.7');
      await page.click('[data-testid="apply-filter"]');

      const visibleEntries = await page.locator('[data-testid="detection-entry"]:visible').count();
      expect(visibleEntries).toBe(2); // Only high confidence detections
    });
  });

  test.describe("Signal Search", () => {
    test("should start signal search", async () => {
      await page.click('[data-testid="start-search-button"]');

      await expect(page.locator('[data-testid="search-status"]')).toContainText('SEARCHING');
      await expect(page.locator('[data-testid="search-progress"]')).toBeVisible();

      // Verify SDR status
      const sdrStatus = page.locator('[data-testid="sdr-status"]');
      await expect(sdrStatus).toContainText('ACTIVE');
    });

    test("should display search progress", async () => {
      await page.click('[data-testid="start-search-button"]');

      const progressBar = page.locator('[data-testid="search-progress-bar"]');
      await expect(progressBar).toBeVisible();

      // Simulate progress updates
      for (let i = 0; i <= 100; i += 20) {
        await page.evaluate((progress) => {
          window.postMessage({
            type: "searchProgress",
            data: { progress: progress }
          }, "*");
        }, i);

        await page.waitForTimeout(200);

        const progressValue = await progressBar.getAttribute('aria-valuenow');
        expect(parseInt(progressValue || '0')).toBe(i);
      }
    });

    test("should stop signal search", async () => {
      await page.click('[data-testid="start-search-button"]');
      await expect(page.locator('[data-testid="search-status"]')).toContainText('SEARCHING');

      await page.click('[data-testid="stop-search-button"]');

      await expect(page.locator('[data-testid="search-status"]')).toContainText('IDLE');
      await expect(page.locator('[data-testid="search-progress"]')).not.toBeVisible();
    });
  });

  test.describe("Frequency Scanning", () => {
    test("should configure frequency scan parameters", async () => {
      await page.click('[data-testid="scan-config-button"]');

      const scanDialog = page.locator('[role="dialog"]:has-text("Frequency Scan Configuration")');
      await expect(scanDialog).toBeVisible();

      await page.fill('[name="startFrequency"]', '433000000');
      await page.fill('[name="endFrequency"]', '434000000');
      await page.fill('[name="stepSize"]', '10000');
      await page.fill('[name="dwellTime"]', '100');

      await page.click('button:has-text("Start Scan")');

      await expect(page.locator('[data-testid="scan-status"]')).toContainText('SCANNING');
    });

    test("should display scan results", async () => {
      // Start scan
      await page.click('[data-testid="quick-scan-button"]');

      // Simulate scan results
      await page.evaluate(() => {
        window.postMessage({
          type: "scanResult",
          data: {
            frequency: 433.92e6,
            rssi: -55,
            detected: true
          }
        }, "*");
      });

      await page.waitForTimeout(500);

      const scanResults = page.locator('[data-testid="scan-results"]');
      await expect(scanResults).toBeVisible();
      await expect(scanResults).toContainText('433.92 MHz');
      await expect(scanResults).toContainText('-55 dBm');
      await expect(scanResults.locator('[data-testid="detected-badge"]')).toBeVisible();
    });

    test("should auto-tune to strongest signal", async () => {
      // Simulate multiple scan results
      const frequencies = [
        { freq: 433.90e6, rssi: -70 },
        { freq: 433.92e6, rssi: -45 }, // Strongest
        { freq: 433.94e6, rssi: -65 }
      ];

      for (const f of frequencies) {
        await page.evaluate((data) => {
          window.postMessage({
            type: "scanResult",
            data: {
              frequency: data.freq,
              rssi: data.rssi,
              detected: true
            }
          }, "*");
        }, f);
      }

      await page.waitForTimeout(500);

      await page.click('[data-testid="auto-tune-button"]');

      await expect(page.locator('[data-testid="current-frequency"]')).toContainText('433.92');
      await expect(page.locator('text=Tuned to strongest signal')).toBeVisible();
    });
  });

  test.describe("Signal Analysis", () => {
    test("should display signal characteristics", async () => {
      const signalAnalysis = page.locator('[data-testid="signal-analysis"]');
      await expect(signalAnalysis).toBeVisible();

      // Simulate signal with specific pattern
      for (let i = 0; i < 10; i++) {
        await page.evaluate((index) => {
          window.postMessage({
            type: "rssi",
            data: {
              rssi: -60 + Math.sin(index) * 10,
              noiseFloor: -90,
              snr: 30 + Math.sin(index) * 10,
              confidence: 0.8
            }
          }, "*");
        }, i);

        await page.waitForTimeout(100);
      }

      // Check analysis results
      await expect(signalAnalysis.locator('[data-testid="avg-rssi"]')).toBeVisible();
      await expect(signalAnalysis.locator('[data-testid="peak-rssi"]')).toBeVisible();
      await expect(signalAnalysis.locator('[data-testid="signal-variance"]')).toBeVisible();
      await expect(signalAnalysis.locator('[data-testid="signal-stability"]')).toBeVisible();
    });

    test("should detect beacon patterns", async () => {
      // Simulate beacon pattern
      const pattern = [
        { duration: 500, rssi: -45 },  // Pulse
        { duration: 1000, rssi: -90 }, // Gap
        { duration: 500, rssi: -45 },  // Pulse
        { duration: 2000, rssi: -90 }, // Long gap
      ];

      for (const p of pattern) {
        await page.evaluate((data) => {
          window.postMessage({
            type: "rssi",
            data: {
              rssi: data.rssi,
              noiseFloor: -95,
              snr: data.rssi + 95,
              confidence: data.rssi > -80 ? 0.1 : 0.9
            }
          }, "*");
        }, p);

        await page.waitForTimeout(p.duration);
      }

      const patternDetection = page.locator('[data-testid="pattern-detection"]');
      await expect(patternDetection).toBeVisible();
      await expect(patternDetection).toContainText('Pattern Detected');
      await expect(patternDetection.locator('[data-testid="pattern-type"]')).toBeVisible();
    });

    test("should calculate signal gradient", async () => {
      // Simulate approaching signal (increasing strength)
      for (let i = 0; i < 10; i++) {
        await page.evaluate((rssi) => {
          window.postMessage({
            type: "rssi",
            data: {
              rssi: rssi,
              noiseFloor: -90,
              snr: rssi + 90,
              confidence: 0.85
            }
          }, "*");
        }, -80 + i * 3);

        await page.waitForTimeout(200);
      }

      const gradient = page.locator('[data-testid="signal-gradient"]');
      await expect(gradient).toBeVisible();
      await expect(gradient).toContainText('APPROACHING');
      await expect(gradient.locator('[data-testid="gradient-arrow-up"]')).toBeVisible();
    });
  });

  test.describe("Detection Notifications", () => {
    test("should show notification for strong detection", async () => {
      await page.evaluate(() => {
        window.postMessage({
          type: "detection",
          data: {
            id: "det-notification",
            timestamp: new Date().toISOString(),
            frequency: 433.92e6,
            rssi: -30,
            snr: 55,
            confidence: 0.99
          }
        }, "*");
      });

      const notification = page.locator('[data-testid="detection-notification"]');
      await expect(notification).toBeVisible({ timeout: 5000 });
      await expect(notification).toContainText('Strong Signal Detected');
      await expect(notification).toContainText('-30 dBm');
    });

    test("should play audio alert for detection", async () => {
      // Enable audio alerts
      await page.click('[data-testid="settings-button"]');
      await page.check('[name="audioAlerts"]');
      await page.click('[data-testid="save-settings"]');

      // Mock audio play
      await page.evaluate(() => {
        window.AudioPlayed = false;
        window.HTMLAudioElement.prototype.play = function() {
          window.AudioPlayed = true;
          return Promise.resolve();
        };
      });

      // Trigger strong detection
      await page.evaluate(() => {
        window.postMessage({
          type: "detection",
          data: {
            id: "det-audio",
            timestamp: new Date().toISOString(),
            frequency: 433.92e6,
            rssi: -25,
            snr: 60,
            confidence: 1.0
          }
        }, "*");
      });

      await page.waitForTimeout(500);

      const audioPlayed = await page.evaluate(() => window.AudioPlayed);
      expect(audioPlayed).toBeTruthy();
    });
  });

  test.describe("Data Export", () => {
    test("should export detection log to CSV", async () => {
      // Generate some detections
      for (let i = 0; i < 5; i++) {
        await page.evaluate((index) => {
          window.postMessage({
            type: "detection",
            data: {
              id: `det-export-${index}`,
              timestamp: new Date().toISOString(),
              frequency: 433.92e6,
              rssi: -50 - index * 5,
              snr: 40 - index * 5,
              confidence: 0.9 - index * 0.1
            }
          }, "*");
        }, i);
      }

      await page.waitForTimeout(500);

      // Export data
      await page.click('[data-testid="export-button"]');
      await page.click('[data-testid="export-csv"]');

      const [download] = await Promise.all([
        page.waitForEvent('download'),
        page.click('[data-testid="confirm-export"]')
      ]);

      expect(download.suggestedFilename()).toContain('.csv');
      expect(download.suggestedFilename()).toContain('detections');
    });

    test("should export signal history graph", async () => {
      await page.click('[data-testid="export-button"]');
      await page.click('[data-testid="export-graph"]');

      const [download] = await Promise.all([
        page.waitForEvent('download'),
        page.click('[data-testid="confirm-export"]')
      ]);

      expect(download.suggestedFilename()).toContain('.png');
      expect(download.suggestedFilename()).toContain('signal-history');
    });
  });
});
