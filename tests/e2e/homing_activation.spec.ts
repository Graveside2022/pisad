import { test, expect, Page } from "@playwright/test";

test.describe("Homing Activation Flow", () => {
  let page: Page;

  test.beforeEach(async ({ page: testPage }) => {
    page = testPage;
    await page.goto("http://localhost:3000");
    await page.waitForLoadState("networkidle");
  });

  test.describe("Successful Activation", () => {
    test("should enable homing with all safety checks passing", async () => {
      await page.waitForSelector('[data-testid="homing-control"]', {
        timeout: 5000,
      });

      const toggleButton = page.locator('button:has-text("ENABLE")');
      await expect(toggleButton).toBeVisible();
      await toggleButton.click();

      await expect(
        page.locator("text=Confirm Homing Activation"),
      ).toBeVisible();
      await expect(
        page.locator("text=WARNING: Enabling homing mode"),
      ).toBeVisible();

      const confirmButton = page.locator('button:has-text("Confirm & Enable")');
      await confirmButton.click();

      await expect(page.locator("text=HOMING ENABLED")).toBeVisible({
        timeout: 5000,
      });
      await expect(page.locator("text=To regain control")).toBeVisible();
    });

    test("should show confirmation dialog with safety information", async () => {
      const toggleButton = page.locator('button:has-text("ENABLE")');
      await toggleButton.click();

      const dialog = page.locator('[role="dialog"]');
      await expect(dialog).toBeVisible();

      await expect(
        dialog.locator(
          "text=Automatically navigate towards the strongest signal",
        ),
      ).toBeVisible();
      await expect(
        dialog.locator(
          "text=Maintain current altitude unless specified otherwise",
        ),
      ).toBeVisible();
      await expect(
        dialog.locator(
          "text=Continue homing until disabled or safety limits reached",
        ),
      ).toBeVisible();
    });

    test("should cancel activation when cancel button is clicked", async () => {
      const toggleButton = page.locator('button:has-text("ENABLE")');
      await toggleButton.click();

      const cancelButton = page.locator(
        '[role="dialog"] button:has-text("Cancel")',
      );
      await cancelButton.click();

      await expect(page.locator('[role="dialog"]')).not.toBeVisible();
      await expect(page.locator("text=HOMING DISABLED")).toBeVisible();
    });
  });

  test.describe("Safety Interlock Blocking", () => {
    test("should show error when flight mode is not GUIDED", async () => {
      await page.evaluate(() => {
        window.postMessage(
          {
            type: "systemState",
            data: {
              flightMode: "MANUAL",
              safetyInterlocks: {
                modeCheck: false,
                batteryCheck: true,
                geofenceCheck: true,
                signalCheck: true,
                operatorCheck: true,
              },
            },
          },
          "*",
        );
      });

      await page.waitForTimeout(500);

      const toggleButton = page.locator('button:has-text("ENABLE")');
      await toggleButton.click();

      const confirmButton = page.locator('button:has-text("Confirm & Enable")');
      await confirmButton.click();

      await expect(page.locator("text=Safety interlock blocked")).toBeVisible({
        timeout: 5000,
      });
    });

    test("should show error when battery is below threshold", async () => {
      await page.evaluate(() => {
        window.postMessage(
          {
            type: "systemState",
            data: {
              batteryPercent: 15,
              safetyInterlocks: {
                modeCheck: true,
                batteryCheck: false,
                geofenceCheck: true,
                signalCheck: true,
                operatorCheck: true,
              },
            },
          },
          "*",
        );
      });

      await page.waitForTimeout(500);

      const toggleButton = page.locator('button:has-text("ENABLE")');
      await toggleButton.click();

      const confirmButton = page.locator('button:has-text("Confirm & Enable")');
      await confirmButton.click();

      await expect(page.locator("text=batteryCheck")).toBeVisible({
        timeout: 5000,
      });
    });

    test("should display all failed safety checks", async () => {
      await page.evaluate(() => {
        window.postMessage(
          {
            type: "systemState",
            data: {
              safetyInterlocks: {
                modeCheck: false,
                batteryCheck: false,
                geofenceCheck: false,
                signalCheck: true,
                operatorCheck: true,
              },
            },
          },
          "*",
        );
      });

      await page.waitForTimeout(500);

      const safetyPanel = page.locator('[data-testid="safety-interlocks"]');
      await expect(safetyPanel.locator("text=CHECKS FAILED")).toBeVisible();

      const failChips = safetyPanel.locator("text=FAIL");
      const failCount = await failChips.count();
      expect(failCount).toBeGreaterThanOrEqual(3);
    });
  });

  test.describe("Auto-Disable Triggers", () => {
    test("should display auto-disable conditions", async () => {
      const conditionsPanel = page.locator(
        '[data-testid="auto-disable-conditions"]',
      );
      await expect(conditionsPanel).toBeVisible();

      await expect(
        conditionsPanel.locator("text=Signal loss for 10 seconds"),
      ).toBeVisible();
      await expect(
        conditionsPanel.locator("text=Flight mode changed from GUIDED"),
      ).toBeVisible();
      await expect(
        conditionsPanel.locator("text=Battery below 20%"),
      ).toBeVisible();
      await expect(
        conditionsPanel.locator("text=Geofence boundary reached"),
      ).toBeVisible();
      await expect(
        conditionsPanel.locator("text=Emergency stop activated"),
      ).toBeVisible();
    });

    test("should highlight active auto-disable conditions", async () => {
      await page.evaluate(() => {
        window.postMessage(
          {
            type: "systemState",
            data: {
              homingEnabled: true,
              safetyInterlocks: {
                modeCheck: true,
                batteryCheck: false,
                geofenceCheck: true,
                signalCheck: false,
                operatorCheck: true,
              },
            },
          },
          "*",
        );
      });

      await page.waitForTimeout(500);

      const conditionsPanel = page.locator(
        '[data-testid="auto-disable-conditions"]',
      );
      await expect(
        conditionsPanel.locator("text=2 condition(s) may trigger auto-disable"),
      ).toBeVisible();
      await expect(
        conditionsPanel.locator("text=ACTIVE").first(),
      ).toBeVisible();
    });
  });

  test.describe("Emergency Stop", () => {
    test("should trigger emergency stop with confirmation", async () => {
      const emergencyButton = page.locator('button:has-text("EMERGENCY STOP")');
      await expect(emergencyButton).toBeVisible();
      await emergencyButton.click();

      const confirmDialog = page.locator(
        '[role="dialog"]:has-text("CONFIRM EMERGENCY STOP")',
      );
      await expect(confirmDialog).toBeVisible();

      await expect(confirmDialog.locator("text=CRITICAL ACTION")).toBeVisible();
      await expect(
        confirmDialog.locator("text=Disable all homing operations"),
      ).toBeVisible();
      await expect(
        confirmDialog.locator("text=Stop all autonomous navigation"),
      ).toBeVisible();

      const confirmButton = confirmDialog.locator(
        'button:has-text("CONFIRM STOP")',
      );
      await confirmButton.click();

      await expect(
        page.locator("text=Emergency stop activated successfully"),
      ).toBeVisible({ timeout: 5000 });
    });

    test("should cancel emergency stop", async () => {
      const emergencyButton = page.locator('button:has-text("EMERGENCY STOP")');
      await emergencyButton.click();

      const cancelButton = page.locator(
        '[role="dialog"] button:has-text("Cancel")',
      );
      await cancelButton.click();

      await expect(page.locator('[role="dialog"]')).not.toBeVisible();
    });
  });

  test.describe("Homing Deactivation", () => {
    test("should disable homing without confirmation", async () => {
      await page.evaluate(() => {
        window.postMessage(
          {
            type: "systemState",
            data: {
              homingEnabled: true,
              currentState: "HOMING",
            },
          },
          "*",
        );
      });

      await page.waitForTimeout(500);
      await expect(page.locator("text=HOMING ACTIVE")).toBeVisible();

      const toggleButton = page.locator('button:has-text("DISABLE")');
      await toggleButton.click();

      await expect(page.locator('[role="dialog"]')).not.toBeVisible();
      await expect(
        page.locator("text=Homing disabled successfully"),
      ).toBeVisible({ timeout: 5000 });
      await expect(page.locator("text=HOMING DISABLED")).toBeVisible();
    });
  });

  test.describe("Real-time Updates", () => {
    test("should update UI when WebSocket sends state changes", async () => {
      await expect(page.locator("text=HOMING DISABLED")).toBeVisible();

      await page.evaluate(() => {
        window.postMessage(
          {
            type: "systemState",
            data: {
              homingEnabled: true,
              currentState: "SEARCHING",
            },
          },
          "*",
        );
      });

      await expect(page.locator("text=HOMING ENABLED")).toBeVisible({
        timeout: 5000,
      });

      await page.evaluate(() => {
        window.postMessage(
          {
            type: "systemState",
            data: {
              homingEnabled: true,
              currentState: "HOMING",
            },
          },
          "*",
        );
      });

      await expect(page.locator("text=HOMING ACTIVE")).toBeVisible({
        timeout: 5000,
      });
    });

    test("should update safety interlock status in real-time", async () => {
      const safetyPanel = page.locator('[data-testid="safety-interlocks"]');

      await page.evaluate(() => {
        window.postMessage(
          {
            type: "safetyStatus",
            data: {
              safetyInterlocks: {
                modeCheck: true,
                batteryCheck: true,
                geofenceCheck: true,
                signalCheck: true,
                operatorCheck: true,
              },
            },
          },
          "*",
        );
      });

      await expect(safetyPanel.locator("text=ALL SYSTEMS GO")).toBeVisible({
        timeout: 5000,
      });

      await page.evaluate(() => {
        window.postMessage(
          {
            type: "safetyStatus",
            data: {
              safetyInterlocks: {
                modeCheck: true,
                batteryCheck: false,
                geofenceCheck: true,
                signalCheck: true,
                operatorCheck: true,
              },
            },
          },
          "*",
        );
      });

      await expect(safetyPanel.locator("text=CHECKS FAILED")).toBeVisible({
        timeout: 5000,
      });
    });
  });

  test.describe("Responsive Design", () => {
    test("should work on Pi display resolution (1024x600)", async () => {
      await page.setViewportSize({ width: 1024, height: 600 });

      const homingControl = page.locator('[data-testid="homing-control"]');
      await expect(homingControl).toBeVisible();

      const toggleButton = page.locator('button:has-text("ENABLE")');
      await expect(toggleButton).toBeVisible();

      const emergencyButton = page.locator('button:has-text("EMERGENCY STOP")');
      await expect(emergencyButton).toBeVisible();

      const viewportSize = page.viewportSize();
      expect(viewportSize?.width).toBe(1024);
      expect(viewportSize?.height).toBe(600);
    });
  });
});
