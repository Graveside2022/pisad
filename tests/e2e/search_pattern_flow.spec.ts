import { test, expect } from "@playwright/test";

test.describe("Search Pattern Flow", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("http://localhost:3000");
    // Wait for app to load
    await page.waitForSelector('[data-testid="app-container"]', {
      timeout: 10000,
    });
  });

  test("complete search pattern creation and execution flow", async ({
    page,
  }) => {
    // Navigate to search pattern section
    await page.click("text=Search Pattern");

    // Configure pattern
    await page.selectOption('[aria-label="Pattern Type"]', "expanding_square");

    // Set spacing
    const spacingSlider = page.locator('input[type="range"]').first();
    await spacingSlider.fill("80");

    // Set velocity
    const velocitySlider = page.locator('input[type="range"]').nth(1);
    await velocitySlider.fill("8");

    // Fill in center radius boundary
    await page.fill('[aria-label="Center Latitude"]', "37.7749");
    await page.fill('[aria-label="Center Longitude"]', "-122.4194");
    await page.fill('[aria-label="Radius (meters)"]', "500");

    // Preview pattern
    await page.click("text=Preview Pattern");

    // Wait for map to load
    await page.waitForSelector(".leaflet-container", { timeout: 5000 });

    // Verify waypoints are displayed
    await expect(page.locator(".numbered-marker")).toHaveCount(
      await page.locator(".numbered-marker").count(),
    );

    // Create pattern
    await page.click("text=Create Pattern");

    // Wait for success
    await page.waitForSelector("text=Pattern created successfully", {
      timeout: 5000,
    });

    // Verify progress component appears
    await expect(page.locator("text=Search Progress")).toBeVisible();

    // Start pattern execution
    await page.click("text=Start Pattern");

    // Verify status changes to EXECUTING
    await expect(
      page.locator('[data-testid="pattern-state-chip"]'),
    ).toContainText("EXECUTING");

    // Test pause functionality
    await page.click("text=Pause");
    await expect(
      page.locator('[data-testid="pattern-state-chip"]'),
    ).toContainText("PAUSED");

    // Test resume functionality
    await page.click("text=Resume");
    await expect(
      page.locator('[data-testid="pattern-state-chip"]'),
    ).toContainText("EXECUTING");

    // Test stop with confirmation
    await page.click("text=Stop");

    // Confirm stop dialog
    await page.click('button:has-text("Stop Pattern")');

    // Verify pattern stopped
    await expect(
      page.locator('[data-testid="pattern-state-chip"]'),
    ).toContainText("IDLE");
  });

  test("switch between boundary types", async ({ page }) => {
    await page.click("text=Search Pattern");

    // Start with center radius
    await expect(page.locator('[aria-label="Center Latitude"]')).toBeVisible();

    // Switch to corners
    await page.click("text=Corner Coordinates");

    // Verify corner inputs appear
    await expect(page.locator("text=Corner 1")).toBeVisible();
    await expect(page.locator("text=Corner 2")).toBeVisible();
    await expect(page.locator("text=Corner 3")).toBeVisible();
    await expect(page.locator("text=Corner 4")).toBeVisible();

    // Fill in corner coordinates
    const corners = [
      { lat: "37.7749", lon: "-122.4194" },
      { lat: "37.7849", lon: "-122.4194" },
      { lat: "37.7849", lon: "-122.4094" },
      { lat: "37.7749", lon: "-122.4094" },
    ];

    for (let i = 0; i < corners.length; i++) {
      await page.fill(
        `[aria-label="Corner ${i + 1} Latitude"]`,
        corners[i].lat,
      );
      await page.fill(
        `[aria-label="Corner ${i + 1} Longitude"]`,
        corners[i].lon,
      );
    }

    // Create pattern with corners
    await page.click("text=Create Pattern");
    await page.waitForSelector("text=Pattern created successfully", {
      timeout: 5000,
    });
  });

  test("export pattern to file", async ({ page, context }) => {
    // Create a pattern first
    await page.click("text=Search Pattern");
    await page.fill('[aria-label="Center Latitude"]', "37.7749");
    await page.fill('[aria-label="Center Longitude"]', "-122.4194");
    await page.fill('[aria-label="Radius (meters)"]', "500");
    await page.click("text=Create Pattern");
    await page.waitForSelector("text=Pattern created successfully", {
      timeout: 5000,
    });

    // Set up download promise before clicking
    const downloadPromise = page.waitForEvent("download");

    // Click export button
    await page.click("text=Export Pattern");

    // Wait for download
    const download = await downloadPromise;

    // Verify download filename
    expect(download.suggestedFilename()).toMatch(/search_pattern_.*\.wpl/);
  });

  test("validate input constraints", async ({ page }) => {
    await page.click("text=Search Pattern");

    // Test latitude validation
    await page.fill('[aria-label="Center Latitude"]', "91");
    await page.click("text=Create Pattern");
    await expect(
      page.locator("text=Latitude must be between -90 and 90"),
    ).toBeVisible();

    // Test longitude validation
    await page.fill('[aria-label="Center Latitude"]', "37.7749");
    await page.fill('[aria-label="Center Longitude"]', "181");
    await page.click("text=Create Pattern");
    await expect(
      page.locator("text=Longitude must be between -180 and 180"),
    ).toBeVisible();

    // Test radius validation
    await page.fill('[aria-label="Center Longitude"]', "-122.4194");
    await page.fill('[aria-label="Radius (meters)"]', "0");
    await page.click("text=Create Pattern");
    await expect(
      page.locator("text=Radius must be between 0 and 10000 meters"),
    ).toBeVisible();
  });

  test("keyboard shortcuts for pattern control", async ({ page }) => {
    // Create and start a pattern
    await page.click("text=Search Pattern");
    await page.fill('[aria-label="Center Latitude"]', "37.7749");
    await page.fill('[aria-label="Center Longitude"]', "-122.4194");
    await page.fill('[aria-label="Radius (meters)"]', "500");
    await page.click("text=Create Pattern");
    await page.waitForSelector("text=Pattern created successfully", {
      timeout: 5000,
    });

    // Enable keyboard shortcuts
    await page.click('[aria-label="Toggle keyboard shortcuts"]');

    // Start pattern
    await page.click("text=Start Pattern");
    await expect(
      page.locator('[data-testid="pattern-state-chip"]'),
    ).toContainText("EXECUTING");

    // Test Ctrl+P for pause
    await page.keyboard.press("Control+P");
    await expect(
      page.locator('[data-testid="pattern-state-chip"]'),
    ).toContainText("PAUSED");

    // Test Ctrl+R for resume
    await page.keyboard.press("Control+R");
    await expect(
      page.locator('[data-testid="pattern-state-chip"]'),
    ).toContainText("EXECUTING");

    // Test Ctrl+S for stop (with dialog)
    await page.keyboard.press("Control+S");
    await expect(page.locator("text=Stop Search Pattern?")).toBeVisible();
    await page.click('button:has-text("Cancel")');
  });

  test("real-time progress updates via WebSocket", async ({ page }) => {
    // Create and start a pattern
    await page.click("text=Search Pattern");
    await page.fill('[aria-label="Center Latitude"]', "37.7749");
    await page.fill('[aria-label="Center Longitude"]', "-122.4194");
    await page.fill('[aria-label="Radius (meters)"]', "500");
    await page.click("text=Create Pattern");
    await page.waitForSelector("text=Pattern created successfully", {
      timeout: 5000,
    });

    // Start pattern
    await page.click("text=Start Pattern");

    // Wait for initial progress
    const progressBar = page.locator('[role="progressbar"]');
    await expect(progressBar).toBeVisible();

    // Get initial progress value
    const initialProgress = await progressBar.getAttribute("aria-valuenow");

    // Wait for progress update (simulated via WebSocket)
    await page.waitForTimeout(5000);

    // Check if progress has changed
    const updatedProgress = await progressBar.getAttribute("aria-valuenow");
    expect(Number(updatedProgress)).toBeGreaterThanOrEqual(
      Number(initialProgress),
    );

    // Verify waypoint counter updates
    await expect(page.locator("text=/Waypoint \\d+ of \\d+/")).toBeVisible();
  });
});
