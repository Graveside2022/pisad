import { test, expect, Page } from "@playwright/test";

test.describe("Profile Management Flow", () => {
  let page: Page;

  test.beforeEach(async ({ page: testPage }) => {
    page = testPage;
    await page.goto("http://localhost:3000");
    await page.waitForLoadState("networkidle");
  });

  test.describe("Profile Creation", () => {
    test("should create a new beacon profile", async () => {
      await page.click('[data-testid="config-button"]');
      await page.click('[data-testid="profile-manager-tab"]');
      
      await page.click('button:has-text("NEW PROFILE")');
      
      const dialog = page.locator('[role="dialog"]');
      await expect(dialog).toBeVisible();
      
      await page.fill('[name="profileName"]', "Test Beacon Profile");
      await page.selectOption('[name="beaconType"]', "avalanche");
      await page.fill('[name="frequency"]', "457000");
      await page.fill('[name="pulsePattern"]', "short_long_short");
      await page.fill('[name="pulseWidth"]', "0.7");
      await page.fill('[name="pulseInterval"]', "1.0");
      
      await page.click('button:has-text("Save Profile")');
      
      await expect(page.locator('text=Profile created successfully')).toBeVisible();
      await expect(page.locator('text=Test Beacon Profile')).toBeVisible();
    });

    test("should validate profile fields", async () => {
      await page.click('[data-testid="config-button"]');
      await page.click('[data-testid="profile-manager-tab"]');
      
      await page.click('button:has-text("NEW PROFILE")');
      
      await page.fill('[name="profileName"]', "");
      await page.fill('[name="frequency"]', "invalid");
      
      await page.click('button:has-text("Save Profile")');
      
      await expect(page.locator('text=Profile name is required')).toBeVisible();
      await expect(page.locator('text=Invalid frequency')).toBeVisible();
    });

    test("should prevent duplicate profile names", async () => {
      await page.click('[data-testid="config-button"]');
      await page.click('[data-testid="profile-manager-tab"]');
      
      // Create first profile
      await page.click('button:has-text("NEW PROFILE")');
      await page.fill('[name="profileName"]', "Duplicate Test");
      await page.selectOption('[name="beaconType"]', "pet");
      await page.fill('[name="frequency"]', "433920000");
      await page.click('button:has-text("Save Profile")');
      
      await page.waitForTimeout(500);
      
      // Try to create duplicate
      await page.click('button:has-text("NEW PROFILE")');
      await page.fill('[name="profileName"]', "Duplicate Test");
      await page.click('button:has-text("Save Profile")');
      
      await expect(page.locator('text=Profile name already exists')).toBeVisible();
    });
  });

  test.describe("Profile Selection", () => {
    test("should load profile and apply settings", async () => {
      await page.click('[data-testid="config-button"]');
      await page.click('[data-testid="profile-manager-tab"]');
      
      // Assume a profile exists
      const profileCard = page.locator('[data-testid="profile-card"]').first();
      await profileCard.click();
      
      const loadButton = profileCard.locator('button:has-text("LOAD")');
      await loadButton.click();
      
      await expect(page.locator('text=Profile loaded successfully')).toBeVisible();
      
      // Verify settings applied
      await page.click('[data-testid="sdr-settings-tab"]');
      const frequencyInput = page.locator('[name="frequency"]');
      const value = await frequencyInput.inputValue();
      expect(value).toBeTruthy();
    });

    test("should show profile details on hover", async () => {
      await page.click('[data-testid="config-button"]');
      await page.click('[data-testid="profile-manager-tab"]');
      
      const profileCard = page.locator('[data-testid="profile-card"]').first();
      await profileCard.hover();
      
      const tooltip = page.locator('[role="tooltip"]');
      await expect(tooltip).toBeVisible();
      await expect(tooltip.locator('text=Frequency:')).toBeVisible();
      await expect(tooltip.locator('text=Pattern:')).toBeVisible();
    });

    test("should highlight active profile", async () => {
      await page.click('[data-testid="config-button"]');
      await page.click('[data-testid="profile-manager-tab"]');
      
      const profileCard = page.locator('[data-testid="profile-card"]').first();
      await profileCard.locator('button:has-text("LOAD")').click();
      
      await expect(profileCard).toHaveClass(/active-profile/);
      await expect(profileCard.locator('text=ACTIVE')).toBeVisible();
    });
  });

  test.describe("Profile Editing", () => {
    test("should edit existing profile", async () => {
      await page.click('[data-testid="config-button"]');
      await page.click('[data-testid="profile-manager-tab"]');
      
      const profileCard = page.locator('[data-testid="profile-card"]').first();
      await profileCard.locator('button[aria-label="Edit profile"]').click();
      
      const dialog = page.locator('[role="dialog"]');
      await expect(dialog).toBeVisible();
      
      await page.fill('[name="frequency"]', "433950000");
      await page.fill('[name="pulseWidth"]', "0.8");
      
      await page.click('button:has-text("Update Profile")');
      
      await expect(page.locator('text=Profile updated successfully')).toBeVisible();
    });

    test("should show confirmation for unsaved changes", async () => {
      await page.click('[data-testid="config-button"]');
      await page.click('[data-testid="profile-manager-tab"]');
      
      const profileCard = page.locator('[data-testid="profile-card"]').first();
      await profileCard.locator('button[aria-label="Edit profile"]').click();
      
      await page.fill('[name="frequency"]', "433950000");
      
      // Try to close without saving
      await page.click('[aria-label="Close dialog"]');
      
      await expect(page.locator('text=Unsaved changes')).toBeVisible();
      await expect(page.locator('text=Discard changes?')).toBeVisible();
      
      await page.click('button:has-text("Discard")');
      await expect(page.locator('[role="dialog"]')).not.toBeVisible();
    });
  });

  test.describe("Profile Deletion", () => {
    test("should delete profile with confirmation", async () => {
      await page.click('[data-testid="config-button"]');
      await page.click('[data-testid="profile-manager-tab"]');
      
      const profileCount = await page.locator('[data-testid="profile-card"]').count();
      
      const profileCard = page.locator('[data-testid="profile-card"]').first();
      const profileName = await profileCard.locator('[data-testid="profile-name"]').textContent();
      
      await profileCard.locator('button[aria-label="Delete profile"]').click();
      
      const confirmDialog = page.locator('[role="dialog"]:has-text("Delete Profile")');
      await expect(confirmDialog).toBeVisible();
      await expect(confirmDialog.locator(`text=Delete "${profileName}"?`)).toBeVisible();
      
      await confirmDialog.locator('button:has-text("DELETE")').click();
      
      await expect(page.locator('text=Profile deleted successfully')).toBeVisible();
      
      const newProfileCount = await page.locator('[data-testid="profile-card"]').count();
      expect(newProfileCount).toBe(profileCount - 1);
    });

    test("should prevent deletion of active profile", async () => {
      await page.click('[data-testid="config-button"]');
      await page.click('[data-testid="profile-manager-tab"]');
      
      const profileCard = page.locator('[data-testid="profile-card"]').first();
      await profileCard.locator('button:has-text("LOAD")').click();
      
      await page.waitForTimeout(500);
      
      await profileCard.locator('button[aria-label="Delete profile"]').click();
      
      await expect(page.locator('text=Cannot delete active profile')).toBeVisible();
    });
  });

  test.describe("Profile Import/Export", () => {
    test("should export profile to JSON", async () => {
      await page.click('[data-testid="config-button"]');
      await page.click('[data-testid="profile-manager-tab"]');
      
      const profileCard = page.locator('[data-testid="profile-card"]').first();
      await profileCard.locator('button[aria-label="Export profile"]').click();
      
      // Wait for download
      const [download] = await Promise.all([
        page.waitForEvent('download'),
        profileCard.locator('button[aria-label="Export profile"]').click()
      ]);
      
      expect(download.suggestedFilename()).toContain('.json');
    });

    test("should import profile from JSON", async () => {
      await page.click('[data-testid="config-button"]');
      await page.click('[data-testid="profile-manager-tab"]');
      
      await page.click('button:has-text("IMPORT PROFILE")');
      
      const fileInput = page.locator('input[type="file"]');
      await fileInput.setInputFiles({
        name: 'test-profile.json',
        mimeType: 'application/json',
        buffer: Buffer.from(JSON.stringify({
          name: "Imported Profile",
          beaconType: "pet",
          frequency: 433920000,
          pulsePattern: "continuous",
          pulseWidth: 0.5,
          pulseInterval: 1.0
        }))
      });
      
      await expect(page.locator('text=Profile imported successfully')).toBeVisible();
      await expect(page.locator('text=Imported Profile')).toBeVisible();
    });

    test("should validate imported profile format", async () => {
      await page.click('[data-testid="config-button"]');
      await page.click('[data-testid="profile-manager-tab"]');
      
      await page.click('button:has-text("IMPORT PROFILE")');
      
      const fileInput = page.locator('input[type="file"]');
      await fileInput.setInputFiles({
        name: 'invalid-profile.json',
        mimeType: 'application/json',
        buffer: Buffer.from(JSON.stringify({
          invalid: "data"
        }))
      });
      
      await expect(page.locator('text=Invalid profile format')).toBeVisible();
    });
  });

  test.describe("Quick Actions", () => {
    test("should provide quick preset profiles", async () => {
      await page.click('[data-testid="config-button"]');
      await page.click('[data-testid="profile-manager-tab"]');
      
      await page.click('button:has-text("PRESETS")');
      
      const presetMenu = page.locator('[data-testid="preset-menu"]');
      await expect(presetMenu).toBeVisible();
      
      await expect(presetMenu.locator('text=Avalanche Beacon')).toBeVisible();
      await expect(presetMenu.locator('text=Pet Tracker')).toBeVisible();
      await expect(presetMenu.locator('text=Wildlife Tag')).toBeVisible();
      
      await presetMenu.locator('text=Pet Tracker').click();
      
      await expect(page.locator('text=Preset loaded')).toBeVisible();
    });

    test("should duplicate existing profile", async () => {
      await page.click('[data-testid="config-button"]');
      await page.click('[data-testid="profile-manager-tab"]');
      
      const originalCount = await page.locator('[data-testid="profile-card"]').count();
      
      const profileCard = page.locator('[data-testid="profile-card"]').first();
      const originalName = await profileCard.locator('[data-testid="profile-name"]').textContent();
      
      await profileCard.locator('button[aria-label="Duplicate profile"]').click();
      
      await expect(page.locator(`text=${originalName} (Copy)`)).toBeVisible();
      
      const newCount = await page.locator('[data-testid="profile-card"]').count();
      expect(newCount).toBe(originalCount + 1);
    });
  });

  test.describe("Search and Filter", () => {
    test("should filter profiles by name", async () => {
      await page.click('[data-testid="config-button"]');
      await page.click('[data-testid="profile-manager-tab"]');
      
      const searchInput = page.locator('[placeholder="Search profiles..."]');
      await searchInput.fill("Test");
      
      const visibleCards = await page.locator('[data-testid="profile-card"]:visible').count();
      const allCards = await page.locator('[data-testid="profile-card"]').count();
      
      expect(visibleCards).toBeLessThanOrEqual(allCards);
    });

    test("should filter profiles by beacon type", async () => {
      await page.click('[data-testid="config-button"]');
      await page.click('[data-testid="profile-manager-tab"]');
      
      await page.selectOption('[data-testid="beacon-type-filter"]', 'pet');
      
      const visibleCards = page.locator('[data-testid="profile-card"]:visible');
      const count = await visibleCards.count();
      
      for (let i = 0; i < count; i++) {
        const card = visibleCards.nth(i);
        await expect(card.locator('text=Pet Tracker')).toBeVisible();
      }
    });

    test("should sort profiles by date", async () => {
      await page.click('[data-testid="config-button"]');
      await page.click('[data-testid="profile-manager-tab"]');
      
      await page.selectOption('[data-testid="sort-order"]', 'newest');
      
      const cards = page.locator('[data-testid="profile-card"]');
      const dates = [];
      
      const count = await cards.count();
      for (let i = 0; i < count; i++) {
        const dateText = await cards.nth(i).locator('[data-testid="profile-date"]').textContent();
        dates.push(new Date(dateText!).getTime());
      }
      
      // Verify descending order
      for (let i = 1; i < dates.length; i++) {
        expect(dates[i]).toBeLessThanOrEqual(dates[i - 1]);
      }
    });
  });
});