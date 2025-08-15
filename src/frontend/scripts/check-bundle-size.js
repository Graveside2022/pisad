#!/usr/bin/env node

/**
 * Bundle Size Checker
 * Story 4.9 Task 11: Frontend Bundle Optimization
 *
 * This script checks the production bundle size and ensures it's under 1.5MB.
 * Run after build: npm run build && node scripts/check-bundle-size.js
 */

const fs = require('fs');
const path = require('path');
const { gzipSync } = require('zlib');

// Configuration
const MAX_BUNDLE_SIZE = 1.5 * 1024 * 1024; // 1.5MB in bytes
const MAX_CHUNK_SIZE = 500 * 1024; // 500KB per chunk
const DIST_DIR = path.join(__dirname, '..', 'dist');

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m'
};

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

function getFileSize(filePath) {
  try {
    const stats = fs.statSync(filePath);
    return stats.size;
  } catch {
    return 0;
  }
}

function getGzipSize(filePath) {
  try {
    const content = fs.readFileSync(filePath);
    const gzipped = gzipSync(content);
    return gzipped.length;
  } catch {
    return 0;
  }
}

function analyzeBundle() {
  console.log(`${colors.cyan}📦 Bundle Size Analysis${colors.reset}\n`);

  if (!fs.existsSync(DIST_DIR)) {
    console.error(`${colors.red}Error: dist directory not found. Run 'npm run build' first.${colors.reset}`);
    process.exit(1);
  }

  const files = [];
  let totalSize = 0;
  let totalGzipSize = 0;

  // Walk through dist directory
  function walkDir(dir) {
    const items = fs.readdirSync(dir);

    for (const item of items) {
      const fullPath = path.join(dir, item);
      const stat = fs.statSync(fullPath);

      if (stat.isDirectory()) {
        walkDir(fullPath);
      } else if (item.endsWith('.js') || item.endsWith('.css')) {
        const size = getFileSize(fullPath);
        const gzipSize = getGzipSize(fullPath);
        const relativePath = path.relative(DIST_DIR, fullPath);

        files.push({
          path: relativePath,
          size,
          gzipSize,
          isLarge: size > MAX_CHUNK_SIZE
        });

        totalSize += size;
        totalGzipSize += gzipSize;
      }
    }
  }

  walkDir(DIST_DIR);

  // Sort files by size
  files.sort((a, b) => b.size - a.size);

  // Display results
  console.log('Individual Files:\n');
  console.log('File'.padEnd(50) + 'Size'.padEnd(12) + 'Gzipped'.padEnd(12) + 'Status');
  console.log('-'.repeat(86));

  for (const file of files) {
    const status = file.isLarge
      ? `${colors.yellow}⚠ Large${colors.reset}`
      : `${colors.green}✓ OK${colors.reset}`;

    console.log(
      file.path.padEnd(50) +
      formatBytes(file.size).padEnd(12) +
      formatBytes(file.gzipSize).padEnd(12) +
      status
    );
  }

  console.log('-'.repeat(86));
  console.log('\n📊 Summary:\n');

  // Check HTML file
  const htmlPath = path.join(DIST_DIR, 'index.html');
  const htmlSize = getFileSize(htmlPath);

  console.log(`HTML Entry Point: ${formatBytes(htmlSize)}`);
  console.log(`Total JS/CSS Size: ${formatBytes(totalSize)}`);
  console.log(`Total Gzipped Size: ${formatBytes(totalGzipSize)}`);

  // Bundle analysis
  const fullBundleSize = totalSize + htmlSize;
  const percentage = ((fullBundleSize / MAX_BUNDLE_SIZE) * 100).toFixed(1);

  console.log(`\n📈 Bundle Size: ${formatBytes(fullBundleSize)} / ${formatBytes(MAX_BUNDLE_SIZE)} (${percentage}%)`);

  // Chunk analysis
  const chunks = files.filter(f => f.path.includes('assets/'));
  const vendorChunks = chunks.filter(f => f.path.includes('vendor') || f.path.includes('react'));
  const muiChunks = chunks.filter(f => f.path.includes('mui'));
  const iconChunks = chunks.filter(f => f.path.includes('icon'));
  const appChunks = chunks.filter(f => !f.path.includes('vendor') && !f.path.includes('mui') && !f.path.includes('icon'));

  console.log('\n📦 Chunk Breakdown:');
  console.log(`  • Vendor (React): ${formatBytes(vendorChunks.reduce((sum, f) => sum + f.size, 0))}`);
  console.log(`  • MUI Core: ${formatBytes(muiChunks.reduce((sum, f) => sum + f.size, 0))}`);
  console.log(`  • Icons: ${formatBytes(iconChunks.reduce((sum, f) => sum + f.size, 0))}`);
  console.log(`  • App Code: ${formatBytes(appChunks.reduce((sum, f) => sum + f.size, 0))}`);

  // Large files warning
  const largeFiles = files.filter(f => f.isLarge);
  if (largeFiles.length > 0) {
    console.log(`\n${colors.yellow}⚠ Warning: ${largeFiles.length} file(s) exceed ${formatBytes(MAX_CHUNK_SIZE)}${colors.reset}`);
    largeFiles.forEach(f => {
      console.log(`  • ${f.path}: ${formatBytes(f.size)}`);
    });
  }

  // Final verdict
  console.log('\n🎯 Result:');
  if (fullBundleSize <= MAX_BUNDLE_SIZE) {
    console.log(`${colors.green}✅ SUCCESS: Bundle size is under 1.5MB target!${colors.reset}`);

    // Performance tips
    console.log('\n💡 Performance Metrics:');
    console.log(`  • Initial Load: ~${(fullBundleSize / (1024 * 100)).toFixed(1)}s @ 100KB/s (3G)`);
    console.log(`  • Initial Load: ~${(fullBundleSize / (1024 * 1024)).toFixed(1)}s @ 1MB/s (4G)`);
    console.log(`  • Cache Hit: <100ms (after first load)`);

    return 0;
  } else {
    console.log(`${colors.red}❌ FAILURE: Bundle size (${formatBytes(fullBundleSize)}) exceeds 1.5MB limit!${colors.reset}`);

    // Optimization suggestions
    console.log('\n💡 Optimization Suggestions:');
    console.log('  1. Check for duplicate dependencies with npm ls --depth=0');
    console.log('  2. Use dynamic imports for large components');
    console.log('  3. Review @mui/icons-material usage');
    console.log('  4. Consider removing unused dependencies');
    console.log('  5. Enable additional compression in Vite config');

    return 1;
  }
}

// Run analysis
const exitCode = analyzeBundle();
process.exit(exitCode);
