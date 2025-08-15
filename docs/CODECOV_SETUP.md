# Codecov Setup Instructions

## Setting up Codecov for PISAD

### 1. Create Codecov Account
1. Go to [codecov.io](https://codecov.io)
2. Sign in with your GitHub account
3. Grant access to your repository

### 2. Get Repository Token
1. Navigate to your repository in Codecov
2. Go to Settings → General
3. Copy the Repository Upload Token

### 3. Add Token to GitHub Secrets
1. Go to your GitHub repository
2. Navigate to Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Name: `CODECOV_TOKEN`
5. Value: Paste your Codecov token
6. Click "Add secret"

### 4. Update Badge URLs
Update the badge URLs in README.md:
1. Replace `yourusername` with your GitHub username
2. Replace `YOUR_TOKEN` with your actual Codecov token (only in the badge URL)

### 5. Verify Integration
1. Push a commit or create a pull request
2. Check GitHub Actions to ensure CI runs successfully
3. Visit your Codecov dashboard to see coverage reports

## Coverage Thresholds

The project enforces these minimum coverage thresholds:
- **Backend**: 65% (current: 67%)
- **Frontend**: 50% (current: 50%)
- **Patch coverage**: 60% for new code

These thresholds are appropriate for hardware-integrated systems where some code paths require physical hardware for testing.

## Running Coverage Locally

Generate a local coverage report:
```bash
./scripts/coverage-report.sh
```

This will:
- Run all tests with coverage
- Generate HTML reports
- Track coverage trends
- Show pass/fail for thresholds