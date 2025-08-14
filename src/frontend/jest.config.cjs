module.exports = {
  preset: "ts-jest",
  testEnvironment: "jsdom",
  roots: ["<rootDir>/../../tests/frontend"],
  collectCoverageFrom: [
    "src/**/*.{ts,tsx}",
    "!src/**/*.d.ts",
    "!src/main.tsx",
    "!src/vite-env.d.ts",
    "!src/**/*.test.{ts,tsx}",
    "!src/**/*.spec.{ts,tsx}",
  ],
  coverageDirectory: "<rootDir>/coverage",
  coverageReporters: ["text", "text-summary", "html", "lcov", "json"],
  coverageThreshold: {
    global: {
      branches: 60,
      functions: 60,
      lines: 60,
      statements: 60,
    },
  },
  setupFilesAfterEnv: ["<rootDir>/jest.setup.js"],
  moduleNameMapper: {
    "\\.(css|less|scss|sass)$": "identity-obj-proxy",
    "^@/(.*)$": "<rootDir>/src/$1",
    "^react$": "<rootDir>/node_modules/react",
    "^react-dom$": "<rootDir>/node_modules/react-dom",
    "^@testing-library/react$": "<rootDir>/node_modules/@testing-library/react",
    "^@testing-library/jest-dom$":
      "<rootDir>/node_modules/@testing-library/jest-dom",
  },
  transform: {
    "^.+\\.tsx?$": [
      "ts-jest",
      {
        tsconfig: {
          jsx: "react-jsx",
          esModuleInterop: true,
          allowSyntheticDefaultImports: true,
          moduleResolution: "node",
          isolatedModules: true,
        },
      },
    ],
  },
  testMatch: [
    "**/__tests__/**/*.+(ts|tsx|js)",
    "**/?(*.)+(spec|test).+(ts|tsx|js)",
  ],
  moduleFileExtensions: ["ts", "tsx", "js", "jsx", "json", "node"],
  moduleDirectories: ["node_modules", "<rootDir>/node_modules"],
  transformIgnorePatterns: ["node_modules/(?!(@testing-library)/)"],
  testTimeout: 10000, // 10 second timeout for tests
  maxWorkers: 2, // Limit parallel test execution to prevent resource issues
};
