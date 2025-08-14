/**
 * Production-ready logger utility for frontend application.
 * Provides controlled logging with environment-based levels.
 */

export const LogLevel = {
  DEBUG: 0,
  INFO: 1,
  WARN: 2,
  ERROR: 3,
  NONE: 4,
} as const;

export type LogLevel = (typeof LogLevel)[keyof typeof LogLevel];

interface LoggerConfig {
  level: LogLevel;
  prefix?: string;
  enableTimestamp?: boolean;
}

class Logger {
  private level: LogLevel;
  private prefix: string;
  private enableTimestamp: boolean;

  constructor(config?: Partial<LoggerConfig>) {
    // In production, default to ERROR level only
    const isDevelopment = process.env.NODE_ENV === "development";
    this.level = config?.level ?? (isDevelopment ? LogLevel.DEBUG : LogLevel.ERROR);
    this.prefix = config?.prefix ?? "[PISAD]";
    this.enableTimestamp = config?.enableTimestamp ?? true;
  }

  private formatMessage(level: string, message: string): string {
    const timestamp = this.enableTimestamp
      ? new Date().toISOString()
      : "";
    const parts = [
      this.prefix,
      timestamp && `[${timestamp}]`,
      `[${level}]`,
      message,
    ].filter(Boolean);
    return parts.join(" ");
  }

  debug(message: string, ...args: any[]): void {
    if (this.level <= LogLevel.DEBUG) {
      console.log(this.formatMessage("DEBUG", message), ...args);
    }
  }

  info(message: string, ...args: any[]): void {
    if (this.level <= LogLevel.INFO) {
      console.info(this.formatMessage("INFO", message), ...args);
    }
  }

  warn(message: string, ...args: any[]): void {
    if (this.level <= LogLevel.WARN) {
      console.warn(this.formatMessage("WARN", message), ...args);
    }
  }

  error(message: string, error?: Error | unknown, ...args: any[]): void {
    if (this.level <= LogLevel.ERROR) {
      const errorMessage = error instanceof Error
        ? `${message}: ${error.message}`
        : message;
      console.error(this.formatMessage("ERROR", errorMessage), error, ...args);
    }
  }

  // Create a child logger with a specific context
  createChild(context: string): Logger {
    return new Logger({
      level: this.level,
      prefix: `${this.prefix}[${context}]`,
      enableTimestamp: this.enableTimestamp,
    });
  }
}

// Export singleton instance
export const logger = new Logger();

// Export factory for component-specific loggers
export const createLogger = (context: string): Logger => {
  return logger.createChild(context);
};

export default logger;