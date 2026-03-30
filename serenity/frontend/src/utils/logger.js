export function createLogger(scope) {
  function formatMessage(message) {
    return `[${scope}] ${message}`;
  }

  return {
    info(message, meta) {
      if (import.meta.env.DEV) {
        console.info(formatMessage(message), meta);
      }
    },
    warn(message, meta) {
      console.warn(formatMessage(message), meta);
    },
    error(message, meta) {
      console.error(formatMessage(message), meta);
    },
  };
}
