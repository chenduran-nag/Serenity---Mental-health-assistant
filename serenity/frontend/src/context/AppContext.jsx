import { createContext, startTransition, useContext, useEffect, useState } from 'react';
import PropTypes from 'prop-types';

const RECENT_SESSIONS_KEY = 'serenity_recent_sessions';

function createSessionId() {
  return crypto.randomUUID();
}

function loadRecentSessions() {
  try {
    const raw = window.localStorage.getItem(RECENT_SESSIONS_KEY);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

const AppContext = createContext(null);

export function AppProvider({ children }) {
  const [mode, setMode] = useState(null);
  const [sessionId, setSessionId] = useState(() => createSessionId());
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [crisisDetected, setCrisisDetected] = useState(false);
  const [pulseSeed, setPulseSeed] = useState(0);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [sessions, setSessions] = useState(() => loadRecentSessions());

  useEffect(() => {
    window.localStorage.setItem(RECENT_SESSIONS_KEY, JSON.stringify(sessions.slice(0, 5)));
  }, [sessions]);

  function registerSession(nextSessionId, preview, timestamp) {
    // A missing id used to be stored anyway, leaving an entry that could never
    // be reloaded because it fetched /chat/history/undefined.
    if (!nextSessionId) {
      return;
    }

    setSessions((currentSessions) => {
      const nextSessions = currentSessions.filter((session) => session.sessionId !== nextSessionId);
      nextSessions.unshift({
        sessionId: nextSessionId,
        preview: preview.slice(0, 52),
        timestamp,
      });
      return nextSessions.slice(0, 5);
    });
  }

  function beginNewSession() {
    const freshSessionId = createSessionId();
    startTransition(() => {
      setSessionId(freshSessionId);
      setMessages([]);
      setMode((currentMode) => currentMode || 'text');
      setCrisisDetected(false);
    });
    registerSession(freshSessionId, 'New session', new Date().toISOString());
  }

  function triggerPulse() {
    setPulseSeed((seed) => seed + 1);
  }

  const value = {
    mode,
    setMode,
    sessionId,
    setSessionId,
    messages,
    setMessages,
    isLoading,
    setIsLoading,
    crisisDetected,
    setCrisisDetected,
    pulseSeed,
    triggerPulse,
    sessions,
    registerSession,
    beginNewSession,
    sidebarOpen,
    setSidebarOpen,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

AppProvider.propTypes = {
  children: PropTypes.node.isRequired,
};

export function useAppContext() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useAppContext must be used within an AppProvider.');
  }
  return context;
}
