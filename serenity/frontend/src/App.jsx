import { startTransition } from 'react';

import styles from './App.module.css';
import { fetchSessionHistory, sendTextChat } from './api/client';
import ChatWindow from './components/ChatWindow';
import CrisisModal from './components/CrisisModal';
import ModeSelector from './components/ModeSelector';
import ParticleCanvas from './components/ParticleCanvas';
import SessionSidebar from './components/SessionSidebar';
import TextInput from './components/TextInput';
import VoiceInterface from './components/VoiceInterface';
import { useAppContext } from './context/AppContext';
import { createLogger } from './utils/logger';

const logger = createLogger('App');

function buildMessage(role, content, options = {}) {
  return {
    id: `${role}-${crypto.randomUUID()}`,
    role,
    content,
    crisisDetected: Boolean(options.crisisDetected),
    timestamp: options.timestamp || new Date().toISOString(),
  };
}

export default function App() {
  const {
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
    sidebarOpen,
    setSidebarOpen,
    registerSession,
    beginNewSession,
  } = useAppContext();

  async function handleSendMessage(rawMessage) {
    const message = rawMessage.trim();
    if (!message || isLoading) {
      return;
    }

    const historySnapshot = messages.map(({ role, content }) => ({ role, content }));
    const userMessage = buildMessage('user', message);
    startTransition(() => {
      setMessages((currentMessages) => [...currentMessages, userMessage]);
      setCrisisDetected(false);
    });
    setIsLoading(true);
    registerSession(sessionId, message, userMessage.timestamp);

    try {
      const response = await sendTextChat({
        sessionId,
        message,
        history: historySnapshot,
      });
      const assistantMessage = buildMessage('assistant', response.reply, {
        crisisDetected: response.crisisDetected,
        timestamp: response.timestamp,
      });

      startTransition(() => {
        setSessionId(response.sessionId);
        setMessages((currentMessages) => [...currentMessages, assistantMessage]);
        setCrisisDetected(response.crisisDetected);
      });
      registerSession(response.sessionId, message, response.timestamp);
      triggerPulse();
    } catch (error) {
      logger.error('Text chat failed.', error);
      startTransition(() => {
        setMessages((currentMessages) => [
          ...currentMessages,
          buildMessage(
            'assistant',
            'The water is a little turbulent right now. Please try again in a moment.',
          ),
        ]);
      });
    } finally {
      setIsLoading(false);
    }
  }

  async function handleLoadSession(targetSessionId) {
    if (isLoading) {
      return;
    }

    setIsLoading(true);
    try {
      const data = await fetchSessionHistory(targetSessionId);
      const restoredMessages = data.messages.map((message) =>
        buildMessage(message.role, message.content, {
          crisisDetected: Boolean(message.crisis_detected),
          timestamp: message.timestamp,
        }),
      );

      startTransition(() => {
        setMode((currentMode) => currentMode || 'text');
        setSessionId(targetSessionId);
        setMessages(restoredMessages);
        setCrisisDetected(restoredMessages.some((message) => message.crisisDetected));
      });
      triggerPulse();
    } catch (error) {
      logger.error('Failed to load session.', error);
    } finally {
      setIsLoading(false);
    }
  }

  function handleModeSelect(nextMode) {
    startTransition(() => {
      setMode(nextMode);
      setSidebarOpen(true);
    });
  }

  function handleVoiceExchange(exchange) {
    const userMessage = buildMessage('user', exchange.transcript, {
      timestamp: exchange.timestamp,
    });
    const assistantMessage = buildMessage('assistant', exchange.reply, {
      crisisDetected: exchange.crisisDetected,
      timestamp: exchange.timestamp,
    });

    startTransition(() => {
      setSessionId(exchange.sessionId);
      setMessages((currentMessages) => [...currentMessages, userMessage, assistantMessage]);
      setCrisisDetected(exchange.crisisDetected);
    });
    registerSession(exchange.sessionId, exchange.transcript, exchange.timestamp);
    triggerPulse();
  }

  return (
    <div className={styles.appShell}>
      <ParticleCanvas pulseSeed={pulseSeed} />
      <div className={styles.noiseLayer} />
      <div className={styles.pageGlow} />

      {!mode ? (
        <ModeSelector onSelectMode={handleModeSelect} />
      ) : (
        <div className={styles.layout}>
          <SessionSidebar
            currentSessionId={sessionId}
            isOpen={sidebarOpen}
            sessions={sessions}
            onLoadSession={handleLoadSession}
            onNewSession={() => {
              beginNewSession();
              setCrisisDetected(false);
            }}
            onToggle={() => setSidebarOpen((open) => !open)}
          />

          <main className={styles.stage}>
            <header className={styles.header}>
              <div>
                <p className={styles.eyebrow}>Local mental health support</p>
                <h1 className={styles.title}>serenity</h1>
              </div>
              <button className={styles.modePill} onClick={() => setMode(mode === 'text' ? 'voice' : 'text')} type="button">
                {mode === 'text' ? 'Switch to voice' : 'Switch to text'}
              </button>
            </header>

            <section className={styles.chatPanel}>
              <ChatWindow isLoading={isLoading} messages={messages} />
            </section>

            <section className={styles.interactionPanel}>
              {mode === 'text' ? (
                <TextInput disabled={isLoading} onSend={handleSendMessage} />
              ) : (
                <VoiceInterface
                  disabled={isLoading}
                  onVoiceExchange={handleVoiceExchange}
                  sessionId={sessionId}
                />
              )}
            </section>
          </main>
        </div>
      )}

      <CrisisModal
        onDismiss={() => setCrisisDetected(false)}
        open={crisisDetected}
      />
    </div>
  );
}
