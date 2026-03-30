import { useEffect, useRef } from 'react';
import PropTypes from 'prop-types';

import styles from './ChatWindow.module.css';

function WaveIcon() {
  return (
    <svg aria-hidden="true" className={styles.waveIcon} viewBox="0 0 32 12">
      <path d="M1 6c4 0 4-4 8-4s4 4 8 4 4-4 8-4 4 4 7 4" fill="none" stroke="currentColor" strokeWidth="2" />
    </svg>
  );
}

export default function ChatWindow({ messages, isLoading }) {
  const scrollRef = useRef(null);

  useEffect(() => {
    const container = scrollRef.current;
    if (!container) {
      return;
    }
    container.scrollTop = container.scrollHeight;
  }, [messages, isLoading]);

  return (
    <div className={styles.shell}>
      <div className={styles.scrollRegion} ref={scrollRef}>
        {messages.length === 0 ? (
          <div className={styles.emptyState}>
            <p>There is room here for silence first.</p>
            <span>When you are ready, say what is weighing on you.</span>
          </div>
        ) : null}

        {messages.map((message) => (
          <article
            className={`${styles.message} ${message.role === 'user' ? styles.userMessage : styles.assistantMessage} ${message.crisisDetected ? styles.crisisMessage : ''}`}
            key={message.id}
          >
            {message.role === 'assistant' ? (
              <div className={styles.assistantHeader}>
                <WaveIcon />
                <span className={styles.assistantLabel}>Serenity</span>
              </div>
            ) : null}
            <p className={styles.content}>{message.content}</p>
            {message.crisisDetected ? <span className={styles.crisisTag}>⚠ crisis resources shown</span> : null}
          </article>
        ))}

        {isLoading ? (
          <div className={`${styles.message} ${styles.assistantMessage} ${styles.typingBubble}`}>
            <div className={styles.assistantHeader}>
              <WaveIcon />
              <span className={styles.assistantLabel}>Serenity</span>
            </div>
            <div className={styles.typingIndicator}>
              <span />
              <span />
              <span />
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}

ChatWindow.propTypes = {
  isLoading: PropTypes.bool.isRequired,
  messages: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      role: PropTypes.string.isRequired,
      content: PropTypes.string.isRequired,
      crisisDetected: PropTypes.bool,
      timestamp: PropTypes.string,
    }),
  ).isRequired,
};
