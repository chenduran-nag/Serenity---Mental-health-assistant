import PropTypes from 'prop-types';

import styles from './SessionSidebar.module.css';

function formatTimestamp(timestamp) {
  return new Date(timestamp).toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

export default function SessionSidebar({
  sessions,
  currentSessionId,
  isOpen,
  onLoadSession,
  onNewSession,
  onToggle,
}) {
  return (
    <aside className={`${styles.sidebar} ${isOpen ? styles.open : styles.closed}`}>
      <button className={styles.toggle} onClick={onToggle} type="button">
        {isOpen ? 'Hide' : 'Show'}
      </button>
      <div className={styles.content}>
        <h2 className={styles.heading}>Currents</h2>
        <button className={styles.newSessionButton} onClick={onNewSession} type="button">
          New Session
        </button>
        <div className={styles.sessionList}>
          {sessions.map((session) => (
            <button
              className={`${styles.sessionCard} ${session.sessionId === currentSessionId ? styles.active : ''}`}
              key={session.sessionId}
              onClick={() => onLoadSession(session.sessionId)}
              type="button"
            >
              <strong>{session.preview || 'Untitled session'}</strong>
              <span>{formatTimestamp(session.timestamp)}</span>
            </button>
          ))}
        </div>
      </div>
    </aside>
  );
}

SessionSidebar.propTypes = {
  currentSessionId: PropTypes.string.isRequired,
  isOpen: PropTypes.bool.isRequired,
  onLoadSession: PropTypes.func.isRequired,
  onNewSession: PropTypes.func.isRequired,
  onToggle: PropTypes.func.isRequired,
  sessions: PropTypes.arrayOf(
    PropTypes.shape({
      sessionId: PropTypes.string.isRequired,
      preview: PropTypes.string.isRequired,
      timestamp: PropTypes.string.isRequired,
    }),
  ).isRequired,
};
