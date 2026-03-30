import PropTypes from 'prop-types';

import styles from './ModeSelector.module.css';

const TITLE = 'serenity'.split('');

export default function ModeSelector({ onSelectMode }) {
  return (
    <section className={styles.shell}>
      <p className={styles.kicker}>a space to breathe</p>
      <h1 className={styles.title} aria-label="serenity">
        {TITLE.map((letter, index) => (
          <span
            className={styles.letter}
            key={`${letter}-${index}`}
            style={{ animationDelay: `${index * 110}ms` }}
          >
            {letter}
          </span>
        ))}
      </h1>
      <p className={styles.description}>
        Drift into a quieter conversation, in text or by voice, with a local model beneath the surface.
      </p>
      <div className={styles.actions}>
        <button className={styles.actionButton} onClick={() => onSelectMode('text')} type="button">
          <span className={styles.ripple} />
          <span className={styles.label}>💬 Text</span>
        </button>
        <button className={styles.actionButton} onClick={() => onSelectMode('voice')} type="button">
          <span className={styles.ripple} />
          <span className={styles.label}>🎙 Voice</span>
        </button>
      </div>
    </section>
  );
}

ModeSelector.propTypes = {
  onSelectMode: PropTypes.func.isRequired,
};
