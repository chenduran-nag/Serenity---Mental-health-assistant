import PropTypes from 'prop-types';

import styles from './CrisisModal.module.css';

export default function CrisisModal({ open, onDismiss }) {
  if (!open) {
    return null;
  }

  return (
    <div className={styles.overlay}>
      <div className={styles.modal}>
        <p className={styles.kicker}>Immediate support</p>
        <h2>You do not have to hold this alone.</h2>
        <p>
          Please reach out to iCall India at <strong>9152987821</strong> or the Vandrevala Foundation at{' '}
          <strong>1860-2662-345</strong> right now.
        </p>
        <p className={styles.secondary}>
          If you are in immediate danger, contact your local emergency services or someone physically near you.
        </p>
        <button className={styles.dismissButton} onClick={onDismiss} type="button">
          I&apos;m safe, continue
        </button>
      </div>
    </div>
  );
}

CrisisModal.propTypes = {
  onDismiss: PropTypes.func.isRequired,
  open: PropTypes.bool.isRequired,
};
