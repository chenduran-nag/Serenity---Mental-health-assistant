import { useRef, useState } from 'react';
import PropTypes from 'prop-types';

import styles from './TextInput.module.css';

export default function TextInput({ onSend, disabled }) {
  const [value, setValue] = useState('');
  const textareaRef = useRef(null);

  function resizeTextarea(nextValue) {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }
    textarea.style.height = 'auto';
    textarea.value = nextValue;
    textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`;
  }

  function handleChange(event) {
    const nextValue = event.target.value;
    setValue(nextValue);
    resizeTextarea(nextValue);
  }

  function submit() {
    if (!value.trim() || disabled) {
      return;
    }
    onSend(value);
    setValue('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  }

  function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  }

  return (
    <div className={styles.shell}>
      <textarea
        className={styles.textarea}
        disabled={disabled}
        maxLength={1800}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        placeholder="Let the words rise slowly."
        ref={textareaRef}
        rows={1}
        value={value}
      />
      <button className={styles.sendButton} disabled={disabled} onClick={submit} type="button">
        ↗
      </button>
    </div>
  );
}

TextInput.propTypes = {
  disabled: PropTypes.bool.isRequired,
  onSend: PropTypes.func.isRequired,
};
