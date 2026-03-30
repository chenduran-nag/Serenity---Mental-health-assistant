import { useRef, useState } from 'react';
import PropTypes from 'prop-types';

import { sendVoiceChat } from '../api/client';
import { createLogger } from '../utils/logger';
import styles from './VoiceInterface.module.css';

const logger = createLogger('VoiceInterface');

function createAudioContext() {
  return new window.AudioContext();
}

export default function VoiceInterface({ sessionId, onVoiceExchange, disabled }) {
  const [voiceState, setVoiceState] = useState('idle');
  const [transcript, setTranscript] = useState('');
  const [reply, setReply] = useState('');
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const audioContextRef = useRef(null);

  let caption = 'Tap to begin';
  if (voiceState === 'listening') {
    caption = 'Listening...';
  } else if (voiceState === 'processing') {
    caption = 'Thinking...';
  } else if (voiceState === 'speaking') {
    caption = 'Speaking...';
  }

  async function playAudioBlob(audioBlob) {
    if (!audioContextRef.current) {
      audioContextRef.current = createAudioContext();
    }
    const context = audioContextRef.current;
    if (context.state === 'suspended') {
      await context.resume();
    }

    const arrayBuffer = await audioBlob.arrayBuffer();
    const audioBuffer = await context.decodeAudioData(arrayBuffer);
    const source = context.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(context.destination);
    source.start();

    await new Promise((resolve) => {
      source.onended = () => resolve();
    });
  }

  async function submitRecording(blob) {
    setVoiceState('processing');
    try {
      const exchange = await sendVoiceChat({ audioBlob: blob, sessionId });
      setTranscript(exchange.transcript);
      setReply(exchange.reply);
      onVoiceExchange(exchange);
      setVoiceState('speaking');
      await playAudioBlob(exchange.audioBlob);
      setVoiceState('idle');
    } catch (error) {
      logger.error('Voice round-trip failed.', error);
      setReply('The current shifted before I could answer. Please try again.');
      setVoiceState('idle');
    }
  }

  async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    chunksRef.current = [];
    mediaRecorderRef.current = recorder;
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        chunksRef.current.push(event.data);
      }
    };
    recorder.onstop = async () => {
      const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
      stream.getTracks().forEach((track) => track.stop());
      await submitRecording(blob);
    };
    recorder.start();
    setTranscript('');
    setReply('');
    setVoiceState('listening');
  }

  async function toggleRecording() {
    if (disabled || voiceState === 'processing' || voiceState === 'speaking') {
      return;
    }

    if (voiceState === 'listening' && mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      return;
    }

    try {
      await startRecording();
    } catch (error) {
      logger.error('Microphone access failed.', error);
      setReply('Microphone access was unavailable. Please check browser permissions.');
      setVoiceState('idle');
    }
  }

  return (
    <div className={styles.shell}>
      <button className={`${styles.orb} ${styles[voiceState]}`} disabled={disabled} onClick={toggleRecording} type="button">
        <span className={styles.core} />
        <span className={styles.ring} />
        <span className={styles.ring} />
      </button>
      <p className={styles.caption}>{caption}</p>
      <div className={styles.transcriptPanel}>
        <div>
          <span className={styles.label}>You said</span>
          <p>{transcript || 'Your voice will appear here once the water settles.'}</p>
        </div>
        <div>
          <span className={styles.label}>Serenity replied</span>
          <p>{reply || 'The reply text will glow here while the audio plays.'}</p>
        </div>
      </div>
    </div>
  );
}

VoiceInterface.propTypes = {
  disabled: PropTypes.bool.isRequired,
  onVoiceExchange: PropTypes.func.isRequired,
  sessionId: PropTypes.string.isRequired,
};
