const BACKEND_URL = (import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

async function unwrapJson(response) {
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}.`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      detail = detail;
    }
    throw new Error(detail);
  }
  return response.json();
}

export async function sendTextChat({ sessionId, message, history }) {
  const response = await fetch(`${BACKEND_URL}/chat/text`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Session-Id': sessionId,
    },
    body: JSON.stringify({
      session_id: sessionId,
      message,
      history,
    }),
  });
  return unwrapJson(response);
}

export async function sendVoiceChat({ audioBlob, sessionId }) {
  const formData = new FormData();
  formData.append('audio_file', audioBlob, 'serenity-voice.webm');
  formData.append('session_id', sessionId);

  const response = await fetch(`${BACKEND_URL}/chat/voice`, {
    method: 'POST',
    headers: {
      'X-Session-Id': sessionId,
    },
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Voice request failed with status ${response.status}.`);
  }

  return {
    audioBlob: await response.blob(),
    transcript: response.headers.get('X-Transcript') || '',
    reply: response.headers.get('X-Reply-Text') || '',
    sessionId: response.headers.get('X-Session-Id') || sessionId,
    crisisDetected: response.headers.get('X-Crisis-Detected') === 'true',
    timestamp: new Date().toISOString(),
  };
}

export async function fetchSessionHistory(sessionId) {
  const response = await fetch(`${BACKEND_URL}/chat/history/${sessionId}`);
  return unwrapJson(response);
}
