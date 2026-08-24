const BACKEND_URL = (import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

async function unwrapJson(response) {
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}.`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      // response body was not JSON; keep the status-based detail
    }
    throw new Error(detail);
  }
  return response.json();
}

function base64ToBlob(base64, mediaType) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new Blob([bytes], { type: mediaType || 'audio/wav' });
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

  const data = await unwrapJson(response);

  // The API speaks snake_case and the app speaks camelCase. Returning the raw
  // body here meant response.crisisDetected was always undefined, so the crisis
  // modal never opened, and response.sessionId was undefined, which wiped the
  // session id after the first message.
  return {
    reply: data.reply,
    sessionId: data.session_id,
    timestamp: data.timestamp,
    crisisDetected: Boolean(data.crisis_detected),
    crisisMessage: data.crisis_message ?? null,
  };
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

  const data = await unwrapJson(response);

  return {
    audioBlob: base64ToBlob(data.audio_base64, data.audio_media_type),
    transcript: data.transcript || '',
    reply: data.reply || '',
    sessionId: data.session_id || sessionId,
    crisisDetected: Boolean(data.crisis_detected),
    timestamp: data.timestamp || new Date().toISOString(),
  };
}

export async function fetchSessionHistory(sessionId) {
  const response = await fetch(`${BACKEND_URL}/chat/history/${sessionId}`);
  return unwrapJson(response);
}
