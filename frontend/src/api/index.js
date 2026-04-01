/** HiveResearch API client */

const BASE = ''

async function request(method, path, body = null) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
  }
  if (body) opts.body = JSON.stringify(body)
  const res = await fetch(`${BASE}${path}`, opts)
  if (!res.ok) throw new Error(`${method} ${path}: ${res.status}`)
  return res.json()
}

export const api = {
  // Sessions
  startResearch: (config) => request('POST', '/research', config),
  getSession: (id) => request('GET', `/session/${id}`),
  getSessionStatus: (id) => request('GET', `/session/${id}/status`),
  getDag: (id) => request('GET', `/session/${id}/dag`),
  getSummary: (id) => request('GET', `/session/${id}/summary`),
  stopSession: (id) => request('POST', `/session/${id}/stop`),
  pauseSession: (id) => request('POST', `/session/${id}/pause`),
  resumeSession: (id) => request('POST', `/session/${id}/resume`),

  // SSE stream
  streamUrl: (id) => `${BASE}/session/${id}/stream`,

  // Agent inspector
  inspectAgent: (sessionId, agentId) =>
    request('GET', `/session/${sessionId}/agent/${agentId}`),

  // Personas
  getTemplates: () => request('GET', '/personas/templates'),
  getPersona: (sessionId, agentId) =>
    request('GET', `/session/${sessionId}/agent/${agentId}/persona`),
  setPersona: (sessionId, agentId, data) =>
    request('POST', `/session/${sessionId}/agent/${agentId}/persona`, data),

  // Interview
  interviewAgent: (sessionId, data) =>
    request('POST', `/session/${sessionId}/interview`, data),
  interviewBatch: (sessionId, data) =>
    request('POST', `/session/${sessionId}/interview/batch`, data),
  interviewAll: (sessionId, prompt) =>
    request('POST', `/session/${sessionId}/interview/all?prompt=${encodeURIComponent(prompt)}`),

  // Report
  planReport: (sessionId) => request('POST', `/session/${sessionId}/report/plan`),
  getReport: (sessionId) => request('GET', `/session/${sessionId}/report`),

  // Audit
  getAuditLog: (sessionId, limit = 100) =>
    request('GET', `/session/${sessionId}/audit?limit=${limit}`),

  // Health
  health: () => request('GET', '/health'),
}
