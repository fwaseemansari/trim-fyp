const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Request failed')
  }
  return res.json()
}

export const api = {
  health: () => fetch(`${BASE}/api/health`).then(r => r.json()),
  tokenize: (text, model) => post('/api/tokenize', { text, model }),
  compress: (payload) => post('/api/compress', payload),
  context: (payload) => post('/api/context', payload),
}
