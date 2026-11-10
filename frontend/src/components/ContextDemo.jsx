import { useState } from 'react'
import { api } from '../api'

const SEED = [
  { speaker: 'User', text: 'What is the capital of Paraguay?' },
  { speaker: 'Assistant', text: 'Asunción is the capital of Paraguay.' },
  { speaker: 'User', text: 'What is the weather like there in July?' },
  { speaker: 'Assistant', text: 'July is winter there, mild and dry, around 15-20°C.' },
  { speaker: 'User', text: 'Tell me about its history with dictatorships.' },
  { speaker: 'Assistant', text: 'Paraguay had a long dictatorship under Alfredo Stroessner from 1954 to 1989.' },
  { speaker: 'User', text: 'Did any Nazis flee there after World War II?' },
]

export default function ContextDemo() {
  const [turns, setTurns] = useState(SEED)
  const [strategy, setStrategy] = useState('sliding_window')
  const [maxTurns, setMaxTurns] = useState(4)
  const [maxTokens, setMaxTokens] = useState(60)
  const [query, setQuery] = useState('Nazis fleeing to Paraguay')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  function updateTurn(i, field, val) {
    const next = [...turns]
    next[i] = { ...next[i], [field]: val }
    setTurns(next)
  }

  async function run() {
    setLoading(true)
    setError(null)
    try {
      const r = await api.context({ turns, strategy, max_turns: maxTurns, max_tokens: maxTokens, query })
      setResult(r)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <section id="context">
      <div className="wrap">
        <div className="eyebrow">Interactive · live API call</div>
        <h2>Try the Adaptive Context Manager</h2>
        <p style={{ color: 'var(--ink-dim)', maxWidth: 640 }}>
          Runs <code>ContextManager</code> from <code>context_manager/manager.py</code> on this
          conversation, sliding-window keeps the last N turns, relevance-aware scores every turn
          against your query and keeps the most relevant ones until the token budget runs out.
        </p>
        <div className="demo-grid">
          <div className="panel">
            <h3>CONVERSATION</h3>
            {turns.map((t, i) => (
              <div className="chat-row" key={i}>
                <select value={t.speaker} onChange={e => updateTurn(i, 'speaker', e.target.value)} style={{ width: 90 }}>
                  <option>User</option>
                  <option>Assistant</option>
                </select>
                <input type="text" value={t.text} onChange={e => updateTurn(i, 'text', e.target.value)} />
                {result && (
                  <span className={`kept-tag ${result.turns[i]?.kept ? 'yes' : 'no'}`}>
                    {result.turns[i]?.kept ? 'kept' : 'dropped'}
                  </span>
                )}
              </div>
            ))}
            <div className="controls">
              <select value={strategy} onChange={e => setStrategy(e.target.value)}>
                <option value="sliding_window">Sliding window</option>
                <option value="relevance_aware">Relevance aware</option>
                <option value="hybrid_scoring">Hybrid (relevance + recency)</option>
              </select>
              {strategy === 'sliding_window' ? (
                <div className="level-row">max turns <input type="range" min="1" max="7" value={maxTurns} onChange={e => setMaxTurns(+e.target.value)} /> {maxTurns}</div>
              ) : (
                <div className="level-row">max tokens <input type="range" min="20" max="200" step="10" value={maxTokens} onChange={e => setMaxTokens(+e.target.value)} /> {maxTokens}</div>
              )}
            </div>
            {strategy !== 'sliding_window' && (
              <input type="text" value={query} onChange={e => setQuery(e.target.value)}
                placeholder="current query to score relevance against" style={{ width: '100%' }} />
            )}
            <div className="controls">
              <button className="primary" onClick={run} disabled={loading}>
                {loading ? 'Running…' : 'Run context manager'}
              </button>
            </div>
            {error && <div className="error-box">Backend error: {error}</div>}
          </div>
          <div className="panel">
            <h3>RESULTING CONTEXT SENT TO THE LLM</h3>
            <div className="output">{result ? result.context : 'Run it to see which turns survive.'}</div>
            {result && (
              <div className="stat-row">
                <div className="stat">{result.kept_count} kept</div>
                <div className="stat">{result.dropped_count} dropped</div>
                <div className="stat">{result.total_tokens} tokens in context</div>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  )
}
