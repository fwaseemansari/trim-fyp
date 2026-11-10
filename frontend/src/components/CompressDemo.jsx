import { useState } from 'react'
import { api } from '../api'

const SAMPLE = `Dustin Ellerman, a Texas shooter, wrapped raw bacon in foil around an M16 and fired rounds to cook it. The process, filmed on his YouTube channel, gained nearly 300,000 views as the gun's heat turned the bacon crisp. After the sizzling shoot, he unveiled the cooked bacon and joked about killing more pigs to the laughter of his friends. Firearms experts noted the stunt was unusual even by viral video standards. Local gun ranges said they had not seen a similar request before.`

export default function CompressDemo() {
  const [text, setText] = useState(SAMPLE)
  const [query, setQuery] = useState('')
  const [scorer, setScorer] = useState('cross_encoder')
  const [level, setLevel] = useState(50)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  async function run() {
    setLoading(true)
    setError(null)
    try {
      const r = await api.compress({
        text, query, method: 'selective', scorer, level: level / 100, backend: 'groq',
      })
      setResult(r)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <section id="demo">
      <div className="wrap">
        <div className="eyebrow">Interactive · live API call</div>
        <h2>Try the compression module</h2>
        <p style={{ color: 'var(--ink-dim)', maxWidth: 640 }}>
          This runs the real <code>SelectiveCompressor</code> from <code>prompt_compression/compressor.py</code>,
          via the FastAPI backend, no key needed for this step.
        </p>
        <div className="demo-grid">
          <div>
            <textarea value={text} onChange={e => setText(e.target.value)} />
            <div className="controls">
              <input type="text" placeholder="e.g. did any Nazis flee to Paraguay?"
                value={query} onChange={e => setQuery(e.target.value)} style={{ flex: 1, minWidth: 220 }} />
              <select value={scorer} onChange={e => setScorer(e.target.value)}>
                <option value="tfidf">TF-IDF</option>
                <option value="bi_encoder">Bi-encoder</option>
                <option value="cross_encoder">Cross-encoder</option>
              </select>
              <div className="level-row">
                Keep {level}%
                <input type="range" min="20" max="90" value={level} onChange={e => setLevel(+e.target.value)} />
              </div>
              <button className="primary" onClick={run} disabled={loading}>
                {loading ? 'Compressing…' : 'Compress'}
              </button>
            </div>
            {error && <div className="error-box">Backend error: {error}. Is <code>uvicorn backend.main:app</code> running on port 8000?</div>}
            <div className="output">{result ? result.text : 'Run it to see the compressed output.'}</div>
            {result && (
              <div className="stat-row">
                <div className="stat">{result.original_tokens} → {result.compressed_tokens} tokens</div>
                <div className="stat">{result.token_reduction_pct}% reduction</div>
                <div className="stat">{result.segment_count} segments scored</div>
              </div>
            )}
          </div>
          <div className="panel">
            <h3>WHY THE QUERY FIELD MATTERS</h3>
            <p style={{ fontSize: '.88rem', color: 'var(--ink-dim)' }}>
              Every scorer ranks sentences by comparing them against your query. Leave it blank
              and every sentence scores the same, so the "smart" scorer can't actually tell
              sentences apart, it just falls back to document order. Add a real question and
              the scorer can tell "this sentence has the answer" from "this one doesn't."
            </p>
            <p style={{ fontSize: '.88rem', color: 'var(--ink-dim)' }}>
              Try it: run once with the query empty, then again with the sample question above,
              same text, same scorer, different sentences kept.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
