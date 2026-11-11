import { useEffect, useRef, useState } from 'react'

const BARS = [
  { label: 'Lead-N', value: 0.552, pct: 83.8 },
  { label: 'Cross-encoder', value: 0.469, pct: 71.2 },
  { label: 'TF-IDF', value: 0.469, pct: 71.2 },
  { label: 'Bi-encoder', value: 0.432, pct: 65.6 },
]

// End-to-end ROUGE-L F1 change vs the no-compression baseline, 1,000-article
// CNN/DailyMail subset. Positive = scored higher than sending the full article.
const CNN_DELTAS = [
  { label: 'Lead-N (0.3)', delta: +0.017 },
  { label: 'Lead-N (0.5)', delta: +0.015 },
  { label: 'Lead-N (0.7)', delta: +0.006 },
  { label: 'Cross-encoder (0.7)', delta: -0.010 },
  { label: 'TF-IDF (0.5)', delta: -0.019 },
  { label: 'Cross-encoder (0.5)', delta: -0.021 },
  { label: 'Bi-encoder (0.5)', delta: -0.035 },
  { label: 'Cross-encoder (0.3)', delta: -0.038 },
]
const MAX_ABS_DELTA = Math.max(...CNN_DELTAS.map(d => Math.abs(d.delta)))

export default function ResultsDashboard() {
  const [animate, setAnimate] = useState(false)
  const ref = useRef(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const obs = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) { setAnimate(true); obs.disconnect() }
    }, { threshold: 0.3 })
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  return (
    <section id="results">
      <div className="wrap">
        <div className="eyebrow">Evidence</div>
        <h2>What the benchmarks actually show</h2>
        <p style={{ color: 'var(--ink-dim)', maxWidth: 640 }}>
          From the full CNN/DailyMail encoder-stage run (13,368 articles), comparing how much of the
          reference summary survives at 50% compression.
        </p>
        <div style={{ marginTop: 24 }} ref={ref}>
          <div style={{ fontSize: '.9rem', color: 'var(--ink-dim)', marginBottom: 10, fontWeight: 600 }}>
            ROUGE-L recall retained at 50% token budget
          </div>
          {BARS.map((b, i) => (
            <div className="bar-row" key={b.label}>
              <div className="bar-label">{b.label}</div>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: animate ? `${b.pct}%` : '0%', transitionDelay: `${i * 90}ms` }}>
                  {animate ? b.value : ''}
                </div>
              </div>
            </div>
          ))}
        </div>
        <div style={{ marginTop: 36 }}>
          <div style={{ fontSize: '.9rem', color: 'var(--ink-dim)', marginBottom: 10, fontWeight: 600 }}>
            CNN/DailyMail: end-to-end ROUGE-L F1 change vs. no-compression baseline (1,000 articles)
          </div>
          {CNN_DELTAS.map(d => {
            const widthPct = (Math.abs(d.delta) / MAX_ABS_DELTA) * 50 // half-width since it's diverging
            const positive = d.delta > 0
            return (
              <div key={d.label} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 7, fontSize: '.82rem' }}>
                <div style={{ width: 140, flexShrink: 0, textAlign: 'right' }}>{d.label}</div>
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', height: 18, position: 'relative', background: 'var(--line)', borderRadius: 6 }}>
                  <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: 1, background: 'var(--ink-dim)', opacity: 0.4 }} />
                  <div
                    style={{
                      position: 'absolute',
                      height: '100%',
                      borderRadius: 6,
                      width: `${widthPct}%`,
                      left: positive ? '50%' : `${50 - widthPct}%`,
                      background: positive ? 'var(--accent)' : 'var(--accent-2)',
                    }}
                  />
                </div>
                <div style={{ width: 54, fontWeight: 700, color: positive ? 'var(--accent)' : 'var(--accent-2)' }}>
                  {positive ? '+' : ''}{d.delta.toFixed(3)}
                </div>
              </div>
            )
          })}
          <p style={{ fontSize: '.82rem', color: 'var(--ink-dim)', marginTop: 10 }}>
            Lead-N scores above the uncompressed baseline at every level; every relevance-scored method
            (TF-IDF, bi-encoder, cross-encoder) scores below it. CNN/DailyMail's compression query is
            generic ("summarize the main points"), so the scorers have no article-specific signal to
            select against, unlike SQuAD below, where the query is the actual question.
          </p>
        </div>

        <div className="grid3" style={{ marginTop: 30 }}>
          <div className="card">
            <div className="num">SQuAD v2</div>
            <h3>QA baseline</h3>
            <p>Cross-encoder: 54% context-token reduction, no detectable F1 loss (300 questions).<span className="badge">done</span></p>
          </div>
          <div className="card">
            <div className="num">CNN/DailyMail</div>
            <h3>Summarization</h3>
            <p>13,368-article encoder stage and 1,000-article end-to-end run both complete.<span className="badge">done</span></p>
          </div>
          <div className="card">
            <div className="num">LoCoMo</div>
            <h3>Context manager</h3>
            <p>Relevance-aware F1 0.305 vs. 0.039 for sliding window (1,542 questions).<span className="badge">done</span></p>
          </div>
        </div>
      </div>
    </section>
  )
}