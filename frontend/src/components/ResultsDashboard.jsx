import { useEffect, useRef, useState } from 'react'

const BARS = [
  { label: 'Lead-N', value: 0.552, pct: 83.8 },
  { label: 'Cross-encoder', value: 0.469, pct: 71.2 },
  { label: 'TF-IDF', value: 0.469, pct: 71.2 },
  { label: 'Bi-encoder', value: 0.432, pct: 65.6 },
]

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
        <div className="grid3" style={{ marginTop: 30 }}>
          <div className="card">
            <div className="num">SQuAD v2</div>
            <h3>QA baseline</h3>
            <p>EM 0.438 · F1 0.657 on the initial sample.<span className="badge">done</span></p>
          </div>
          <div className="card">
            <div className="num">CNN/DailyMail</div>
            <h3>Encoder stage</h3>
            <p>Full 13,368-article run complete, LLM summarization stage in progress.<span className="badge pending">in progress</span></p>
          </div>
          <div className="card">
            <div className="num">LoCoMo</div>
            <h3>Context manager</h3>
            <p>150 QA pairs evaluated for the sliding-window strategy.<span className="badge">done</span></p>
          </div>
        </div>
      </div>
    </section>
  )
}
