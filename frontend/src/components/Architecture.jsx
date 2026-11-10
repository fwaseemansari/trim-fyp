export default function Architecture() {
  return (
    <>
      <header className="hero">
        <div className="wrap">
          <span className="tag">Final Year Project · FAST-NUCES</span>
          <h1><span className="accent">Token Reduction</span> & Intelligent Management for LLM applications</h1>
          <p>TRIM cuts what you send to an LLM, the prompt and the conversation history, without cutting
            the answer quality. Everything below calls the real Python pipeline through a live API,
            not a simulation.</p>
        </div>
      </header>

      <section id="architecture">
        <div className="wrap">
          <div className="eyebrow">How it works</div>
          <h2>Three modules, one pipeline</h2>
          <div className="grid3">
            <div className="card">
              <div className="num">01</div>
              <h3>Token Analysis Module</h3>
              <p>Counts tokens with the real tokenizer per backend (tiktoken for OpenAI, the actual
                gpt-oss-20b tokenizer for Groq) and estimates cost before and after compression.</p>
            </div>
            <div className="card">
              <div className="num">02</div>
              <h3>Prompt Compression Module</h3>
              <p>Scores sentence windows for relevance, TF-IDF, a bi-encoder, or a cross-encoder, and
                keeps only enough to hit a target token budget.</p>
            </div>
            <div className="card">
              <div className="num">03</div>
              <h3>Adaptive Context Manager</h3>
              <p>For multi-turn chats, decides which earlier turns to keep or drop as the conversation
                grows past the token budget, sliding-window or relevance-aware.</p>
            </div>
          </div>
        </div>
      </section>
    </>
  )
}
