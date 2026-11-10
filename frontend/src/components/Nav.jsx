import { useEffect, useState } from 'react'
import { api } from '../api'

export default function Nav() {
  const [up, setUp] = useState(null) // null = checking, true/false after
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    api.health().then(() => setUp(true)).catch(() => setUp(false))
    const onScroll = () => setScrolled(window.scrollY > 8)
    window.addEventListener('scroll', onScroll)
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <nav className={scrolled ? 'scrolled' : ''}>
      <div className="wrap inner">
        <div className="brand"><span className="dot"></span>TRIM</div>
        <div className="navlinks">
          <a href="#architecture">Architecture</a>
          <a href="#demo">Compress</a>
          <a href="#context">Context</a>
          <a href="#results">Results</a>
        </div>
        {up === null && <span className="badge-live off">checking API…</span>}
        {up === true && <span className="badge-live on"><i className="pulse"></i>backend connected</span>}
        {up === false && <span className="badge-live off"><i className="pulse off"></i>backend offline</span>}
      </div>
    </nav>
  )
}
