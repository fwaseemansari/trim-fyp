import Nav from './components/Nav'
import Architecture from './components/Architecture'
import CompressDemo from './components/CompressDemo'
import ContextDemo from './components/ContextDemo'
import ResultsDashboard from './components/ResultsDashboard'
import Reveal from './components/Reveal'

export default function App() {
  return (
    <>
      <Nav />
      <Architecture />
      <Reveal><CompressDemo /></Reveal>
      <Reveal><ContextDemo /></Reveal>
      <Reveal><ResultsDashboard /></Reveal>
      <footer>TRIM — Faiqa Waseem · Easha Javed · Dania Athar · FAST-NUCES, Fall 2026</footer>
    </>
  )
}
