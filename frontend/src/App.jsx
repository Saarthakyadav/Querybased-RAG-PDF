import { NavLink, Route, Routes } from 'react-router-dom';
import HomePage from './pages/HomePage';
import EvaluationPage from './pages/EvaluationPage';

export default function App() {
  return (
    <div className="min-h-screen text-slate-100">
      <nav className="sticky top-0 z-10 border-b border-slate-800 bg-slate-950/90 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div>
            <p className="text-xs uppercase tracking-[0.35em] text-cyan-300">DocInsight AI</p>
            <h2 className="text-lg font-semibold text-white">FastAPI + React + Tailwind</h2>
          </div>
          <div className="flex gap-2">
            <NavLink to="/" className={({ isActive }) => `rounded-full px-4 py-2 text-sm ${isActive ? 'bg-cyan-400 text-slate-950' : 'bg-slate-900 text-slate-200 hover:bg-slate-800'}`}>
              Home
            </NavLink>
            <NavLink to="/evaluation" className={({ isActive }) => `rounded-full px-4 py-2 text-sm ${isActive ? 'bg-cyan-400 text-slate-950' : 'bg-slate-900 text-slate-200 hover:bg-slate-800'}`}>
              Evaluation
            </NavLink>
          </div>
        </div>
      </nav>

      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/evaluation" element={<EvaluationPage />} />
      </Routes>
    </div>
  );
}
