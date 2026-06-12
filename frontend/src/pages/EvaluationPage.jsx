import EvaluationDashboard from '../components/EvaluationDashboard';

export default function EvaluationPage() {
  return (
    <main className="min-h-screen p-6 text-slate-100">
      <div className="mx-auto flex max-w-7xl flex-col gap-6">
        <header className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6 shadow-2xl shadow-slate-950/30">
          <p className="text-xs uppercase tracking-[0.35em] text-cyan-300">Evaluation</p>
          <h1 className="mt-2 text-3xl font-semibold text-white">RAGAS metrics and comparison</h1>
          <p className="mt-2 text-sm text-slate-300">Review the existing backend evaluation results with metric cards, trend visuals, and comparison tables.</p>
        </header>
        <EvaluationDashboard />
      </div>
    </main>
  );
}
