import { useEffect, useMemo, useState } from 'react';
import { getMetrics, runEvaluation } from '../services/api';
import Loader from './Loader';

const metricLabels = {
  faithfulness: 'Faithfulness',
  answer_relevancy: 'Answer Relevancy',
  context_precision: 'Context Precision',
  context_recall: 'Context Recall',
};

export default function EvaluationDashboard() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const loadMetrics = async () => {
    setLoading(true);
    try {
      const response = await getMetrics();
      setMetrics(response.data || response);
      setError('');
    } catch (err) {
      setError(err?.response?.data?.detail || 'Unable to load metrics.');
    } finally {
      setLoading(false);
    }
  };

  const runEval = async () => {
    setLoading(true);
    try {
      await runEvaluation();
      await loadMetrics();
    } catch (err) {
      setError(err?.response?.data?.detail || 'Evaluation failed.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadMetrics();
  }, []);

  const aggregate = metrics?.aggregate || {};

  const bars = useMemo(() => Object.entries(metricLabels).map(([key, label]) => ({
    key,
    label,
    value: aggregate[key] || 0,
  })), [aggregate]);

  return (
    <section className="rounded-3xl border border-slate-800 bg-slate-900/85 p-5 shadow-2xl shadow-slate-950/30">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.32em] text-cyan-300">Evaluation</p>
          <h2 className="text-xl font-semibold text-white">RAGAS dashboard</h2>
        </div>
        <button onClick={runEval} className="rounded-2xl bg-cyan-400 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-cyan-300">Run evaluation</button>
      </div>
      {error && <p className="mb-4 text-sm text-rose-300">{error}</p>}
      {loading ? <Loader label="Loading evaluation metrics…" /> : (
        <>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {bars.map((item) => (
              <article key={item.key} className="rounded-2xl border border-slate-800 bg-slate-950/80 p-4">
                <p className="text-sm text-slate-300">{item.label}</p>
                <p className="mt-2 text-3xl font-semibold text-white">{(item.value * 100).toFixed(1)}%</p>
                <div className="mt-3 h-2 rounded-full bg-slate-800">
                  <div className="h-2 rounded-full bg-gradient-to-r from-cyan-400 to-violet-400" style={{ width: `${Math.min(item.value * 100, 100)}%` }} />
                </div>
              </article>
            ))}
          </div>
          <div className="mt-6 rounded-2xl border border-slate-800 bg-slate-950/80 p-4">
            <h3 className="text-sm font-semibold text-white">Comparison view</h3>
            <p className="mt-1 text-sm text-slate-300">Baseline • Hybrid Retrieval • Hybrid + Reranker</p>
            <div className="mt-4 overflow-x-auto">
              <table className="min-w-full text-left text-sm text-slate-200">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400">
                    <th className="pb-2 pr-4">Metric</th>
                    <th className="pb-2 pr-4">Baseline</th>
                    <th className="pb-2 pr-4">Hybrid</th>
                    <th className="pb-2">Hybrid + Reranker</th>
                  </tr>
                </thead>
                <tbody>
                  {bars.map((item) => (
                    <tr key={item.key} className="border-b border-slate-800/80">
                      <td className="py-3 pr-4">{item.label}</td>
                      <td className="py-3 pr-4">{(aggregate[item.key] || 0).toFixed(2)}</td>
                      <td className="py-3 pr-4">{(aggregate[item.key] || 0).toFixed(2)}</td>
                      <td className="py-3">{(aggregate[item.key] || 0).toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </section>
  );
}
