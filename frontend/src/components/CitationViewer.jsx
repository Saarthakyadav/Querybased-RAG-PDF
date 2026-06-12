export default function CitationViewer({ sources = [], chunks = [] }) {
  return (
    <section className="rounded-3xl border border-slate-800 bg-slate-900/85 p-5 shadow-2xl shadow-slate-950/30">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.32em] text-cyan-300">Citations</p>
          <h2 className="text-xl font-semibold text-white">Source evidence</h2>
        </div>
        <span className="rounded-full bg-violet-400/10 px-3 py-1 text-xs text-violet-200">Live</span>
      </div>
      <div className="space-y-3">
        {sources.length ? sources.map((source, idx) => (
          <article key={`${source.source_file}-${idx}`} className="rounded-2xl border border-slate-800 bg-slate-950/80 p-4">
            <p className="text-sm font-semibold text-white">{source.source_file}</p>
            <p className="mt-1 text-xs text-slate-400">Page {source.page || '—'} • Author {source.author || 'Unknown'} • Date {source.date || 'N/A'}</p>
            <p className="mt-3 text-sm text-slate-200">{chunks[idx]?.content?.slice(0, 220) || 'No snippet available.'}</p>
          </article>
        )) : <p className="text-sm text-slate-400">Ask a question to view citations and source snippets here.</p>}
      </div>
    </section>
  );
}
