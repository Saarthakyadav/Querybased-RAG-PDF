export default function SourceCard({ source, chunk, index }) {
  return (
    <article className="rounded-2xl border border-slate-800 bg-slate-900/90 p-4 shadow-lg shadow-slate-950/30">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-cyan-300">Source {index + 1}</p>
          <h3 className="text-sm font-semibold text-white">{source?.source_file || 'Unknown source'}</h3>
        </div>
        <span className="rounded-full bg-cyan-400/10 px-3 py-1 text-xs font-medium text-cyan-200">
          Score {chunk?.score ? `${(chunk.score * 100).toFixed(1)}%` : 'N/A'}
        </span>
      </div>
      <p className="text-xs text-slate-400">Page {source?.page || '—'} • Author {source?.author || 'Unknown'} • Date {source?.date || 'N/A'}</p>
      <p className="mt-3 text-sm text-slate-200 line-clamp-5">{chunk?.content || 'No snippet available.'}</p>
    </article>
  );
}
