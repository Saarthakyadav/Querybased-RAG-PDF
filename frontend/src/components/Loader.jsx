export default function Loader({ label = 'Loading…' }) {
  return (
    <div className="flex items-center gap-3 rounded-2xl border border-slate-800 bg-slate-900/80 p-4 text-slate-200 shadow-lg shadow-slate-950/30">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-cyan-400 border-t-transparent" />
      <span className="text-sm text-slate-300">{label}</span>
    </div>
  );
}
