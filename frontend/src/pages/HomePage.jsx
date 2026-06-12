import { useState } from 'react';
import PDFUploader from '../components/PDFUploader';
import ChatWindow from '../components/ChatWindow';
import CitationViewer from '../components/CitationViewer';

export default function HomePage() {
  const [answer, setAnswer] = useState(null);

  return (
    <main className="min-h-screen p-6 text-slate-100">
      <div className="mx-auto flex max-w-7xl flex-col gap-6">
        <header className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6 shadow-2xl shadow-slate-950/30">
          <p className="text-xs uppercase tracking-[0.35em] text-cyan-300">DocInsight AI</p>
          <h1 className="mt-2 text-3xl font-semibold text-white">Research PDF Q&A and evaluation workspace</h1>
          <p className="mt-2 max-w-2xl text-sm text-slate-300">Upload PDFs, ask questions, inspect citations, and review evaluation metrics through the existing FastAPI backend.</p>
        </header>

        <div className="grid gap-6 xl:grid-cols-[340px_1fr_380px]">
          <aside className="space-y-6">
            <PDFUploader />
            <section className="rounded-3xl border border-slate-800 bg-slate-900/85 p-5 shadow-2xl shadow-slate-950/30">
              <h3 className="text-sm font-semibold text-white">Retrieval settings</h3>
              <p className="mt-2 text-sm text-slate-300">Use the existing backend toggles for hybrid retrieval and reranker behavior.</p>
              <div className="mt-4 space-y-3 text-sm text-slate-200">
                <div className="rounded-2xl border border-slate-800 bg-slate-950/80 p-3">Top K: 5</div>
                <div className="rounded-2xl border border-slate-800 bg-slate-950/80 p-3">Threshold: 0.35</div>
                <div className="rounded-2xl border border-slate-800 bg-slate-950/80 p-3">Hybrid: Enabled</div>
              </div>
            </section>
          </aside>

          <section>
            <ChatWindow onAnswer={setAnswer} />
          </section>

          <aside>
            <CitationViewer sources={answer?.sources || []} chunks={answer?.chunks || []} />
          </aside>
        </div>
      </div>
    </main>
  );
}
