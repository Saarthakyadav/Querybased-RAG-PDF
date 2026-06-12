import { useEffect, useRef, useState } from 'react';
import { queryDocuments } from '../services/api';
import Loader from './Loader';

export default function ChatWindow({ onAnswer }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const question = input.trim();
    setInput('');
    const userMessage = { role: 'user', content: question };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);

    try {
      const response = await queryDocuments({
        question,
        top_k: 5,
        score_threshold: 0.35,
        use_hybrid: true,
        use_reranker: true,
      });

      const data = response.data || response;
      const answer = data.answer || 'No answer returned.';
      const sources = data.sources || [];
      const chunks = data.chunks || [];

      setMessages((prev) => [...prev, { role: 'assistant', content: answer, sources, chunks }]);
      onAnswer?.({ answer, sources, chunks });
    } catch (error) {
      setMessages((prev) => [...prev, { role: 'assistant', content: error?.response?.data?.detail || 'Unable to answer right now.' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="flex h-[700px] flex-col rounded-3xl border border-slate-800 bg-slate-900/85 shadow-2xl shadow-slate-950/30">
      <div className="border-b border-slate-800 p-5">
        <p className="text-xs uppercase tracking-[0.32em] text-cyan-300">Chat</p>
        <h2 className="text-xl font-semibold text-white">Research paper Q&A</h2>
      </div>
      <div className="flex-1 space-y-4 overflow-y-auto p-5">
        {messages.length === 0 && <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-5 text-sm text-slate-300">Ask a question about your uploaded PDFs to begin.</div>}
        {messages.map((msg, index) => (
          <article key={`${msg.role}-${index}`} className={`max-w-[90%] rounded-3xl p-4 ${msg.role === 'user' ? 'ml-auto bg-cyan-400 text-slate-950' : 'bg-slate-950/80 text-slate-100 border border-slate-800'}`}>
            <p className="text-sm whitespace-pre-line">{msg.content}</p>
          </article>
        ))}
        {loading && <Loader label="Generating answer…" />}
        <div ref={endRef} />
      </div>
      <div className="border-t border-slate-800 p-4">
        <div className="flex gap-3">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
            className="flex-1 rounded-2xl border border-slate-700 bg-slate-950/80 px-4 py-3 text-sm text-slate-100 outline-none ring-0 placeholder:text-slate-400 focus:border-cyan-400"
            placeholder="Ask a question about the documents…"
          />
          <button
            onClick={sendMessage}
            className="rounded-2xl bg-cyan-400 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-300"
          >
            Ask
          </button>
        </div>
      </div>
    </section>
  );
}
