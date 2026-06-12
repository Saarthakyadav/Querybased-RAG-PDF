import { useState } from 'react';
import { uploadPDF } from '../services/api';

export default function PDFUploader({ onUploaded }) {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState('');

  const handleChange = (event) => {
    setFiles(Array.from(event.target.files || []));
  };

  const handleUpload = async () => {
    if (!files.length) return;
    setUploading(true);
    setStatus('Uploading PDFs…');
    try {
      const result = await uploadPDF(files);
      setStatus(`Uploaded ${result.files_processed || files.length} PDF(s).`);
      onUploaded?.(result);
    } catch (error) {
      setStatus(error?.response?.data?.detail || 'Upload failed.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <section className="rounded-3xl border border-slate-800 bg-slate-900/85 p-5 shadow-2xl shadow-slate-950/30">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.32em] text-cyan-300">Upload</p>
          <h2 className="text-xl font-semibold text-white">PDF ingestion</h2>
        </div>
        <span className="rounded-full bg-emerald-400/10 px-3 py-1 text-xs text-emerald-200">FastAPI</span>
      </div>
      <label className="block rounded-2xl border border-dashed border-slate-700 bg-slate-950/60 p-6 text-center text-slate-300 hover:border-cyan-400 hover:bg-slate-950/90">
        <input type="file" multiple accept="application/pdf" onChange={handleChange} className="hidden" />
        <p className="text-sm">Drag & drop or click to select PDFs</p>
        <p className="mt-1 text-xs text-slate-400">Multiple files supported</p>
      </label>
      <div className="mt-3 text-xs text-slate-400">{files.length ? files.map((f) => f.name).join(', ') : 'No files selected yet.'}</div>
      <button
        onClick={handleUpload}
        disabled={uploading || !files.length}
        className="mt-4 w-full rounded-2xl bg-cyan-400 px-4 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
      >
        {uploading ? 'Uploading…' : 'Upload PDFs'}
      </button>
      {status && <p className="mt-3 text-sm text-emerald-200">{status}</p>}
    </section>
  );
}
