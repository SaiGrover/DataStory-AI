import { useRef } from "react";
import { useMutation, useQuery } from "react-query";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Activity, Bike, Database, Dumbbell, Flower2, GraduationCap, Hash, HeartPulse, Ship, Upload, Users, Wine } from "lucide-react";
import { listSamples, loadSample, uploadCsv } from "../lib/api";
import { useAppState } from "../lib/store";

const sampleIcons: Record<string, React.ElementType> = {
  titanic: Ship,
  student: GraduationCap,
  churn: Users,
  bike: Bike,
  iris: Flower2,
  wine: Wine,
  breast_cancer: HeartPulse,
  diabetes: Activity,
  digits: Hash,
  linnerud: Dumbbell,
};

export function StartPage() {
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();
  const { setDataset, setResults, setTarget, setTaskType } = useAppState();
  const { data, error: samplesError, isLoading: samplesLoading, refetch } = useQuery("samples", listSamples);

  const onDataset = (dataset: Awaited<ReturnType<typeof uploadCsv>>) => {
    setDataset(dataset);
    setResults([]);
    setTarget("");
    setTaskType("");
    navigate("/health");
  };

  const uploadMutation = useMutation(uploadCsv, { onSuccess: onDataset });
  const sampleMutation = useMutation(loadSample, { onSuccess: onDataset });
  const handleFile = (file?: File) => {
    if (file) uploadMutation.mutate(file);
  };

  return (
    <div className="pb-28">
      <section className="grid min-h-[390px] grid-cols-[1fr_.95fr] items-center gap-8 overflow-hidden rounded-none">
        <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }}>
          <div className="mb-5 inline-flex rounded-full bg-[#EEF1EA] px-4 py-2 text-sm font-bold text-sage">AI-POWERED DATA ANALYSIS</div>
          <h1 className="max-w-4xl text-6xl font-extrabold leading-tight tracking-normal">
            Data Analysis Made <span className="text-sage">Simple</span>,<br />
            Insights Made <span className="text-sage">Clear</span>.
          </h1>
          <p className="mt-5 max-w-2xl text-lg leading-8 text-zinc-600 dark:text-zinc-300">
            Upload a CSV file and let DataStory AI clean your data, discover insights, train the best models, and generate a complete report.
          </p>
          <div className="mt-7 flex gap-6">
            <button className="ds-button-primary min-w-[250px]" onClick={() => inputRef.current?.click()} disabled={uploadMutation.isLoading}>
              <Upload className="h-5 w-5" />
              Upload CSV File
            </button>
            <a className="ds-button-secondary min-w-[250px]" href="#samples">
              <Database className="h-5 w-5" />
              Choose Sample Dataset
            </a>
          </div>
          <input
            ref={inputRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(event) => {
              handleFile(event.target.files?.[0]);
              event.target.value = "";
            }}
          />
          <div className="mt-5 text-sm text-zinc-500">100% private • Your data stays on your device until you upload it to the local API.</div>
          {uploadMutation.error instanceof Error && <div className="mt-4 text-sm font-semibold text-red-600">{uploadMutation.error.message}</div>}
        </motion.div>
        <div className="relative h-[360px]">
          <div className="absolute inset-y-0 right-0 w-[680px] rounded-[50%] bg-[#E8E2D9]" />
          <img className="absolute bottom-0 right-0 z-10 w-[720px] max-w-none" src="/hero_workspace.png" alt="Workspace dashboard" />
        </div>
      </section>

      <section id="samples" className="mt-4 grid grid-cols-[.55fr_1.45fr] items-start gap-4">
        <div className="ds-card self-start p-7">
          <h2 className="text-2xl font-extrabold">Start your analysis</h2>
          <p className="mt-2 text-zinc-600 dark:text-zinc-300">Upload your CSV file to begin your data journey.</p>
          <button
            className="mt-6 flex h-44 w-full flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-[#D5CEC3]"
            onClick={() => inputRef.current?.click()}
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault();
              handleFile(event.dataTransfer.files?.[0]);
            }}
            disabled={uploadMutation.isLoading}
          >
            <Upload className="h-10 w-10 rounded-full bg-[#EEF1EA] p-2 text-sage" />
            <strong>Drag & drop your CSV file here</strong>
            <span className="text-sage">or click to browse</span>
            <small className="text-zinc-500">Supports CSV files up to 50MB</small>
          </button>
        </div>
        <div className="ds-card overflow-hidden">
          <div className="flex items-start justify-between px-8 py-6">
            <div>
              <h2 className="text-2xl font-extrabold">Try with a sample dataset</h2>
              <p className="mt-2 text-zinc-600 dark:text-zinc-300">Explore sample datasets and see DataStory AI in action.</p>
            </div>
          </div>
          {samplesError instanceof Error && (
            <div className="border-t border-line bg-[#FBF7ED] px-8 py-8">
              <div className="rounded-xl border border-amber-200 bg-amber-50 p-5 text-amber-950">
                <h3 className="text-lg font-black">Backend is not running</h3>
                <p className="mt-2 text-sm">
                  The sample CSV files are still in the project, but the React app needs the FastAPI backend at <strong>127.0.0.1:8000</strong> to load them.
                </p>
                <div className="mt-4 rounded-lg bg-white px-4 py-3 font-mono text-xs text-zinc-700">
                  cd datastory_ai<br />
                  python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
                </div>
                <button className="ds-button-primary mt-4" onClick={() => refetch()}>
                  Retry sample datasets
                </button>
              </div>
            </div>
          )}
          {samplesLoading && (
            <div className="border-t border-line px-8 py-10 text-sm font-semibold text-zinc-500">Loading sample datasets...</div>
          )}
          <div className="grid grid-cols-1 border-t border-line sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
            {(data?.samples ?? []).map((sample) => {
              const Icon = sampleIcons[sample.id] ?? Database;
              return (
                <button
                  key={sample.id}
                  className="flex min-h-[225px] flex-col items-center border-b border-r border-line px-4 py-5 text-center last:border-r-0 hover:bg-stone-50 dark:hover:bg-zinc-900 xl:[&:nth-child(5n)]:border-r-0"
                  onClick={() => sampleMutation.mutate(sample.id)}
                  disabled={sampleMutation.isLoading}
                >
                  <Icon className="mb-4 h-16 w-16 rounded-full bg-[#EEF1EA] p-4 text-sage" />
                  <strong>{sample.name}</strong>
                  <span className="mt-2 min-h-10 text-sm text-zinc-500">{sample.description}</span>
                  <small className="mt-auto text-zinc-600">{sample.rows.toLocaleString()} rows • {sample.columns} columns</small>
                  <span className="mt-4 w-full border-t border-line pt-3 text-base font-semibold">Load</span>
                </button>
              );
            })}
          </div>
        </div>
      </section>
    </div>
  );
}
