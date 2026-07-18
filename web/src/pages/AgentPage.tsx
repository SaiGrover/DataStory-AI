import { useEffect, useRef, useState } from "react";
import { useMutation } from "react-query";
import { Bot, Check, Clipboard, Database, RotateCcw, Send, Sparkles } from "lucide-react";
import { askDatasetAgent } from "../lib/api";
import { useAppState } from "../lib/store";

type AgentMessage = {
  role: "user" | "agent";
  text: string;
  sources?: string[];
  confidence?: "high" | "medium" | "low";
  mode?: "ai-grounded" | "local-grounded";
};

const starterPrompts = [
  "Give me a concise dataset health check.",
  "Which columns need cleaning first, and why?",
  "Does the selected target have class imbalance?",
  "What are the strongest relationships in this dataset?",
];

function AnswerText({ text }: { text: string }) {
  return (
    <div className="space-y-1.5">
      {text.split("\n").map((line, index) => {
        const trimmed = line.trim();
        if (!trimmed) return <div key={index} className="h-1" />;
        if (trimmed.startsWith("- ")) {
          return <div key={index} className="flex gap-2"><span className="font-black text-sage">•</span><span>{trimmed.slice(2)}</span></div>;
        }
        return <p key={index}>{trimmed.replace(/^#{1,4}\s+/, "")}</p>;
      })}
    </div>
  );
}

export function AgentPage() {
  const { dataset, target, taskType } = useAppState();
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [suggestions, setSuggestions] = useState(starterPrompts);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const ask = useMutation(
    (text: string) => askDatasetAgent(dataset!.dataset_id, text, {
      target,
      taskType,
      history: messages.slice(-6).map((message) => ({
        role: message.role === "agent" ? "assistant" : "user",
        content: message.text,
      })),
    }),
    {
      onMutate: (text) => {
        setMessages((current) => [...current, { role: "user", text }]);
        setQuestion("");
      },
      onSuccess: (data) => {
        setMessages((current) => [...current, {
          role: "agent",
          text: data.answer,
          sources: data.sources,
          confidence: data.confidence,
          mode: data.mode,
        }]);
        if (data.follow_up_questions?.length) setSuggestions(data.follow_up_questions);
      },
      onError: (error) => {
        const message = error instanceof Error ? error.message : "Could not reach the DataStory AI agent.";
        setMessages((current) => [...current, {
          role: "agent",
          text: `The dataset agent could not reach the backend: ${message}\n\nStart the FastAPI service and try again. The grounded local answer mode does not require an AI API key.`,
          confidence: "low",
        }]);
      },
    },
  );

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, ask.isLoading]);

  useEffect(() => {
    setMessages([]);
    setSuggestions(starterPrompts);
  }, [dataset?.dataset_id]);

  if (!dataset) return <div className="ds-card p-8">Load a dataset first to ask DataStory AI questions.</div>;

  const submit = (text = question) => {
    const trimmed = text.trim();
    if (trimmed && !ask.isLoading) ask.mutate(trimmed);
  };

  const copyAnswer = async (text: string, index: number) => {
    await navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    window.setTimeout(() => setCopiedIndex(null), 1500);
  };

  return (
    <div className="grid h-[calc(100vh-265px)] min-h-[620px] grid-cols-[340px_minmax(0,1fr)] gap-5 overflow-hidden max-xl:grid-cols-1 max-xl:h-auto max-xl:overflow-visible">
      <aside className="ds-card overflow-y-auto p-6 max-xl:overflow-visible">
        <div className="flex items-center gap-3">
          <span className="grid h-12 w-12 place-items-center rounded-full bg-[#EEF5E9] text-sage dark:bg-[#263224]">
            <Bot className="h-6 w-6" />
          </span>
          <div>
            <h1 className="text-2xl font-black">DataStory AI</h1>
            <p className="text-sm text-zinc-500">Grounded dataset analyst</p>
          </div>
        </div>

        <div className="mt-6 rounded-2xl border border-line bg-stone-50/70 p-4 text-sm dark:bg-[#1d232c]">
          <div className="flex items-center gap-2 font-black"><Database className="h-4 w-4 text-sage" />{dataset.filename}</div>
          <div className="mt-3 grid grid-cols-2 gap-2 text-zinc-500">
            <div><span className="block font-black text-ink">{dataset.rows.toLocaleString()}</span>rows</div>
            <div><span className="block font-black text-ink">{dataset.columns.toLocaleString()}</span>columns</div>
          </div>
          <div className="mt-3 border-t border-line pt-3 text-zinc-500">Target: <span className="font-bold text-ink">{target || "not selected"}</span></div>
          <div className="mt-1 text-zinc-500">Task: <span className="font-bold text-ink">{taskType || "not detected"}</span></div>
        </div>

        <div className="mt-5">
          <div className="mb-2 text-xs font-black uppercase tracking-wider text-zinc-500">Try asking</div>
          <div className="space-y-2">
            {suggestions.map((prompt) => (
              <button key={prompt} className="w-full rounded-xl border border-line px-4 py-3 text-left text-sm font-bold transition hover:border-sage hover:bg-stone-50 dark:hover:bg-[#1d232c]" onClick={() => submit(prompt)}>
                {prompt}
              </button>
            ))}
          </div>
        </div>
      </aside>

      <main className="ds-card flex min-h-0 flex-col overflow-hidden max-xl:min-h-[680px]">
        <div className="flex shrink-0 items-center justify-between border-b border-line p-5">
          <div className="flex items-center gap-3">
            <Sparkles className="h-5 w-5 text-sage" />
            <div>
              <h2 className="text-xl font-black">Ask about this dataset</h2>
              <p className="text-xs text-zinc-500">Answers use live profile and model evidence</p>
            </div>
          </div>
          {!!messages.length && (
            <button className="flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-bold text-zinc-500 hover:bg-stone-100 dark:hover:bg-[#263224]" onClick={() => { setMessages([]); setSuggestions(starterPrompts); }}>
              <RotateCcw className="h-3.5 w-3.5" /> Clear
            </button>
          )}
        </div>

        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto scroll-smooth p-5">
          {!messages.length && (
            <div className="grid h-full place-items-center text-center text-sm text-zinc-500">
              <div>
                <span className="mx-auto grid h-16 w-16 place-items-center rounded-2xl bg-[#EEF5E9] dark:bg-[#263224]"><Bot className="h-8 w-8 text-sage" /></span>
                <p className="mt-4 max-w-md font-bold text-ink">Ask a concrete question—even one that names a column.</p>
                <p className="mt-1 max-w-md">I’ll use its distribution, quality profile, target setup, correlations, and trained model metrics when relevant.</p>
              </div>
            </div>
          )}

          {messages.map((message, index) => (
            <div key={index} className={`group max-w-[84%] rounded-2xl border border-line p-4 text-sm leading-6 ${message.role === "user" ? "ml-auto bg-[#EEF5E9] text-ink dark:bg-[#263224]" : "bg-white dark:bg-[#20272f]"}`}>
              <div className="mb-2 flex items-center justify-between gap-3">
                <div className="flex items-center gap-2 text-xs font-black uppercase tracking-wide text-sage">
                  {message.role === "user" ? "You" : "DataStory AI"}
                  {message.role === "agent" && message.confidence && (
                    <span className={`rounded-full px-2 py-0.5 text-[10px] normal-case tracking-normal ${message.confidence === "high" ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300" : message.confidence === "low" ? "bg-red-50 text-red-700" : "bg-amber-50 text-amber-700 dark:bg-amber-950/40 dark:text-amber-300"}`}>
                      {message.confidence} confidence
                    </span>
                  )}
                </div>
                {message.role === "agent" && (
                  <button aria-label="Copy answer" className="rounded-md p-1.5 text-zinc-400 opacity-0 transition hover:bg-stone-100 hover:text-ink group-hover:opacity-100 dark:hover:bg-[#303944]" onClick={() => copyAnswer(message.text, index)}>
                    {copiedIndex === index ? <Check className="h-3.5 w-3.5 text-emerald-600" /> : <Clipboard className="h-3.5 w-3.5" />}
                  </button>
                )}
              </div>
              <AnswerText text={message.text} />
              {message.role === "agent" && !!message.sources?.length && (
                <div className="mt-4 border-t border-line pt-3">
                  <div className="mb-2 text-[10px] font-black uppercase tracking-wider text-zinc-400">Evidence used</div>
                  <div className="flex flex-wrap gap-1.5">
                    {message.sources.map((source) => <span key={source} className="rounded-full border border-line bg-stone-50 px-2.5 py-1 text-[11px] font-bold text-zinc-600 dark:bg-[#171d24] dark:text-zinc-300">{source}</span>)}
                  </div>
                  {message.mode && <div className="mt-2 text-[10px] text-zinc-400">{message.mode === "local-grounded" ? "Local grounded mode" : "AI answer grounded in dataset evidence"}</div>}
                </div>
              )}
            </div>
          ))}

          {ask.isLoading && (
            <div className="max-w-[84%] rounded-2xl border border-line bg-white p-4 text-sm font-bold text-zinc-500 dark:bg-[#20272f]">
              <span className="inline-flex items-center gap-2"><span className="h-2 w-2 animate-pulse rounded-full bg-sage" />Reading the relevant dataset evidence…</span>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="shrink-0 border-t border-line p-4">
          <div className="flex items-end gap-3">
            <textarea
              className="max-h-32 min-h-[50px] min-w-0 flex-1 resize-none rounded-xl border border-line bg-white px-4 py-3 outline-none transition focus:border-sage dark:bg-[#151a21]"
              value={question}
              rows={1}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  submit();
                }
              }}
              placeholder="Ask about a column, cleaning issue, relationship, or model result…"
            />
            <button className="ds-button-primary min-h-[50px]" disabled={!question.trim() || ask.isLoading} onClick={() => submit()}>
              <Send className="h-4 w-4" /> Ask
            </button>
          </div>
          <p className="mt-2 px-1 text-[11px] text-zinc-400">Enter to send · Shift + Enter for a new line</p>
        </div>
      </main>
    </div>
  );
}
