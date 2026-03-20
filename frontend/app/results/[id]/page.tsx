"use client";

import { useEffect, useState, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import { pollStatus, StatusResponse } from "@/lib/api";

type Stage =
  | "searching"
  | "researching"
  | "scoring"
  | "generating"
  | "done"
  | "error";

const STAGE_LABELS: Record<Stage, string> = {
  searching:   "Searching the database...",
  researching: "Researching each company...",
  scoring:     "Ranking and scoring matches...",
  generating:  "Generating your report...",
  done:        "Done!",
  error:       "Something went wrong",
};

// Estimate pipeline stage by elapsed time (seconds)
function estimateStage(elapsed: number): Stage {
  if (elapsed < 15)  return "searching";
  if (elapsed < 60)  return "researching";
  if (elapsed < 100) return "scoring";
  return "generating";
}

export default function ResultsPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();

  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [stage, setStage] = useState<Stage>("searching");
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState("");
  const [pageUrl, setPageUrl] = useState("");
  const startRef = useRef(Date.now());
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => { setPageUrl(window.location.href); }, []);

  useEffect(() => {
    if (!id) return;

    // Elapsed timer
    timerRef.current = setInterval(() => {
      const s = Math.floor((Date.now() - startRef.current) / 1000);
      setElapsed(s);
      setStage((prev) => {
        if (prev === "done" || prev === "error") return prev;
        return estimateStage(s);
      });
    }, 1000);

    // Poll pipeline status every 3s
    // Allow up to 5 × 404 retries (~10s) to handle the brief window between
    // session creation and the first poll. After that, treat 404 as "not found".
    let notFoundRetries = 0;
    const MAX_NOT_FOUND_RETRIES = 5;

    const doPoll = async () => {
      try {
        const res = await pollStatus(id);
        if (res.status === "done") {
          clearInterval(timerRef.current!);
          setStage("done");
          setStatus(res);
        } else if (res.status === "error") {
          clearInterval(timerRef.current!);
          setStage("error");
          setError("The pipeline encountered an error. Please try again.");
        } else {
          pollRef.current = setTimeout(doPoll, 3000);
        }
      } catch (err: unknown) {
        if ((err as Error).message?.includes("not found")) {
          notFoundRetries += 1;
          if (notFoundRetries <= MAX_NOT_FOUND_RETRIES) {
            // Session may not be written yet — retry briefly
            pollRef.current = setTimeout(doPoll, 2000);
          } else {
            clearInterval(timerRef.current!);
            setStage("error");
            setError("Session not found. This link may have expired or the session ID is invalid.");
          }
        } else {
          clearInterval(timerRef.current!);
          setStage("error");
          setError((err as Error).message || "Network error");
        }
      }
    };
    doPoll();

    return () => {
      clearInterval(timerRef.current!);
      clearTimeout(pollRef.current!);
    };
  }, [id]);

  // ── Loading screen ────────────────────────────────────────────────────────
  if (stage !== "done" && stage !== "error") {
    return (
      <div className="min-h-[80vh] flex flex-col items-center justify-center px-4">
        {/* Animated pipeline steps */}
        <div className="w-full max-w-sm mb-10">
          <PipelineSteps current={stage} />
        </div>

        <div className="text-center space-y-3">
          <h2 className="text-xl font-bold text-navy">{STAGE_LABELS[stage]}</h2>
          <p className="text-gray-500 text-sm">
            Our AI pipeline is finding your best-matched partners.<br />
            This typically takes 30–90 seconds.
          </p>
          <p className="text-xs text-gray-400">{elapsed}s elapsed</p>
        </div>

        {/* Share URL */}
        <div className="mt-8 bg-white border border-gray-100 rounded-xl px-5 py-3 shadow-sm text-sm text-gray-500">
          <span className="font-semibold text-navy">Shareable URL:</span>{" "}
          <span className="font-mono text-teal">
            {pageUrl}
          </span>
          <p className="text-xs mt-1 text-gray-400">
            Bookmark this page — results are saved for 30 days.
          </p>
        </div>
      </div>
    );
  }

  // ── Error screen ──────────────────────────────────────────────────────────
  if (stage === "error") {
    return (
      <div className="min-h-[60vh] flex flex-col items-center justify-center px-4 gap-6">
        <div className="text-center">
          <div className="text-5xl mb-4">⚠️</div>
          <h2 className="text-xl font-bold text-navy mb-2">Pipeline Error</h2>
          <p className="text-gray-500 text-sm max-w-sm">{error}</p>
        </div>
        <button onClick={() => router.push("/")} className="btn-primary">
          Try Again
        </button>
      </div>
    );
  }

  // ── Done — show report iframe ─────────────────────────────────────────────
  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-6">
      {/* Minimal top bar — session ID + actions only */}
      <div className="bg-navy text-white rounded-2xl px-6 py-3 flex items-center gap-4">
        <div>
          <span className="text-xs text-white/40 uppercase tracking-wide font-semibold mr-2">Session</span>
          <span className="font-mono text-teal font-bold text-sm">{id}</span>
        </div>
        <div className="ml-auto flex gap-2">
          <button
            onClick={() => pageUrl && navigator.clipboard.writeText(pageUrl)}
            className="text-xs bg-white/10 hover:bg-white/20 text-white px-3 py-1.5 rounded-lg transition-colors"
          >
            Copy Link
          </button>
          <button
            onClick={() => router.push("/")}
            className="text-xs bg-teal hover:bg-teal-light text-white px-3 py-1.5 rounded-lg transition-colors"
          >
            New Search
          </button>
        </div>
      </div>

      {/* HTML report in iframe */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
        <iframe
          src={`/api/report/${id}`}
          className="w-full"
          style={{ height: "calc(100vh - 220px)", minHeight: "600px", border: "none" }}
          title="Partner Finder Report"
        />
      </div>
    </div>
  );
}


// ── Pipeline step indicator ───────────────────────────────────────────────

const STEPS: { key: Stage; label: string; icon: string }[] = [
  { key: "searching",   label: "Search",   icon: "🔍" },
  { key: "researching", label: "Research", icon: "🌐" },
  { key: "scoring",     label: "Score",    icon: "📊" },
  { key: "generating",  label: "Report",   icon: "📄" },
];

const STAGE_ORDER: Stage[] = ["searching", "researching", "scoring", "generating", "done"];

function PipelineSteps({ current }: { current: Stage }) {
  const currentIdx = STAGE_ORDER.indexOf(current);

  return (
    <div className="flex items-center gap-0">
      {STEPS.map((step, i) => {
        const stepIdx = STAGE_ORDER.indexOf(step.key);
        const isDone = stepIdx < currentIdx;
        const isActive = stepIdx === currentIdx;

        return (
          <div key={step.key} className="flex items-center flex-1">
            <div className="flex flex-col items-center gap-1 flex-1">
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center text-lg transition-all
                  ${isDone ? "bg-teal text-white" : isActive ? "bg-navy text-white ring-4 ring-navy/20 animate-pulse" : "bg-gray-100 text-gray-400"}`}
              >
                {isDone ? "✓" : step.icon}
              </div>
              <span className={`text-xs font-medium ${isActive ? "text-navy" : isDone ? "text-teal" : "text-gray-400"}`}>
                {step.label}
              </span>
            </div>
            {i < STEPS.length - 1 && (
              <div className={`h-0.5 flex-1 mx-1 mb-5 rounded transition-colors ${isDone ? "bg-teal" : "bg-gray-200"}`} />
            )}
          </div>
        );
      })}
    </div>
  );
}
