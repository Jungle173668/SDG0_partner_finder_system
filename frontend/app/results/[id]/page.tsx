"use client";

import { useEffect, useState, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  pollStatus,
  refineSearch,
  startSearch,
  StatusResponse,
  Company,
  CompanyFeedback,
  RefineResponse,
} from "@/lib/api";

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

function estimateStage(elapsed: number): Stage {
  if (elapsed < 15)  return "searching";
  if (elapsed < 60)  return "researching";
  if (elapsed < 100) return "scoring";
  return "generating";
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function ResultsPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();

  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [stage, setStage]   = useState<Stage>("searching");
  const [elapsed, setElapsed] = useState(0);
  const [error, setError]   = useState("");
  const [pageUrl, setPageUrl] = useState("");
  const startRef = useRef(Date.now());
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const pollRef  = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => { setPageUrl(window.location.href); }, []);

  useEffect(() => {
    if (!id) return;

    timerRef.current = setInterval(() => {
      const s = Math.floor((Date.now() - startRef.current) / 1000);
      setElapsed(s);
      setStage((prev) => {
        if (prev === "done" || prev === "error") return prev;
        return estimateStage(s);
      });
    }, 1000);

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

  // ── Loading ────────────────────────────────────────────────────────────────
  if (stage !== "done" && stage !== "error") {
    return (
      <div className="min-h-[80vh] flex flex-col items-center justify-center px-4">
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
        <div className="mt-8 bg-white border border-gray-100 rounded-xl px-5 py-3 shadow-sm text-sm text-gray-500">
          <span className="font-semibold text-navy">Shareable URL:</span>{" "}
          <span className="font-mono text-teal">{pageUrl}</span>
          <p className="text-xs mt-1 text-gray-400">
            Bookmark this page — results are saved for 30 days.
          </p>
        </div>
      </div>
    );
  }

  // ── Error ──────────────────────────────────────────────────────────────────
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

  // ── Done ───────────────────────────────────────────────────────────────────
  return (
    <DoneView
      id={id}
      status={status!}
      pageUrl={pageUrl}
      onNewSearch={() => router.push("/")}
      onNavigate={(newId) => router.push(`/results/${newId}`)}
    />
  );
}

// ---------------------------------------------------------------------------
// DoneView — tabs: Report | Refine
// ---------------------------------------------------------------------------

function DoneView({
  id,
  status,
  pageUrl,
  onNewSearch,
  onNavigate,
}: {
  id: string;
  status: StatusResponse;
  pageUrl: string;
  onNewSearch: () => void;
  onNavigate: (newId: string) => void;
}) {
  const [activeTab, setActiveTab] = useState<"report" | "refine">("report");

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-4">
      {/* Top bar */}
      <div className="bg-navy text-white rounded-2xl px-6 py-3 flex items-center gap-4">
        <div>
          <span className="text-xs text-white/40 uppercase tracking-wide font-semibold mr-2">Session</span>
          <span className="font-mono text-teal font-bold text-sm">{id}</span>
        </div>
        <div className="ml-auto flex gap-2">
          {status.prev_id && (
            <button
              onClick={() => onNavigate(status.prev_id!)}
              className="text-xs bg-white/10 hover:bg-white/20 text-white px-3 py-1.5 rounded-lg transition-colors"
            >
              ← Previous
            </button>
          )}
          {status.next_id && (
            <button
              onClick={() => onNavigate(status.next_id!)}
              className="text-xs bg-white/10 hover:bg-white/20 text-white px-3 py-1.5 rounded-lg transition-colors"
            >
              Next →
            </button>
          )}
          <button
            onClick={() => pageUrl && navigator.clipboard.writeText(pageUrl)}
            className="text-xs bg-white/10 hover:bg-white/20 text-white px-3 py-1.5 rounded-lg transition-colors"
          >
            Copy Link
          </button>
          <button
            onClick={onNewSearch}
            className="text-xs bg-teal hover:bg-teal-light text-white px-3 py-1.5 rounded-lg transition-colors"
          >
            New Search
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 bg-gray-100 rounded-xl p-1 w-fit">
        <button
          onClick={() => setActiveTab("report")}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            activeTab === "report"
              ? "bg-white text-navy shadow-sm"
              : "text-gray-500 hover:text-navy"
          }`}
        >
          Full Report
        </button>
        <button
          onClick={() => setActiveTab("refine")}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            activeTab === "refine"
              ? "bg-white text-navy shadow-sm"
              : "text-gray-500 hover:text-navy"
          }`}
        >
          Refine Search
        </button>
      </div>

      {/* Tab content */}
      {activeTab === "report" ? (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
          <iframe
            src={`/api/report/${id}`}
            className="w-full"
            style={{ height: "calc(100vh - 220px)", minHeight: "600px", border: "none" }}
            title="Partner Finder Report"
          />
        </div>
      ) : (
        <RefineTab
          sessionId={id}
          companies={status.scored_companies ?? []}
          onNavigate={onNavigate}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Refine tab
// ---------------------------------------------------------------------------

function RefineTab({
  sessionId,
  companies,
  onNavigate,
}: {
  sessionId: string;
  companies: Company[];
  onNavigate: (newId: string) => void;
}) {
  const [liked, setLiked]     = useState<Set<string>>(new Set());
  const [disliked, setDisliked] = useState<Set<string>>(new Set());
  const [userText, setUserText] = useState("");

  const [isLoading, setIsLoading]         = useState(false);
  const [confirmation, setConfirmation]   = useState<RefineResponse | null>(null);
  const [errorMsg, setErrorMsg]           = useState("");
  const [isSearching, setIsSearching]     = useState(false);

  const hasSignals = liked.size > 0 || disliked.size > 0 || userText.trim().length > 0;

  function toggleLike(name: string) {
    setLiked((prev) => {
      const next = new Set(prev);
      if (next.has(name)) { next.delete(name); return next; }
      next.add(name);
      // Can't be both liked and disliked
      setDisliked((d) => { const nd = new Set(d); nd.delete(name); return nd; });
      return next;
    });
  }

  function toggleDislike(name: string) {
    setDisliked((prev) => {
      const next = new Set(prev);
      if (next.has(name)) { next.delete(name); return next; }
      next.add(name);
      setLiked((l) => { const nl = new Set(l); nl.delete(name); return nl; });
      return next;
    });
  }

  function companyToFeedback(name: string): CompanyFeedback {
    const c = companies.find((x) => x.name === name);
    return {
      name,
      categories:    c?.categories,
      sdg_tags:      c?.sdg_tags,
      business_type: c?.business_type,
      city:          c?.city,
      country:       c?.country,
    };
  }

  async function handleRefine() {
    setIsLoading(true);
    setErrorMsg("");
    setConfirmation(null);
    try {
      const result = await refineSearch(sessionId, {
        liked:    Array.from(liked).map(companyToFeedback),
        disliked: Array.from(disliked).map(companyToFeedback),
        user_text: userText,
      });
      if (result.action === "unclear") {
        setErrorMsg(result.summary || "没有理解您的意图，请重新描述");
      } else {
        setConfirmation(result);
      }
    } catch (e: unknown) {
      setErrorMsg((e as Error).message || "Request failed");
    } finally {
      setIsLoading(false);
    }
  }

  async function handleConfirm() {
    if (!confirmation) return;
    setIsSearching(true);
    try {
      const res = await startSearch(confirmation.new_search_params);
      onNavigate(res.session_id);
    } catch (e: unknown) {
      setErrorMsg((e as Error).message || "Failed to start search");
      setIsSearching(false);
    }
  }

  return (
    <div className="space-y-4">
      {/* Company list */}
      <div className="space-y-2">
        {companies.map((company) => (
          <CompanyCard
            key={company.id || company.name}
            company={company}
            isLiked={liked.has(company.name)}
            isDisliked={disliked.has(company.name)}
            onLike={() => toggleLike(company.name)}
            onDislike={() => toggleDislike(company.name)}
          />
        ))}
      </div>

      {/* Signal summary */}
      {(liked.size > 0 || disliked.size > 0) && (
        <div className="bg-gray-50 border border-gray-200 rounded-xl px-5 py-3 text-sm space-y-1">
          <p className="font-medium text-navy text-xs uppercase tracking-wide mb-2">Current signals</p>
          {Array.from(liked).map((n) => (
            <p key={n} className="text-green-700">✓ More like <span className="font-medium">{n}</span></p>
          ))}
          {Array.from(disliked).map((n) => (
            <p key={n} className="text-red-600">✗ Exclude <span className="font-medium">{n}</span></p>
          ))}
        </div>
      )}

      {/* Text input + submit */}
      <div className="bg-white border border-gray-200 rounded-xl p-4 space-y-3">
        <textarea
          value={userText}
          onChange={(e) => setUserText(e.target.value)}
          placeholder="Describe any other adjustments... (e.g. switch to London, certified only)"
          rows={2}
          className="w-full text-sm text-navy placeholder-gray-400 resize-none outline-none border-none"
        />
        <div className="flex justify-end">
          <button
            onClick={handleRefine}
            disabled={!hasSignals || isLoading}
            className="text-sm bg-navy text-white px-5 py-2 rounded-lg hover:bg-navy/80 transition-colors disabled:opacity-40"
          >
            {isLoading ? "Analysing..." : "Preview Changes"}
          </button>
        </div>
      </div>

      {/* Error message — below the button so user sees it */}
      {errorMsg && (
        <div className="bg-red-50 border border-red-200 rounded-xl px-5 py-3 text-sm text-red-700">
          {errorMsg}
        </div>
      )}

      {/* Confirmation strip — below the button */}
      {confirmation && (
        <div className="bg-teal/10 border border-teal/30 rounded-xl px-5 py-4 space-y-3">
          <p className="text-sm text-navy">
            <span className="font-semibold">Will update: </span>
            {confirmation.summary}
          </p>
          {confirmation.modes && Object.keys(confirmation.modes).length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {Object.entries(confirmation.modes).map(([field, mode]) => (
                <span
                  key={field}
                  className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                    mode === "hard"
                      ? "bg-navy text-white"
                      : "bg-white border border-gray-300 text-gray-600"
                  }`}
                >
                  {field}: {mode === "hard" ? "Must" : "Prefer"}
                </span>
              ))}
            </div>
          )}
          {confirmation.rejected?.length > 0 && (
            <div className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 space-y-0.5">
              <p className="font-semibold mb-1">Some values were not recognised and will be skipped:</p>
              {confirmation.rejected.map((r, i) => <p key={i}>• {r}</p>)}
            </div>
          )}
          <div className="flex gap-2 justify-end">
            <button
              onClick={() => setConfirmation(null)}
              className="text-xs border border-gray-300 text-gray-600 px-3 py-1.5 rounded-lg hover:bg-gray-50 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleConfirm}
              disabled={isSearching}
              className="text-xs bg-teal text-white px-4 py-1.5 rounded-lg hover:bg-teal-light transition-colors disabled:opacity-60"
            >
              {isSearching ? "Starting..." : "Confirm & Re-search"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Company card (refine tab)
// ---------------------------------------------------------------------------

function CompanyCard({
  company,
  isLiked,
  isDisliked,
  onLike,
  onDislike,
}: {
  company: Company;
  isLiked: boolean;
  isDisliked: boolean;
  onLike: () => void;
  onDislike: () => void;
}) {
  const pct = Math.round((company.cross_encoder_score ?? 0) * 100);
  const qualityColor =
    company.match_quality === "strong"
      ? "text-green-600"
      : company.match_quality === "partial"
      ? "text-yellow-600"
      : "text-gray-400";

  return (
    <div
      className={`bg-white border rounded-xl px-4 py-3 flex items-center gap-3 transition-colors ${
        isLiked
          ? "border-green-300 bg-green-50"
          : isDisliked
          ? "border-red-200 bg-red-50"
          : "border-gray-200"
      }`}
    >
      {/* Info */}
      <div className="flex-1 min-w-0">
        <p className="font-semibold text-navy text-sm truncate">{company.name}</p>
        <p className="text-xs text-gray-500 truncate">
          {[company.categories, company.city, company.sdg_tags]
            .filter(Boolean)
            .join(" · ")}
        </p>
      </div>

      {/* Score */}
      <span className={`text-sm font-bold tabular-nums ${qualityColor}`}>
        {pct}%
      </span>

      {/* Buttons */}
      <div className="flex gap-1 shrink-0">
        <button
          onClick={onLike}
          title="More like this"
          className={`px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors ${
            isLiked
              ? "bg-green-600 text-white border-green-600"
              : "border-gray-300 text-gray-600 hover:border-green-500 hover:text-green-600"
          }`}
        >
          ＋ More
        </button>
        <button
          onClick={onDislike}
          title="Exclude this type"
          className={`px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors ${
            isDisliked
              ? "bg-red-500 text-white border-red-500"
              : "border-gray-300 text-gray-600 hover:border-red-400 hover:text-red-500"
          }`}
        >
          − Exclude
        </button>
      </div>
    </div>
  );
}


// ---------------------------------------------------------------------------
// Pipeline step indicator
// ---------------------------------------------------------------------------

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
        const isDone   = stepIdx < currentIdx;
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
