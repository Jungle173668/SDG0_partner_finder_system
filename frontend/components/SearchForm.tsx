"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import FilterRow, { FilterMode } from "./FilterRow";
import { getSchema, startSearch, Schema, FilterEntry, SearchRequest } from "@/lib/api";

// SDG full name → canonical number mapping (matches report_agent.py)
const SDG_NUMBER: Record<string, string> = {
  "no poverty": "SDG 1", "zero hunger": "SDG 2",
  "good health and well-being": "SDG 3", "quality education": "SDG 4",
  "gender equality": "SDG 5", "clean water and sanitation": "SDG 6",
  "affordable and clean energy": "SDG 7", "decent work and economic growth": "SDG 8",
  "industry innovation and infrastructure": "SDG 9",
  "industry innovation cities and communities": "SDG 9",
  "reduced inequalities": "SDG 10", "reduced inequality": "SDG 10",
  "sustainable cities and communities": "SDG 11",
  "responsible consumption and production": "SDG 12",
  "climate action": "SDG 13", "life below water": "SDG 14",
  "life on land": "SDG 15", "peace, justice and strong institutions": "SDG 16",
  "peace justice and strong institutions": "SDG 16", "partnerships for the goals": "SDG 17",
  "partnerships": "SDG 17",
};

// DB stores no-comma version; display layer uses official UN names with punctuation
const SDG_DISPLAY: Record<string, string> = {
  "industry innovation and infrastructure": "Industry, Innovation and Infrastructure",
};

function sdgOptionLabel(name: string): string {
  const key = name.toLowerCase().trim();
  const num = SDG_NUMBER[key];
  const display = SDG_DISPLAY[key] ?? name;
  return num ? `${num} · ${display}` : display;
}

// Sort SDG tags by their SDG number (1–17), unmapped tags go last
function sortSdgTags(tags: string[]): string[] {
  return [...tags].sort((a, b) => {
    const na = SDG_NUMBER[a.toLowerCase().trim()];
    const nb = SDG_NUMBER[b.toLowerCase().trim()];
    const numA = na ? parseInt(na.replace("SDG ", ""), 10) : 99;
    const numB = nb ? parseInt(nb.replace("SDG ", ""), 10) : 99;
    return numA - numB;
  });
}

type FilterState = {
  value: string | string[] | null;
  mode: FilterMode;
};

export default function SearchForm() {
  const router = useRouter();
  const [schema, setSchema] = useState<Schema | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Core text fields
  const [userDesc, setUserDesc] = useState("");
  const [partnerDesc, setPartnerDesc] = useState("");
  const [otherReq, setOtherReq] = useState("");

  // Filter states keyed by field name
  const [filters, setFilters] = useState<Record<string, FilterState>>({});
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [allowGlobalFallback, setAllowGlobalFallback] = useState(false);

  useEffect(() => {
    getSchema()
      .then(setSchema)
      .catch(() => {/* schema optional — still functional without it */});
  }, []);

  const handleFilterChange = (
    fieldName: string,
    value: string | string[] | null,
    mode: FilterMode
  ) => {
    setFilters((prev) => ({ ...prev, [fieldName]: { value, mode } }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!userDesc.trim()) {
      setError("Please describe your company first.");
      return;
    }
    setError("");
    setLoading(true);

    try {
      // Build request — only include filters with non-empty values
      const req: SearchRequest & Record<string, unknown> = {
        user_company_desc: userDesc,
        partner_type_desc: partnerDesc,
        other_requirements: otherReq,
        allow_global_fallback: allowGlobalFallback,
      };

      for (const [field, state] of Object.entries(filters)) {
        if (state.value !== null && state.value !== undefined) {
          const val = state.value;
          const isEmpty = Array.isArray(val) ? val.length === 0 : val === "";
          if (!isEmpty) {
            req[field] = { value: val, mode: state.mode } as FilterEntry;
          }
        }
      }

      const { session_id } = await startSearch(req as SearchRequest);
      router.push(`/results/${session_id}`);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Something went wrong. Please try again.");
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="max-w-3xl mx-auto px-4 py-10 space-y-6">
      {/* Page heading */}
      <div>
        <h1 className="text-3xl font-bold text-navy mb-1">Find Your Partner</h1>
        <p className="text-gray-500 text-sm">
          Describe your company and set filters. Our AI pipeline finds the best-matching
          sustainability partners from the SDG: Zero directory.
        </p>
      </div>

      {/* Core inputs */}
      <div className="form-card space-y-5">
        <div>
          <label className="block text-sm font-semibold text-navy mb-2">
            My company does... <span className="text-red-400">*</span>
          </label>
          <textarea
            value={userDesc}
            onChange={(e) => setUserDesc(e.target.value)}
            rows={4}
            placeholder="e.g. We provide corporate carbon footprint auditing and net-zero roadmap consulting for UK SMEs in the manufacturing sector..."
            className="w-full border border-gray-200 rounded-lg px-4 py-3 text-sm text-navy
                       focus:outline-none focus:ring-2 focus:ring-teal/40 focus:border-teal
                       resize-none transition-colors"
            required
          />
          <p className="text-xs text-gray-400 mt-1">
            This is the core anchor — the more specific, the better the results.
          </p>
        </div>

        <div>
          <label className="block text-sm font-semibold text-navy mb-2">
            I am looking for... <span className="text-gray-400 font-normal">(optional)</span>
          </label>
          <input
            type="text"
            value={partnerDesc}
            onChange={(e) => setPartnerDesc(e.target.value)}
            placeholder="e.g. a marketing or PR agency specialising in sustainability communications"
            className="w-full border border-gray-200 rounded-lg px-4 py-2.5 text-sm text-navy
                       focus:outline-none focus:ring-2 focus:ring-teal/40 focus:border-teal
                       transition-colors"
          />
        </div>

        <div>
          <label className="block text-sm font-semibold text-navy mb-2">
            Other requirements <span className="text-gray-400 font-normal">(optional)</span>
          </label>
          <textarea
            value={otherReq}
            onChange={(e) => setOtherReq(e.target.value)}
            rows={2}
            placeholder="e.g. prefer under 50 employees, willing to co-market, open to referral partnerships"
            className="w-full border border-gray-200 rounded-lg px-4 py-3 text-sm text-navy
                       focus:outline-none focus:ring-2 focus:ring-teal/40 focus:border-teal
                       resize-none transition-colors"
          />
        </div>
      </div>

      {/* Filter conditions */}
      <div className="form-card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-bold text-navy uppercase tracking-wide">Filter Conditions</h2>
          <div className="flex gap-3 text-xs text-gray-500">
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-3 rounded-full bg-navy"></span> Must = hard filter
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-3 rounded-full border border-gray-400 bg-white"></span> Prefer = soft boost
            </span>
          </div>
        </div>

        {schema ? (
          <>
            {/* Primary filters */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
              <FilterRow
                label="City"
                fieldName="city"
                options={schema.city}
                onChange={handleFilterChange}
              />
              <div className="sm:col-span-2">
                <FilterRow
                  label="SDG Goals (hold Ctrl/Cmd to select multiple)"
                  fieldName="sdg_tags"
                  options={sortSdgTags(schema.sdg_tags)}
                  multi
                  formatOption={sdgOptionLabel}
                  onChange={handleFilterChange}
                />
              </div>
            </div>

            {/* Advanced filters toggle */}
            <button
              type="button"
              onClick={() => setShowAdvanced((v) => !v)}
              className="mt-5 flex items-center gap-1.5 text-xs font-semibold text-gray-400 hover:text-navy transition-colors"
            >
              <svg
                className={`w-3.5 h-3.5 transition-transform ${showAdvanced ? "rotate-90" : ""}`}
                viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
              >
                <path d="M9 18l6-6-6-6" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Advanced filters
            </button>

            {showAdvanced && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-5 mt-4 pt-4 border-t border-gray-100">
                <FilterRow
                  label="Business Type"
                  fieldName="business_type"
                  options={schema.business_type}
                  onChange={handleFilterChange}
                />
                <FilterRow
                  label="Job Sector"
                  fieldName="job_sector"
                  options={schema.job_sector}
                  onChange={handleFilterChange}
                />
                <FilterRow
                  label="Company Size"
                  fieldName="company_size"
                  options={schema.company_size}
                  onChange={handleFilterChange}
                />
                <div className="sm:col-span-2">
                  <FilterRow
                    label="Categories (hold Ctrl/Cmd to select multiple)"
                    fieldName="categories"
                    options={schema.categories}
                    multi
                    onChange={handleFilterChange}
                  />
                </div>
              </div>
            )}
          </>
        ) : (
          <p className="text-sm text-gray-400 italic">
            Loading filter options... (you can still search without filters)
          </p>
        )}

        {/* Verified only checkbox */}
        <div className="mt-4 flex items-center gap-3">
          <input
            type="checkbox"
            id="claimed"
            onChange={(e) =>
              handleFilterChange(
                "claimed",
                e.target.checked ? "Yes" : null,
                filters["claimed"]?.mode ?? "soft"
              )
            }
            className="w-4 h-4 accent-teal cursor-pointer"
          />
          <label htmlFor="claimed" className="text-sm text-navy cursor-pointer">
            Verified profiles only
          </label>
          <button
            type="button"
            onClick={() =>
              handleFilterChange(
                "claimed",
                filters["claimed"]?.value ?? null,
                filters["claimed"]?.mode === "soft" ? "hard" : "soft"
              )
            }
            className={`mode-pill text-[10px] ${
              filters["claimed"]?.mode === "soft" ? "mode-pill-soft" : "mode-pill-inactive"
            }`}
          >
            {filters["claimed"]?.mode === "soft" ? "Prefer" : "Must"}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg px-4 py-3 text-sm">
          {error}
        </div>
      )}

      {/* Global fallback opt-in */}
      <label className="flex items-center gap-2 cursor-pointer select-none w-fit ml-auto">
        <input
          type="checkbox"
          checked={allowGlobalFallback}
          onChange={(e) => setAllowGlobalFallback(e.target.checked)}
          className="w-3.5 h-3.5 accent-teal"
        />
        <span className="text-xs text-gray-400">
          Expand search beyond filters if no strong matches found
        </span>
      </label>

      {/* Submit */}
      <div className="flex justify-end">
        <button
          type="submit"
          disabled={loading || !userDesc.trim()}
          className="btn-primary min-w-[160px]"
        >
          {loading ? (
            <span className="flex items-center gap-2 justify-center">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
              Starting...
            </span>
          ) : (
            "Find Partners"
          )}
        </button>
      </div>
    </form>
  );
}
