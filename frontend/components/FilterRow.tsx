"use client";

/**
 * FilterRow — a single filter field with a Hard/Soft mode toggle.
 *
 * Hard (Must): value is sent to SearchAgent WHERE clause — strict match.
 * Soft (Prefer): value is sent to ScoringAgent as bonus — boosts rank, doesn't exclude.
 */

import { useState } from "react";

export type FilterMode = "hard" | "soft";

interface FilterRowProps {
  label: string;
  fieldName: string;
  options: string[];
  multi?: boolean;
  formatOption?: (value: string) => string;  // optional display label formatter
  onChange: (fieldName: string, value: string | string[] | null, mode: FilterMode) => void;
}

export default function FilterRow({ label, fieldName, options, multi = false, formatOption, onChange }: FilterRowProps) {
  const [mode, setMode] = useState<FilterMode>("soft");
  const [selected, setSelected] = useState<string[]>([]);

  const toggleMode = () => {
    const next: FilterMode = mode === "hard" ? "soft" : "hard";
    setMode(next);
    if (selected.length > 0) {
      onChange(fieldName, multi ? selected : selected[0], next);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    let vals: string[];
    if (multi) {
      vals = Array.from(e.target.selectedOptions, (o) => o.value).filter(Boolean);
    } else {
      vals = e.target.value ? [e.target.value] : [];
    }
    setSelected(vals);
    onChange(fieldName, multi ? (vals.length ? vals : null) : (vals[0] || null), mode);
  };

  return (
    <div className="flex items-start gap-3">
      <div className="flex-1 min-w-0">
        <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
          {label}
        </label>
        {multi ? (
          <select
            multiple
            className="filter-select h-24"
            onChange={handleChange}
          >
            {options.map((o) => (
              <option key={o} value={o}>
                {formatOption ? formatOption(o) : o}
              </option>
            ))}
          </select>
        ) : (
          <select className="filter-select" onChange={handleChange} defaultValue="">
            <option value="">— any —</option>
            {options.map((o) => (
              <option key={o} value={o}>
                {formatOption ? formatOption(o) : o}
              </option>
            ))}
          </select>
        )}
      </div>

      {/* Hard / Soft toggle */}
      <div className="mt-6 flex flex-col gap-1">
        <button
          type="button"
          onClick={toggleMode}
          title={mode === "hard" ? "Click to switch to Preferred (soft filter)" : "Click to switch to Must (hard filter)"}
          className={`mode-pill ${
            mode === "hard" ? "mode-pill-hard" : "mode-pill-soft"
          }`}
        >
          {mode === "hard" ? "Must" : "Prefer"}
        </button>
        <span className="text-[10px] text-gray-400 text-center leading-tight">
          {mode === "hard" ? "strict" : "bonus"}
        </span>
        <span className={`text-[9px] text-gray-300 text-center leading-tight max-w-[52px] ${mode === "hard" ? "" : "invisible"}`}>
          may limit results
        </span>
      </div>
    </div>
  );
}
