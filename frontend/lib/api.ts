/**
 * API client for the FastAPI backend.
 * All requests go to /api/* — proxied to localhost:8000 in dev via next.config.js.
 */

const API_BASE = "/api";

export interface FilterEntry {
  value: string | string[] | boolean;
  mode: "hard" | "soft";
}

export interface SearchRequest {
  user_company_desc: string;
  partner_type_desc?: string;
  other_requirements?: string;
  city?: FilterEntry;
  business_type?: FilterEntry;
  job_sector?: FilterEntry;
  company_size?: FilterEntry;
  claimed?: FilterEntry;
  sdg_tags?: FilterEntry;
  categories?: FilterEntry;
}

export interface Company {
  id: string;
  slug: string;
  name: string;
  city?: string;
  country?: string;
  categories?: string;
  sdg_tags?: string;
  predicted_sdg_tags?: string;
  business_type?: string;
  website?: string;
  linkedin?: string;
  cross_encoder_score: number;
  match_quality: "strong" | "partial" | "fallback";
  reasoning: string;
  soft_filter_hit: string[];
}

export interface StatusResponse {
  session_id: string;
  status: "running" | "done" | "error";
  scored_companies?: Company[];
  search_fallback_level?: number;
  filters?: Record<string, unknown>;
  soft_filters?: Record<string, unknown>;
  partner_type_desc?: string;
  user_company_desc?: string;
  errors?: string[];
}

export interface Schema {
  city: string[];
  business_type: string[];
  job_sector: string[];
  company_size: string[];
  categories: string[];
  sdg_tags: string[];
}

export async function startSearch(req: SearchRequest): Promise<{ session_id: string }> {
  const res = await fetch(`${API_BASE}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Search failed: ${res.status}`);
  }
  return res.json();
}

export async function pollStatus(sessionId: string): Promise<StatusResponse> {
  const res = await fetch(`${API_BASE}/search/${sessionId}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Poll failed: ${res.status}`);
  }
  return res.json();
}

export async function getSchema(): Promise<Schema> {
  const res = await fetch(`${API_BASE}/schema`);
  if (!res.ok) throw new Error(`Schema fetch failed: ${res.status}`);
  return res.json();
}

/** Poll until status !== 'running', with configurable interval (ms). */
export async function waitForResult(
  sessionId: string,
  onPoll?: (attempt: number) => void,
  intervalMs = 3000,
  maxAttempts = 120,
): Promise<StatusResponse> {
  for (let i = 0; i < maxAttempts; i++) {
    onPoll?.(i);
    const status = await pollStatus(sessionId);
    if (status.status !== "running") return status;
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error("Pipeline timed out after 6 minutes");
}
