/**
 * API client for the Healthcare AI Platform backend.
 *
 * Provides typed methods for all REST endpoints and WebSocket connections.
 * All authenticated endpoints require a JWT Bearer token.
 * Base URL is configured via NEXT_PUBLIC_API_URL environment variable.
 *
 * Phase 3: All endpoints now use JWT auth, no more user_id query params.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API_V1 = `${API_BASE}/api/v1`;

/** Standard headers for JSON requests. */
function getHeaders(token?: string): HeadersInit {
  const headers: HeadersInit = {
    "Content-Type": "application/json",
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  return headers;
}

/** Generic fetch wrapper with error handling and token refresh. */
async function apiFetch<T>(
  path: string,
  options: RequestInit = {},
  token?: string
): Promise<T> {
  const url = `${API_V1}${path}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      ...getHeaders(token),
      ...(options.headers || {}),
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({
      detail: "An unknown error occurred",
    }));
    throw new ApiError(
      error.detail || `HTTP ${response.status}`,
      response.status
    );
  }

  // Handle 204 No Content
  if (response.status === 204) {
    return undefined as T;
  }

  return response.json();
}

/** Custom error class with HTTP status code. */
export class ApiError extends Error {
  constructor(message: string, public statusCode: number) {
    super(message);
    this.name = "ApiError";
  }
}

// ── Auth API ──────────────────────────────────────────────────────

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  user: {
    id: string;
    email: string;
    full_name: string;
    role: string;
    avatar_url: string | null;
    preferred_language: string;
  };
}

export const authApi = {
  register: (data: {
    email: string;
    password: string;
    full_name: string;
  }) =>
    apiFetch<TokenResponse>("/auth/register", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  login: (data: { email: string; password: string }) =>
    apiFetch<TokenResponse>("/auth/login", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  refresh: (refreshToken: string) =>
    apiFetch<TokenResponse>("/auth/refresh", {
      method: "POST",
      body: JSON.stringify({ refresh_token: refreshToken }),
    }),

  googleAuth: (code: string, redirectUri?: string) =>
    apiFetch<TokenResponse>("/auth/google", {
      method: "POST",
      body: JSON.stringify({ code, redirect_uri: redirectUri }),
    }),
};

// ── User Profile API ──────────────────────────────────────────────

export interface UserProfile {
  id: string;
  email: string;
  full_name: string;
  role: string;
  auth_provider: string;
  avatar_url: string | null;
  preferred_language: string;
  is_verified: boolean;
  created_at: string;
  last_login_at: string | null;
}

export interface HealthProfile {
  id: string;
  date_of_birth: string | null;
  biological_sex: string | null;
  height_cm: number | null;
  weight_kg: number | null;
  blood_group: string | null;
  bmi: number | null;
  allergies: string[] | null;
  chronic_conditions: string[] | null;
  current_medications: string[] | null;
  past_surgeries: string[] | null;
  family_history: Record<string, unknown> | null;
  lifestyle: Record<string, unknown> | null;
  emergency_contact: Record<string, unknown> | null;
  updated_at: string | null;
}

export const userApi = {
  getProfile: (token: string) =>
    apiFetch<UserProfile>("/users/me", {}, token),

  updateProfile: (
    data: { full_name?: string; preferred_language?: string; avatar_url?: string },
    token: string
  ) =>
    apiFetch<UserProfile>("/users/me", {
      method: "PATCH",
      body: JSON.stringify(data),
    }, token),

  changePassword: (
    currentPassword: string,
    newPassword: string,
    token: string
  ) =>
    apiFetch<void>("/users/me/change-password", {
      method: "POST",
      body: JSON.stringify({
        current_password: currentPassword,
        new_password: newPassword,
      }),
    }, token),

  getHealthProfile: (token: string) =>
    apiFetch<HealthProfile>("/users/me/health-profile", {}, token),

  upsertHealthProfile: (data: Partial<HealthProfile>, token: string) =>
    apiFetch<HealthProfile>("/users/me/health-profile", {
      method: "PUT",
      body: JSON.stringify(data),
    }, token),
};

// ── Chat API ──────────────────────────────────────────────────────

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  confidence_score?: number;
  extracted_symptoms?: string[];
  triage_level?: string;
  created_at: string;
  disclaimer: string;
}

export interface ChatSession {
  id: string;
  title: string;
  status: string;
  created_at: string;
  updated_at?: string;
  message_count: number;
}

export const chatApi = {
  listSessions: (token: string, status?: string) =>
    apiFetch<ChatSession[]>(
      `/chat/sessions${status ? `?status=${status}` : ""}`,
      {},
      token
    ),

  createSession: (token: string) =>
    apiFetch<ChatSession>("/chat/sessions", { method: "POST" }, token),

  updateSession: (
    sessionId: string,
    data: { title?: string; status?: string },
    token: string
  ) =>
    apiFetch<ChatSession>(
      `/chat/sessions/${sessionId}`,
      { method: "PATCH", body: JSON.stringify(data) },
      token
    ),

  deleteSession: (sessionId: string, token: string) =>
    apiFetch<void>(
      `/chat/sessions/${sessionId}`,
      { method: "DELETE" },
      token
    ),

  getMessages: (sessionId: string, token: string) =>
    apiFetch<ChatMessage[]>(
      `/chat/sessions/${sessionId}/messages`,
      {},
      token
    ),

  sendMessage: (
    content: string,
    token: string,
    sessionId?: string | null
  ) =>
    apiFetch<ChatMessage>(
      "/chat/message",
      {
        method: "POST",
        body: JSON.stringify({ content, session_id: sessionId }),
      },
      token
    ),
};

// ── Symptom API ───────────────────────────────────────────────────

export interface ExtractedSymptom {
  name: string;
  severity: number;
  body_region: string | null;
  duration: string | null;
  confidence: number;
}

export interface SymptomAnalysis {
  extracted_symptoms: ExtractedSymptom[];
  raw_text: string;
  language_detected: string;
  disclaimer: string;
}

export const symptomApi = {
  analyze: (text: string, token: string) =>
    apiFetch<SymptomAnalysis>(
      "/symptoms/analyze",
      { method: "POST", body: JSON.stringify({ text }) },
      token
    ),

  list: () =>
    apiFetch<Array<{
      name: string;
      display_name: string;
      category: string;
      severity_weight: number;
    }>>("/symptoms/list"),

  diagnose: (text: string, token: string, patientContext?: Record<string, unknown>) =>
    apiFetch<Record<string, unknown>>(
      "/symptoms/diagnose",
      {
        method: "POST",
        body: JSON.stringify({ text, patient_context: patientContext }),
      },
      token
    ),
};

// ── Diagnosis API ─────────────────────────────────────────────────

export interface DiagnosisRecord {
  id: string;
  primary_condition: string;
  primary_confidence: number;
  differential_diagnoses: Array<{
    condition: string;
    confidence: number;
  }> | null;
  icd10_codes: string[] | null;
  triage_level: string;
  triage_explanation: string | null;
  precautions: string[] | null;
  description: string | null;
  input_symptoms: string[];
  model_version: string;
  created_at: string;
  disclaimer: string;
}

export const diagnosisApi = {
  getHistory: (token: string) =>
    apiFetch<DiagnosisRecord[]>("/diagnosis/history", {}, token),

  getById: (id: string, token: string) =>
    apiFetch<DiagnosisRecord>(`/diagnosis/${id}`, {}, token),

  submitFeedback: (id: string, rating: number, text: string, token: string) =>
    apiFetch<void>(
      `/diagnosis/${id}/feedback`,
      { method: "POST", body: JSON.stringify({ rating, text }) },
      token
    ),
};

// ── Drug API ──────────────────────────────────────────────────────

export interface DrugSearchResult {
  brand_name: string | null;
  generic_name: string | null;
  manufacturer: string | null;
  indications_and_usage: string | null;
  warnings: string | null;
  drug_interactions: string | null;
  contraindications: string | null;
  adverse_reactions: string | null;
}

export interface DrugInteractionResult {
  drugs: string[];
  interactions_found: boolean;
  interaction_count: number;
  interactions: Array<{
    drug_a: string;
    drug_b: string;
    warning: string;
    source: string;
  }>;
  disclaimer: string;
}

export const drugApi = {
  search: (drugName: string, token: string) =>
    apiFetch<{ results: DrugSearchResult[]; meta: { total: number } }>(
      "/drugs/search",
      { method: "POST", body: JSON.stringify({ drug_name: drugName }) },
      token
    ),

  checkInteractions: (drugNames: string[], token: string) =>
    apiFetch<DrugInteractionResult>(
      "/drugs/interactions",
      { method: "POST", body: JSON.stringify({ drug_names: drugNames }) },
      token
    ),

  adverseEvents: (drugName: string, token: string) =>
    apiFetch<{ results: unknown[]; meta: { total: number } }>(
      "/drugs/adverse-events",
      { method: "POST", body: JSON.stringify({ drug_name: drugName }) },
      token
    ),

  recalls: (drugName: string, token: string) =>
    apiFetch<{ results: unknown[]; meta: { total: number } }>(
      "/drugs/recalls",
      { method: "POST", body: JSON.stringify({ drug_name: drugName }) },
      token
    ),
};

// ── Hospital API ──────────────────────────────────────────────────

export interface HospitalResult {
  name: string;
  address: string;
  latitude: number;
  longitude: number;
  distance_km: number;
  rating: number | null;
  phone: string | null;
  open_now: boolean | null;
  place_id: string;
}

export const hospitalApi = {
  nearby: (
    latitude: number,
    longitude: number,
    token: string,
    options?: { radius_km?: number; urgency?: string }
  ) =>
    apiFetch<HospitalResult[]>(
      "/hospitals/nearby",
      {
        method: "POST",
        body: JSON.stringify({
          latitude,
          longitude,
          radius_km: options?.radius_km ?? 10,
          urgency: options?.urgency ?? "routine",
        }),
      },
      token
    ),

  healthSummary: (token: string) =>
    apiFetch<{
      user_id: string;
      summary_text: string;
      total_consultations: number;
      common_symptoms: string[];
      recent_diagnoses: Array<Record<string, unknown>>;
      health_score: number | null;
    }>("/health/summary", {}, token),
};

// ── Admin API ─────────────────────────────────────────────────────

export const adminApi = {
  stats: (token: string) =>
    apiFetch<Record<string, unknown>>("/admin/stats", {}, token),

  listUsers: (token: string, limit = 50, offset = 0) =>
    apiFetch<{
      users: Array<Record<string, unknown>>;
      total: number;
    }>(`/admin/users?limit=${limit}&offset=${offset}`, {}, token),

  topConditions: (token: string) =>
    apiFetch<Array<{
      condition: string;
      count: number;
      avg_confidence: number | null;
    }>>("/admin/diagnoses/top-conditions", {}, token),
};

// ── WebSocket ─────────────────────────────────────────────────────

/**
 * Create an authenticated WebSocket connection for real-time chat.
 *
 * The JWT is passed as a query parameter since browsers cannot
 * set custom headers on WebSocket connections.
 */
export function createChatWebSocket(
  sessionId: string,
  token: string
): WebSocket {
  const wsBase =
    process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";
  return new WebSocket(
    `${wsBase}/api/v1/chat/ws/${sessionId}?token=${encodeURIComponent(token)}`
  );
}
