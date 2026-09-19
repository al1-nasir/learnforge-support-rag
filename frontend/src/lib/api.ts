import type { ChatRequest, ChatResponse } from "./types";

const DEFAULT_API_BASE_URL = "http://localhost:8000";

export function getApiBaseUrl(): string {
  return (
    import.meta.env.VITE_API_BASE_URL?.trim() ||
    DEFAULT_API_BASE_URL
  ).replace(/\/+$/, "");
}

export class ApiError extends Error {
  status?: number;
  details?: unknown;

  constructor(
    message: string,
    status?: number,
    details?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
  }
}

export async function sendChatMessage(
  request: ChatRequest,
  abortSignal?: AbortSignal,
): Promise<ChatResponse> {
  const baseUrl = getApiBaseUrl();
  const url = `${baseUrl}/chat`;

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        session_id: request.session_id,
        message: request.message,
      }),
      signal: abortSignal,
    });

    if (!response.ok) {
      let errorMessage = `Server returned status ${response.status}`;
      try {
        const errorJson = await response.json();
        if (errorJson?.detail) {
          errorMessage = typeof errorJson.detail === "string"
            ? errorJson.detail
            : JSON.stringify(errorJson.detail);
        }
      } catch {
        // Fallback to generic status message
      }
      throw new ApiError(errorMessage, response.status);
    }

    const data: ChatResponse = await response.json();
    return data;
  } catch (err: unknown) {
    if (err instanceof ApiError) {
      throw err;
    }
    if (err instanceof Error && err.name === "AbortError") {
      throw err;
    }
    const msg = err instanceof Error ? err.message : "Network error";
    throw new ApiError(`Failed to connect to LearnForge backend: ${msg}`);
  }
}
