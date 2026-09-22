import type {
  HealthResponse,
  UnderwriteError,
  UnderwriteRequest,
  UnderwriteResponse,
  VerifierBlockedResponse,
} from "@/types/api"

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

async function postUnderwrite(
  path: string,
  payload: UnderwriteRequest,
): Promise<UnderwriteResponse> {
  let res: Response
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
  } catch {
    const err: UnderwriteError = {
      kind: "network",
      message:
        "Could not reach the underwriting service. Check your connection and try again.",
    }
    throw err
  }

  if (res.ok) {
    return (await res.json()) as UnderwriteResponse
  }

  let body: any = null
  try {
    body = await res.json()
  } catch {
    body = null
  }

  // Intercept the 422. FastAPI might return the payload directly, or wrapped in a "detail" object.
  if (res.status === 422 && body) {
    const verifierData = body.verification_passed !== undefined ? body : body.detail;
    
    if (verifierData && verifierData.verification_passed === false) {
      const err: UnderwriteError = {
        kind: "verifier_blocked",
        message: "Decision blocked by deterministic verifier.",
        detail: verifierData, // Passes the failures list to the UI
      }
      throw err
    }
  }

  if (res.status === 502) {
    const err: UnderwriteError = {
      kind: "bedrock",
      message:
        "The Bedrock LLM service is currently unavailable. No narrative could be synthesised.",
    }
    throw err
  }

  const err: UnderwriteError = {
    kind: "network",
    message: `The underwriting service returned an unexpected error (HTTP ${res.status}).`,
  }
  throw err
}

export function underwrite(
  payload: UnderwriteRequest,
): Promise<UnderwriteResponse> {
  return postUnderwrite("/api/v1/underwrite", payload)
}

export function verifyDemo(
  payload: UnderwriteRequest,
): Promise<UnderwriteResponse> {
  return postUnderwrite("/api/v1/underwrite/verify-demo", payload)
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/health`)
  if (!res.ok) {
    throw new Error(`Health check failed (HTTP ${res.status})`)
  }
  return (await res.json()) as HealthResponse
}