"use client"

import { useCallback, useState } from "react"

import { underwrite, verifyDemo } from "@/lib/api"
import type {
  UnderwriteError,
  UnderwriteRequest,
  UnderwriteResponse,
} from "@/types/api"

function toUnderwriteError(e: unknown): UnderwriteError {
  if (
    typeof e === "object" &&
    e !== null &&
    "kind" in e &&
    "message" in e
  ) {
    return e as UnderwriteError
  }
  return {
    kind: "network",
    message: "An unexpected error occurred while running the analysis.",
  }
}

export function useUnderwrite() {
  const [loading, setLoading] = useState(false)
  const [data, setData] = useState<UnderwriteResponse | null>(null)
  const [error, setError] = useState<UnderwriteError | null>(null)

  const run = useCallback(async (payload: UnderwriteRequest) => {
    setLoading(true)
    setError(null)
    setData(null)
    try {
      const result = await underwrite(payload)
      setData(result)
    } catch (e) {
      setError(toUnderwriteError(e))
    } finally {
      setLoading(false)
    }
  }, [])

  const runVerifierDemo = useCallback(async (payload: UnderwriteRequest) => {
    setLoading(true)
    setError(null)
    setData(null)
    try {
      const result = await verifyDemo(payload)
      setData(result)
    } catch (e) {
      setError(toUnderwriteError(e))
    } finally {
      setLoading(false)
    }
  }, [])

  const reset = useCallback(() => {
    setError(null)
    setData(null)
  }, [])

  return { loading, data, error, run, runVerifierDemo, reset }
}
