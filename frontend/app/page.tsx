"use client"

import { useState } from "react"
import { useForm } from "react-hook-form"
import { zodResolver } from "@hookform/resolvers/zod"

import { CenterColumn } from "@/components/dashboard/center-column"
import { IntakeForm } from "@/components/dashboard/intake-form"
import { RightColumn } from "@/components/dashboard/right-column"
import { TopBar } from "@/components/dashboard/top-bar"
import { useUnderwrite } from "@/hooks/use-underwrite"
import {
  defaultFormValues,
  toUnderwriteRequest,
  underwriteFormSchema,
  type UnderwriteFormValues,
} from "@/lib/underwrite-form"

export default function Page() {
  const form = useForm<UnderwriteFormValues>({
    resolver: zodResolver(underwriteFormSchema),
    defaultValues: defaultFormValues,
  })
  const { loading, data, error, run, runVerifierDemo } = useUnderwrite()
  const [hasSubmitted, setHasSubmitted] = useState(false)

  const submit = (values: UnderwriteFormValues) => {
    setHasSubmitted(true)
    run(toUnderwriteRequest(values))
  }

  const verifierDemo = () => {
    setHasSubmitted(true)
    runVerifierDemo(toUnderwriteRequest(form.getValues()))
  }

  const retry = () => {
    setHasSubmitted(true)
    run(toUnderwriteRequest(form.getValues()))
  }

  return (
    <div className="dark flex h-dvh flex-col bg-slate-950 text-slate-100">
      <TopBar onVerifierDemo={verifierDemo} demoDisabled={loading} />
      <main className="grid min-h-0 flex-1 grid-cols-1 gap-4 overflow-y-auto p-4 lg:grid-cols-[360px_1fr_400px] lg:overflow-hidden">
        <div className="min-h-0 lg:overflow-y-auto">
          <IntakeForm form={form} onSubmit={submit} loading={loading} />
        </div>
        <div className="min-h-0 lg:overflow-y-auto">
          <CenterColumn
            loading={loading}
            data={data}
            error={error}
            hasSubmitted={hasSubmitted}
          />
        </div>
        <div className="min-h-0 lg:overflow-y-auto">
          <RightColumn
            loading={loading}
            data={data}
            error={error}
            hasSubmitted={hasSubmitted}
            onRetry={retry}
          />
        </div>
      </main>
    </div>
  )
}
