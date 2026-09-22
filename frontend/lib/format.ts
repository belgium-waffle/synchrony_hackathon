const currencyFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
})

export function formatCurrency(value: number): string {
  return currencyFormatter.format(value)
}

/** Renders a SHAP / log-odds number to exactly 4 decimals, never rounded for display beyond that precision. */
export function formatShap(value: number): string {
  return value.toFixed(4)
}

export function formatPercent(value: number, decimals = 1): string {
  return `${value.toFixed(decimals)}%`
}

/** probability_of_default is 0..1; render as a percentage with 2 decimals. */
export function formatPd(value: number): string {
  return `${(value * 100).toFixed(2)}%`
}

export function formatMs(value: number): string {
  return `${value.toFixed(1)} ms`
}
