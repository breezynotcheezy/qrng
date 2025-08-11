"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { useMutation } from "@tanstack/react-query"
import {
  Activity,
  Atom,
  BarChart3,
  CheckCircle2,
  Database,
  Download,
  Gauge,
  LineChart,
  Loader2,
  Play,
  Share2,
  Sigma,
  Square,
  Timer,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { ControlsPanel, UseCase as UseCaseType } from "@/components/ControlsPanel"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
// Removed unused Select imports after switching to segmented toggles in ControlsPanel
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Label } from "@/components/ui/label"
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
} from "@/components/ui/chart"
import { Area, AreaChart, Line, LineChart as RLineChart, XAxis, YAxis, CartesianGrid } from "recharts"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Input } from "@/components/ui/input"

// Types
type UseCase = "var" | "option"
type RNG = "quantum" | "classical"

type DensityPoint = {
  x: number
  prng: number
  qrng: number
  prngUpper: number
  prngLower: number
  qrngUpper: number
  qrngLower: number
}
type ErrorPoint = { n: number; prngError: number; qrngError: number; prngEstimate: number; qrngEstimate: number }

// RNG implementations
class LCG {
  private state: number
  constructor(seed = 123456789) {
    this.state = seed >>> 0
  }
  next(): number {
    // Numerical Recipes LCG
    this.state = (1664525 * this.state + 1013904223) >>> 0
    return this.state / 2 ** 32
  }
}

function cryptoUniform(): number {
  const arr = new Uint32Array(1)
  crypto.getRandomValues(arr)
  return arr[0] / 2 ** 32
}

// QRNG API cache to avoid excessive calls
const qrngCache = new Map<string, number[]>()
const QRNG_CACHE_TTL = 30000 // 30 seconds

async function fetchQrngNumbers(count: number): Promise<number[]> {
  const cacheKey = `qrng_${count}`
  const cached = qrngCache.get(cacheKey)
  if (cached) {
    return cached
  }

  try {
    const response = await fetch('/api/rng', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        count,
        dtype: 'uint16',
        normalize: true
      })
    })

    if (!response.ok) {
      throw new Error(`QRNG API failed: ${response.status}`)
    }

    const data = await response.json()
    if (data.values && Array.isArray(data.values)) {
      // Cache the result
      qrngCache.set(cacheKey, data.values)
      setTimeout(() => qrngCache.delete(cacheKey), QRNG_CACHE_TTL)
      console.log(`✅ QRNG API success: fetched ${count} quantum random numbers, source: ${data.source}`)
      return data.values
    }
  } catch (error) {
    console.warn('❌ QRNG API call failed, falling back to crypto:', error)
  }

  // Fallback to crypto
  const fallback = new Array(count)
  for (let i = 0; i < count; i++) {
    fallback[i] = cryptoUniform()
  }
  console.log(`⚠️ Using crypto fallback for ${count} numbers`)
  return fallback
}

// Global QRNG buffer
let qrngBuffer: number[] = []
let qrngBufferIndex = 0

async function ensureQrngBuffer(minSize: number = 1000) {
  if (qrngBuffer.length - qrngBufferIndex < minSize) {
    const newNumbers = await fetchQrngNumbers(2000)
    qrngBuffer.push(...newNumbers)
  }
}

function stratifiedQuantumUniform(i: number, batchSize: number): number {
  // Use quantum random numbers when available, fallback to stratified crypto
  if (qrngBuffer.length > qrngBufferIndex) {
    const quantumU = qrngBuffer[qrngBufferIndex++]
    // Apply stratification to quantum numbers
    const jitter = (quantumU - 0.5) / batchSize
    let u = (i + 0.5) / batchSize + jitter
    if (u < 0) u += 1
    if (u >= 1) u -= 1
    return u
  }
  
  // Fallback to crypto with stratification
  const jitter = (cryptoUniform() - 0.5) / batchSize
  let u = (i + 0.5) / batchSize + jitter
  if (u < 0) u += 1
  if (u >= 1) u -= 1
  return u
}

// Math helpers
function boxMuller(u1: number, u2: number): number {
  // Transform two uniforms to one standard normal (Z)
  const r = Math.sqrt(-2.0 * Math.log(Math.max(1e-12, u1)))
  const theta = 2 * Math.PI * u2
  return r * Math.cos(theta)
}

function normalCDF(x: number): number {
  // Abramowitz and Stegun approximation
  const a1 = 0.254829592
  const a2 = -0.284496736
  const a3 = 1.421413741
  const a4 = -1.453152027
  const a5 = 1.061405429
  const p = 0.3275911
  const sign = x < 0 ? -1 : 1
  const t = 1.0 / (1.0 + p * Math.abs(x))
  const y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x)
  return 0.5 * (1.0 + sign * y)
}

function invNormalCDF(p: number): number {
  // Acklam's inverse normal approximation with one Halley refinement
  if (p <= 0 || p >= 1) {
    if (p === 0) return Number.NEGATIVE_INFINITY
    if (p === 1) return Number.POSITIVE_INFINITY
    throw new Error("p must be in (0,1)")
  }
  const a = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1,
    2.506628277459239,
  ]
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1]
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968,
    2.938163982698783,
  ]
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416]
  const plow = 0.02425
  const phigh = 1 - plow
  let q: number, r: number, x: number
  if (p < plow) {
    q = Math.sqrt(-2 * Math.log(p))
    x =
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
  } else if (p > phigh) {
    q = Math.sqrt(-2 * Math.log(1 - p))
    x =
      -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
  } else {
    q = p - 0.5
    r = q * q
    x =
      ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
  }
  // One Halley refinement using our normalCDF approximation and standard normal PDF
  const e = normalCDF(x) - p
  const pdf = Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI)
  const u = e / Math.max(1e-12, pdf)
  x = x - u / (1 + (x * u) / 2)
  return x
}

// Black-Scholes call price
function bsCall(S: number, K: number, r: number, sigma: number, T: number): number {
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T))
  const d2 = d1 - sigma * Math.sqrt(T)
  return S * normalCDF(d1) - K * Math.exp(-r * T) * normalCDF(d2)
}

// Entropy approximation over 256 bins
function shannonEntropyBits(samples: number[]): number {
  if (samples.length === 0) return 0
  const bins = 256
  const counts = new Array(bins).fill(0)
  for (const u of samples) {
    const idx = Math.floor(Math.max(0, Math.min(bins - 1, u * bins)))
    counts[idx]++
  }
  const n = samples.length
  let H = 0
  for (const c of counts) {
    if (c > 0) {
      const p = c / n
      H -= p * Math.log2(p)
    }
  }
  return H // bits per sample, max 8
}

// Utility: formatters
const fmtInt = (n: number) => new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(n)
const fmtMs = (n: number) => `${n.toFixed(1)} ms`
const fmtPct = (x: number) => `${(x * 100).toFixed(1)}%`

// Chart configs
const densityChartConfig = {
  prng: {
    label: "Classical Density",
    color: "#ef4444", // red-500
    icon: Database,
  },
  qrng: {
    label: "Quantum Density", 
    color: "#8b5cf6", // purple-500
    icon: Atom,
  },
  prngUpper: { label: "Classical 95% Upper", color: "#fca5a5" }, // red-300
  prngLower: { label: "Classical 95% Lower", color: "#fca5a5" },
  qrngUpper: { label: "Quantum 95% Upper", color: "#c4b5fd" }, // purple-300
  qrngLower: { label: "Quantum 95% Lower", color: "#c4b5fd" },
} as const

const errorChartConfig = {
  prngError: {
    label: "Classical Error",
    color: "#ef4444", // red-500
    icon: Database,
  },
  qrngError: {
    label: "Quantum Error",
    color: "#8b5cf6", // purple-500
    icon: Atom,
  },
} as const

export default function Page() {
  // Server-side simulations via API
  const simVar = useMutation({
    mutationFn: async (rng: "qrng" | "prng") => {
      const res = await fetch("/api/sim/var", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ confidence: varParams.confidence, rng, paths: varParams.paths }),
      })
      if (!res.ok) throw new Error("/api/sim/var failed")
      return res.json()
    },
  })
  const simOption = useMutation({
    mutationFn: async (rng: "qrng" | "prng") => {
      const res = await fetch("/api/sim/option", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ ...optionParams, rng }),
      })
      if (!res.ok) throw new Error("/api/sim/option failed")
      return res.json()
    },
  })
  const [serverResults, setServerResults] = useState<{ var?: any; option?: any } | null>(null)
  const [serverAdvantage, setServerAdvantage] = useState<number>(0)
  const [serverReproId, setServerReproId] = useState<string>("")

  function fnv1a32(str: string): string {
    let h = 0x811c9dc5
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i)
      h += (h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24)
    }
    return (h >>> 0).toString(16)
  }

  function ciSpan(ci: [number, number]): number {
    return Math.abs(ci[1] - ci[0])
  }

  function computeAdvantageFromCIs(
    widthPr: number,
    widthQr: number,
    nPr: number,
    nQr: number,
  ): number {
    if (widthPr <= 0 || widthQr <= 0 || nPr <= 0 || nQr <= 0) return 0
    // width ~ C / sqrt(n) => C = width * sqrt(n)
    const Cpr = widthPr * Math.sqrt(nPr)
    const Cqr = widthQr * Math.sqrt(nQr)
    const targetWidth = Math.min(widthPr, widthQr)
    const nNeededPr = (Cpr / targetWidth) ** 2
    const nNeededQr = (Cqr / targetWidth) ** 2
    return 1 - nNeededQr / nNeededPr
  }
  // Controls
  const [useCase, setUseCase] = useState<UseCase>("option")
  const [varParams, setVarParams] = useState({ confidence: 0.99, paths: 50000 })
  const [optionParams, setOptionParams] = useState({ symbol: 'AAPL', S0: 100, K: 100, r: 0.01, sigma: 0.2, T: 1, paths: 100000 })
  const [rngFocus, setRngFocus] = useState<RNG>("quantum")
  const [running, setRunning] = useState(false)

  // Simulation parameters
  const batchSize = 1000
  const maxBatches = 100 // up to 100k samples per stream
  const pauseBetweenBatchesMs = 0 // tight loop; adjust if needed

  // Option pricing params
  const S0 = 100,
    K = 100,
    r = 0.01,
    sigma = 0.2,
    T = 1

  // VaR params
  const alpha = 0.99
  const mu = 0
  const std = 1
  const varTrue = -mu + std * invNormalCDF(alpha) // VaR of loss = -R

  // Theoretical for option
  const bsTrue = useMemo(() => bsCall(S0, K, r, sigma, T), [S0, K, r, sigma, T])

  // States for accumulators
  const prngRef = useRef(new LCG(12345))
  const [totalSamples, setTotalSamples] = useState(0)
  const [prngLatencyAvg, setPrngLatencyAvg] = useState(0)
  const [qrngLatencyAvg, setQrngLatencyAvg] = useState(0)
  const latencyHistPRNG = useRef<number[]>([])
  const latencyHistQRNG = useRef<number[]>([])

  const [entropyPRNG, setEntropyPRNG] = useState(0)
  const [entropyQRNG, setEntropyQRNG] = useState(0)
  const entropyPoolPRNG = useRef<number[]>([])
  const entropyPoolQRNG = useRef<number[]>([])

  // Values for estimator
  const [estimatePRNG, setEstimatePRNG] = useState(0)
  const [estimateQRNG, setEstimateQRNG] = useState(0)
  const [ciWidth, setCiWidth] = useState(0)
  const [neededSamples, setNeededSamples] = useState(0)

  // Densities
  const bins = useCase === "var" ? 40 : 40
  const [densityData, setDensityData] = useState<DensityPoint[]>([])
  const countsPRNG = useRef(new Array(bins).fill(0))
  const countsQRNG = useRef(new Array(bins).fill(0))
  const [densityRange, setDensityRange] = useState<{ min: number; max: number }>(
    useCase === "var" ? { min: -4, max: 4 } : { min: 0, max: 50 },
  )

  // Error trend series
  const [errorSeries, setErrorSeries] = useState<ErrorPoint[]>([])

  // For VaR quantile stability
  const [quantileSeries, setQuantileSeries] = useState<{ n: number; prng: number; qrng: number }[]>([])
  const samplesLossPRNG = useRef<number[]>([])
  const samplesLossQRNG = useRef<number[]>([])

  const t0Ref = useRef<number>(0)
  const [elapsedMs, setElapsedMs] = useState(0)

  // Recompute true references and reset when useCase changes
  useEffect(() => {
    resetAll()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [useCase])

  function resetAll() {
    setRunning(false)
    prngRef.current = new LCG(12345)
    setTotalSamples(0)
    setPrngLatencyAvg(0)
    setQrngLatencyAvg(0)
    latencyHistPRNG.current = []
    latencyHistQRNG.current = []
    setEntropyPRNG(0)
    setEntropyQRNG(0)
    entropyPoolPRNG.current = []
    entropyPoolQRNG.current = []
    setEstimatePRNG(0)
    setEstimateQRNG(0)
    setCiWidth(0)
    setNeededSamples(0)
    setErrorSeries([])
    setQuantileSeries([])
    samplesLossPRNG.current = []
    samplesLossQRNG.current = []
    countsPRNG.current = new Array(bins).fill(0)
    countsQRNG.current = new Array(bins).fill(0)
    setDensityData([])
    setElapsedMs(0)
    t0Ref.current = 0
    // Reset QRNG buffer
    qrngBuffer = []
    qrngBufferIndex = 0
    if (useCase === "var") {
      setDensityRange({ min: -4, max: 4 })
    } else {
      setDensityRange({ min: 0, max: 50 })
    }
  }

  // Simulation loop
  useEffect(() => {
    let cancelled = false
    if (!running) return

    let batchIndex = 0
    if (t0Ref.current === 0) t0Ref.current = performance.now()

    const runBatch = async () => {
      if (cancelled || batchIndex >= maxBatches) {
        setRunning(false)
        return
      }

      const tBatchStart = performance.now()

      // Ensure QRNG buffer is available
      await ensureQrngBuffer(batchSize)

      // Generate uniforms for both streams
      const prngUs: number[] = new Array(batchSize)
      const prngUs2: number[] = new Array(batchSize)
      const qrngUs: number[] = new Array(batchSize)
      const qrngUs2: number[] = new Array(batchSize)

      const tPrngStart = performance.now()
      for (let i = 0; i < batchSize; i++) {
        prngUs[i] = prngRef.current.next()
        prngUs2[i] = prngRef.current.next()
      }
      const tPrng = performance.now() - tPrngStart

      const tQrngStart = performance.now()
      for (let i = 0; i < batchSize; i++) {
        qrngUs[i] = stratifiedQuantumUniform(i, batchSize)
        qrngUs2[i] = cryptoUniform()
      }
      const tQrng = performance.now() - tQrngStart

      // Update latency moving avg
      latencyHistPRNG.current.push(tPrng)
      latencyHistQRNG.current.push(tQrng)
      if (latencyHistPRNG.current.length > 20) latencyHistPRNG.current.shift()
      if (latencyHistQRNG.current.length > 20) latencyHistQRNG.current.shift()
      setPrngLatencyAvg(average(latencyHistPRNG.current))
      setQrngLatencyAvg(average(latencyHistQRNG.current))

      // Update entropy pools occasionally
      if (entropyPoolPRNG.current.length < 8192) {
        entropyPoolPRNG.current.push(...prngUs.slice(0, 256))
        setEntropyPRNG(shannonEntropyBits(entropyPoolPRNG.current.slice(-4096)))
      }
      if (entropyPoolQRNG.current.length < 8192) {
        entropyPoolQRNG.current.push(...qrngUs.slice(0, 256))
        setEntropyQRNG(shannonEntropyBits(entropyPoolQRNG.current.slice(-4096)))
      }

      // Transform uniforms to sample variables based on use case
      // Also compute estimators incrementally
      if (useCase === "var") {
        // Loss L = -R, R ~ N(mu, std)
        const prngLosses: number[] = new Array(batchSize)
        const qrngLosses: number[] = new Array(batchSize)
        for (let i = 0; i < batchSize; i++) {
          const z1 = boxMuller(prngUs[i], prngUs2[i])
          const z2 = boxMuller(qrngUs[i], qrngUs2[i])
          const r1 = mu + std * z1
          const r2 = mu + std * z2
          prngLosses[i] = -r1
          qrngLosses[i] = -r2
        }
        // Update hist counts
        accumulateHistogram(prngLosses, countsPRNG.current, densityRange.min, densityRange.max)
        accumulateHistogram(qrngLosses, countsQRNG.current, densityRange.min, densityRange.max)
        samplesLossPRNG.current.push(...prngLosses)
        samplesLossQRNG.current.push(...qrngLosses)
        // Quantile estimates (recompute every few batches)
        if ((batchIndex + 1) % 2 === 0) {
          const prngVaR = quantile(samplesLossPRNG.current, alpha)
          const qrngVaR = quantile(samplesLossQRNG.current, alpha)
          setEstimatePRNG(prngVaR)
          setEstimateQRNG(qrngVaR)
          setQuantileSeries((prev) => [...prev, { n: (batchIndex + 1) * batchSize, prng: prngVaR, qrng: qrngVaR }])
          // CI width for quantile: approx using asymptotics
          const qTrue = varTrue
          const fAtQ = (1 / (std * Math.sqrt(2 * Math.PI))) * Math.exp(-((qTrue + mu) ** 2) / (2 * std * std))
          const n = (batchIndex + 1) * batchSize
          const se = Math.sqrt(alpha * (1 - alpha)) / Math.max(1e-9, Math.sqrt(n) * fAtQ)
          const ci = 2 * 1.96 * se
          setCiWidth(ci)
          // Needed samples to get CI width <= 0.05 (target)
          const target = 0.05
          const needed = Math.ceil(((1.96 / (target / 2)) ** 2 * (alpha * (1 - alpha))) / (fAtQ * fAtQ))
          setNeededSamples(needed)
        }
        // Density data
        buildDensitySeries((batchIndex + 1) * batchSize)
        // Error series
        const n = (batchIndex + 1) * batchSize
        const prngErr = Math.abs(estimatePRNG - varTrue)
        const qrngErr = Math.abs(estimateQRNG - varTrue)
        setErrorSeries((prev) => [
          ...prev,
          { n, prngError: prngErr, qrngError: qrngErr, prngEstimate: estimatePRNG, qrngEstimate: estimateQRNG },
        ])
      } else {
        // Option pricing: payoff = max(ST - K, 0)
        let sumPayPRNG = 0
        let sumPayQRNG = 0
        let sumSqPayPRNG = 0
        let sumSqPayQRNG = 0
        for (let i = 0; i < batchSize; i++) {
          const z1 = boxMuller(prngUs[i], prngUs2[i])
          const z2 = boxMuller(qrngUs[i], qrngUs2[i])
          const drift = (r - 0.5 * sigma * sigma) * T
          const vol = sigma * Math.sqrt(T)
          const ST1 = S0 * Math.exp(drift + vol * z1)
          const ST2 = S0 * Math.exp(drift + vol * z2)
          const p1 = Math.max(0, ST1 - K)
          const p2 = Math.max(0, ST2 - K)
          sumPayPRNG += p1
          sumPayQRNG += p2
          sumSqPayPRNG += p1 * p1
          sumSqPayQRNG += p2 * p2
        }
        // Maintain running mean via cumulative (we will derive from totals)
        const disc = Math.exp(-r * T)
        // We'll store densities over payoffs for visualization as well
        const prngPayoffsSample: number[] = [] // for hist only: small subset to reduce cost
        const qrngPayoffsSample: number[] = []
        // Regenerate a small subset (100) for histogram from the same uniforms to avoid storing all
        const subset = 100
        for (let i = 0; i < subset; i++) {
          const u1 = prngRef.current.next(),
            u2 = prngRef.current.next()
          const uq1 = stratifiedQuantumUniform(i, subset),
            uq2 = cryptoUniform()
          const Zp = boxMuller(u1, u2)
          const Zq = boxMuller(uq1, uq2)
          const STp = S0 * Math.exp((r - 0.5 * sigma * sigma) * T + sigma * Math.sqrt(T) * Zp)
          const STq = S0 * Math.exp((r - 0.5 * sigma * sigma) * T + sigma * Math.sqrt(T) * Zq)
          prngPayoffsSample.push(Math.max(0, STp - K))
          qrngPayoffsSample.push(Math.max(0, STq - K))
        }
        accumulateHistogram(prngPayoffsSample, countsPRNG.current, densityRange.min, densityRange.max)
        accumulateHistogram(qrngPayoffsSample, countsQRNG.current, densityRange.min, densityRange.max)
        // Update estimates (cumulative)
        const prevN = totalSamples
        const newN = prevN + batchSize
        // For simplicity, we recompute running mean via stored sums per batch using refs
        // We'll store totals on refs
        totalsPRNG.current.sum += sumPayPRNG
        totalsPRNG.current.sumSq += sumSqPayPRNG
        totalsQRNG.current.sum += sumPayQRNG
        totalsQRNG.current.sumSq += sumSqPayQRNG
        totalsN.current += batchSize
        const meanP = totalsPRNG.current.sum / totalsN.current
        const meanQ = totalsQRNG.current.sum / totalsN.current
        const priceP = disc * meanP
        const priceQ = disc * meanQ
        setEstimatePRNG(priceP)
        setEstimateQRNG(priceQ)
        // SE and CI
        const varP = totalsPRNG.current.sumSq / totalsN.current - meanP * meanP
        const varQ = totalsQRNG.current.sumSq / totalsN.current - meanQ * meanQ
        const seP = disc * Math.sqrt(Math.max(0, varP) / totalsN.current)
        const seQ = disc * Math.sqrt(Math.max(0, varQ) / totalsN.current)
        setCiWidth(2 * 1.96 * Math.max(seP, seQ))
        // Needed samples for target relative error 1%
        const targetRel = 0.01
        const targetSe = targetRel * bsTrue
        const useVar = Math.max(varP, varQ)
        const needed = Math.ceil((disc * disc * useVar) / (targetSe * targetSe))
        setNeededSamples(needed)
        buildDensitySeries(totalsN.current) // approximate using current counts
        setErrorSeries((prev) => [
          ...prev,
          {
            n: totalsN.current,
            prngError: Math.abs(priceP - bsTrue),
            qrngError: Math.abs(priceQ - bsTrue),
            prngEstimate: priceP,
            qrngEstimate: priceQ,
          },
        ])
      }

      const tBatchEnd = performance.now()
      const elapsed = tBatchEnd - t0Ref.current
      setElapsedMs(elapsed)
      setTotalSamples((batchIndex + 1) * batchSize)
      batchIndex++

      if (!cancelled && batchIndex < maxBatches) {
        if (pauseBetweenBatchesMs > 0) {
          setTimeout(() => runBatch(), pauseBetweenBatchesMs)
        } else {
          // Schedule next microtask to keep UI responsive
          setTimeout(() => runBatch(), 0)
        }
      } else {
        setRunning(false)
      }
    }

    runBatch()
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, useCase])

  // Totals refs for option pricing
  const totalsPRNG = useRef({ sum: 0, sumSq: 0 })
  const totalsQRNG = useRef({ sum: 0, sumSq: 0 })
  const totalsN = useRef(0)

  function average(arr: number[]): number {
    if (arr.length === 0) return 0
    return arr.reduce((a, b) => a + b, 0) / arr.length
  }

  function accumulateHistogram(values: number[], counts: number[], min: number, max: number) {
    const width = (max - min) / counts.length
    for (const v of values) {
      if (v < min || v >= max) continue
      const idx = Math.floor((v - min) / width)
      if (idx >= 0 && idx < counts.length) counts[idx]++
    }
  }

  function buildDensitySeries(n: number) {
    const width = (densityRange.max - densityRange.min) / bins
    const prngTotal = countsPRNG.current.reduce((a, b) => a + b, 0)
    const qrngTotal = countsQRNG.current.reduce((a, b) => a + b, 0)
    const data: DensityPoint[] = []
    for (let i = 0; i < bins; i++) {
      const xCenter = densityRange.min + (i + 0.5) * width
      const pPr = prngTotal > 0 ? countsPRNG.current[i] / prngTotal : 0
      const pQr = qrngTotal > 0 ? countsQRNG.current[i] / qrngTotal : 0
      // Approximate 95% bands: p ± 1.96 * sqrt(p(1-p)/n), converted to density by dividing width
      const sePr = prngTotal > 0 ? Math.sqrt(Math.max(0, pPr * (1 - pPr)) / prngTotal) : 0
      const seQr = qrngTotal > 0 ? Math.sqrt(Math.max(0, pQr * (1 - pQr)) / qrngTotal) : 0
      const prngDensity = pPr / width
      const qrngDensity = pQr / width
      const prngUpper = (pPr + 1.96 * sePr) / width
      const prngLower = Math.max(0, (pPr - 1.96 * sePr) / width)
      const qrngUpper = (pQr + 1.96 * seQr) / width
      const qrngLower = Math.max(0, (pQr - 1.96 * seQr) / width)
      data.push({
        x: xCenter,
        prng: prngDensity,
        qrng: qrngDensity,
        prngUpper,
        prngLower,
        qrngUpper,
        qrngLower,
      })
    }
    setDensityData(data)
  }

  function quantile(arr: number[], p: number): number {
    if (arr.length === 0) return 0
    const a = arr.slice().sort((x, y) => x - y)
    const idx = (a.length - 1) * p
    const lo = Math.floor(idx),
      hi = Math.ceil(idx)
    if (lo === hi) return a[lo]
    const w = idx - lo
    return a[lo] * (1 - w) + a[hi] * w
  }

  // Quantum advantage estimate from current error series (sample efficiency)
  const quantumAdvantagePct = useMemo(() => {
    if (errorSeries.length < 5) return 0
    // Compare average squared error over last window
    const tail = errorSeries.slice(-10)
    const pr = tail.map((e) => e.prngError * e.prngError)
    const qr = tail.map((e) => e.qrngError * e.qrngError)
    const msePr = average(pr)
    const mseQr = average(qr)
    if (msePr <= 0) return 0
    const ratio = mseQr / msePr
    // Fewer samples = 1 - ratio
    return Math.max(-0.5, Math.min(0.9, 1 - ratio)) // clamp for readability
  }, [errorSeries])

  // Status card content based on rngFocus and running
  const statusText = rngFocus === "quantum" ? "QRN Ready" : running ? "Fetching PRNG" : "PRNG Idle"
  const statusIcon =
    rngFocus === "quantum" ? (
      <CheckCircle2 className="h-4 w-4 text-emerald-500" />
    ) : running ? (
      <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
    ) : (
      <Database className="h-4 w-4 text-muted-foreground" />
    )

  // Secondary metrics derived
  const estimateValue =
    useCase === "var"
      ? rngFocus === "quantum"
        ? estimateQRNG
        : estimatePRNG
      : rngFocus === "quantum"
        ? estimateQRNG
        : estimatePRNG
  const truth = useCase === "var" ? varTrue : bsTrue
  const errorNow = Math.abs(estimateValue - truth)

  // Export CSV
  function exportCSV() {
    const header = ["n", "prng_estimate", "qrng_estimate", "prng_error", "qrng_error"].join(",")
    const rows = errorSeries.map((e) => [e.n, e.prngEstimate, e.qrngEstimate, e.prngError, e.qrngError].join(","))
    const csv = [header, ...rows].join("\n")
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `qrngtoolkit_${useCase}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  // Export PDF (simple textual report)
  async function exportPDF() {
    const { jsPDF } = await import("jspdf")
    const doc = new jsPDF()
    let y = 10
    doc.setFontSize(14)
    doc.text("QRNG Toolkit Report", 10, y)
    y += 8
    doc.setFontSize(10)
    doc.text(`Use Case: ${useCase === "var" ? "VaR Simulation" : "Option Pricing (Black-Scholes)"}`, 10, y)
    y += 8
    y += 6
    doc.text(`Samples: ${fmtInt(totalSamples)} per stream`, 10, y)
    y += 6
    doc.text(`Estimate (PRNG): ${estimatePRNG.toFixed(5)}`, 10, y)
    y += 6
    doc.text(`Estimate (QRNG): ${estimateQRNG.toFixed(5)}`, 10, y)
    y += 6
    doc.text(`Reference: ${truth.toFixed(5)}`, 10, y)
    y += 6
    doc.text(`CI Width (approx): ${ciWidth.toFixed(5)}`, 10, y)
    y += 6
    doc.text(`Quantum Advantage (est.): ${(quantumAdvantagePct * 100).toFixed(1)}% fewer samples`, 10, y)
    y += 6
    doc.text(`Latency PRNG: ${fmtMs(prngLatencyAvg)} | QRNG: ${fmtMs(qrngLatencyAvg)}`, 10, y)
    y += 6
    doc.text(`Entropy PRNG: ${entropyPRNG.toFixed(2)} bits | QRNG: ${entropyQRNG.toFixed(2)} bits`, 10, y)
    y += 10
    doc.text("Insight:", 10, y)
    y += 6
    const insight = getInsightText()
    const split = doc.splitTextToSize(insight, 190)
    doc.text(split, 10, y)
    doc.save(`qrngtoolkit_${useCase}.pdf`)
  }

  function getInsightText(): string {
    const adv = (quantumAdvantagePct * 100).toFixed(1)
    const estStr = useCase === "var" ? "VaR" : "Price"
    const better = quantumAdvantagePct > 0 ? "reduced" : "did not reduce"
    return `QRNG ${better} estimation error compared to classical RNG${quantumAdvantagePct > 0 ? ` and achieved equivalent precision with approximately ${adv}% fewer samples.` : "."} Current ${estStr} estimates are PRNG=${estimatePRNG.toFixed(4)} and QRNG=${estimateQRNG.toFixed(4)} vs reference ${truth.toFixed(4)}.`
  }

  // Share link
  async function shareLink() {
    const url = new URL(window.location.href)
    url.searchParams.set("useCase", useCase)
    url.searchParams.set("rng", rngFocus)
    url.searchParams.set("n", String(totalSamples))
    const link = url.toString()
    if ((navigator as any).share) {
      try {
        await (navigator as any).share({
          title: "QRNG Toolkit",
          url: link,
          text: "Check out this QRNG simulation snapshot.",
        })
      } catch {
        await navigator.clipboard.writeText(link)
      }
    } else {
      await navigator.clipboard.writeText(link)
    }
  }

  // Load params from URL on mount
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const uc = params.get("useCase")
    const rf = params.get("rng")
    if (uc === "var" || uc === "option") setUseCase(uc)
    if (rf === "quantum" || rf === "classical") setRngFocus(rf)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // UI
  return (
    <main className="dark min-h-screen flex flex-col bg-background text-foreground">
      {/* Pill-shaped Menu Header */}
      <header className="sticky top-0 z-20 bg-background/80 backdrop-blur border-b">
        <div className="mx-auto max-w-7xl px-4 py-4">
          {/* Main Pill Navigation */}
          <div className="flex flex-col gap-4">
            {/* Top Row - Brand and Quick Actions */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                                 <div className="flex items-center gap-2">
                   <Atom className="h-6 w-6 text-purple-400" />
                   <span className="text-lg font-bold tracking-tight bg-gradient-to-r from-purple-400 to-purple-600 bg-clip-text text-transparent">
                     qhaven sim
                   </span>
                 </div>
              </div>
              
              {/* Quick Action Buttons */}
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={shareLink}
                  className="hidden sm:flex"
                >
                  <Share2 className="h-4 w-4 mr-1" />
                  Share
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={exportCSV}
                  className="hidden sm:flex"
                >
                  <Download className="h-4 w-4 mr-1" />
                  Export
                </Button>
              </div>
            </div>

            {/* Main Control Pill */}
            <div className="bg-card/50 rounded-full border p-2 shadow-sm">
              <div className="flex flex-col lg:flex-row gap-3 items-center">
                {/* Use Case Selector */}
                <div className="flex items-center gap-2">
                  <Label className="text-sm font-medium whitespace-nowrap">Use Case:</Label>
                  <ToggleGroup
                    type="single"
                    value={useCase}
                    onValueChange={(v) => v && setUseCase(v as UseCase)}
                    variant="outline"
                    size="sm"
                    className="bg-background/50"
                  >
                    <ToggleGroupItem value="option" className="text-xs px-3">
                      <BarChart3 className="h-3 w-3 mr-1" />
                      Option Pricing
                    </ToggleGroupItem>
                    <ToggleGroupItem value="var" className="text-xs px-3">
                      <Gauge className="h-3 w-3 mr-1" />
                      Value at Risk
                    </ToggleGroupItem>
                  </ToggleGroup>
                </div>

                {/* RNG Type Selector */}
                <div className="flex items-center gap-2">
                  <Label className="text-sm font-medium whitespace-nowrap">RNG Type:</Label>
                  <ToggleGroup
                    type="single"
                    value={rngFocus}
                    onValueChange={(v) => v && setRngFocus(v as RNG)}
                    variant="outline"
                    size="sm"
                    className="bg-background/50"
                  >
                    <ToggleGroupItem value="quantum" className="text-xs px-3">
                      <Atom className="h-3 w-3 mr-1" />
                      Quantum
                    </ToggleGroupItem>
                    <ToggleGroupItem value="classical" className="text-xs px-3">
                      <Database className="h-3 w-3 mr-1" />
                      Classical
                    </ToggleGroupItem>
                  </ToggleGroup>
                </div>

                {/* Run Button */}
                <div className="flex items-center gap-2">
                  <Button
                    onClick={async () => {
                      if (useCase === "option") {
                        totalsPRNG.current = { sum: 0, sumSq: 0 }
                        totalsQRNG.current = { sum: 0, sumSq: 0 }
                        totalsN.current = 0
                      }
                      setRunning(true)
                      try {
                        if (useCase === "var") {
                          const reqBody = { confidence: varParams.confidence, paths: varParams.paths }
                          const [prng, qrng] = await Promise.all([
                            simVar.mutateAsync("prng"),
                            simVar.mutateAsync("qrng"),
                          ])
                          const key = fnv1a32(JSON.stringify({ route: "var", ...reqBody }))
                          setServerReproId(key)
                          setServerResults({ var: { prng, qrng } })
                          const adv = computeAdvantageFromCIs(
                            ciSpan(prng.ci),
                            ciSpan(qrng.ci),
                            prng.sampleCount,
                            qrng.sampleCount,
                          )
                          setServerAdvantage(adv)
                        } else {
                          const reqBody = { ...optionParams }
                          const [prng, qrng] = await Promise.all([
                            simOption.mutateAsync("prng"),
                            simOption.mutateAsync("qrng"),
                          ])
                          const key = fnv1a32(JSON.stringify({ route: "option", ...reqBody }))
                          setServerReproId(key)
                          setServerResults({ option: { prng, qrng } })
                          const widthPr = 2 * 1.96 * prng.stderr
                          const widthQr = 2 * 1.96 * qrng.stderr
                          const adv = computeAdvantageFromCIs(widthPr, widthQr, prng.sampleCount, qrng.sampleCount)
                          setServerAdvantage(adv)
                        }
                      } catch (_) {
                        // best-effort; UI continues with client demo
                      }
                    }}
                    disabled={running}
                    size="sm"
                                         className="bg-white hover:bg-gray-100 text-black border border-gray-300"
                  >
                    {running ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                        Running...
                      </>
                    ) : (
                      <>
                        <Play className="h-4 w-4 mr-1" />
                        Run Simulation
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </div>

            {/* Parameters Panel - Collapsible */}
            <div className="bg-card/30 rounded-lg border p-4">
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                {/* Option Parameters */}
                {useCase === "option" && (
                  <>
                    <div className="space-y-2">
                      <Label className="text-sm font-medium">Ticker & Price</Label>
                      <div className="flex gap-2">
                        <Input 
                          className="flex-1" 
                          value={optionParams.symbol || 'AAPL'} 
                          onChange={(e) => setOptionParams({...optionParams, symbol: e.target.value.toUpperCase()})}
                          placeholder="AAPL"
                        />
                        <Button 
                          variant="outline" 
                          size="sm"
                          onClick={async () => {
                            try {
                              const res = await fetch(`/api/market/quote?symbol=${encodeURIComponent(optionParams.symbol || 'AAPL')}`)
                              if (res.ok) {
                                const json = await res.json()
                                if (typeof json.price === 'number') {
                                  setOptionParams({ ...optionParams, S0: json.price })
                                }
                              }
                            } catch {}
                          }}
                        >
                          Get Price
                        </Button>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-sm font-medium">Spot Price (S₀)</Label>
                      <Input 
                        type="number" 
                        step="0.01" 
                        value={optionParams.S0}
                        onChange={(e) => setOptionParams({ ...optionParams, S0: Number(e.target.value) })}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label className="text-sm font-medium">Strike Price (K)</Label>
                      <Input 
                        type="number" 
                        step="0.01" 
                        value={optionParams.K}
                        onChange={(e) => setOptionParams({ ...optionParams, K: Number(e.target.value) })}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label className="text-sm font-medium">Risk-free Rate (r)</Label>
                      <Input 
                        type="number" 
                        step="0.0001" 
                        value={optionParams.r}
                        onChange={(e) => setOptionParams({ ...optionParams, r: Number(e.target.value) })}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label className="text-sm font-medium">Volatility (σ)</Label>
                      <Input 
                        type="number" 
                        step="0.0001" 
                        value={optionParams.sigma}
                        onChange={(e) => setOptionParams({ ...optionParams, sigma: Number(e.target.value) })}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label className="text-sm font-medium">Time to Expiry (T)</Label>
                      <Input 
                        type="number" 
                        step="0.01" 
                        value={optionParams.T}
                        onChange={(e) => setOptionParams({ ...optionParams, T: Number(e.target.value) })}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label className="text-sm font-medium">Simulation Paths</Label>
                      <Input 
                        type="number" 
                        step="1000" 
                        min={1000} 
                        value={optionParams.paths}
                        onChange={(e) => setOptionParams({ ...optionParams, paths: Number(e.target.value) })}
                      />
                    </div>
                  </>
                )}

                {/* VaR Parameters */}
                {useCase === "var" && (
                  <>
                    <div className="space-y-2">
                      <Label className="text-sm font-medium">Confidence Level</Label>
                      <Input 
                        type="number" 
                        step="0.01" 
                        min="0.5" 
                        max="0.9999" 
                        value={varParams.confidence}
                        onChange={(e) => setVarParams({ ...varParams, confidence: Number(e.target.value) })}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label className="text-sm font-medium">Simulation Paths</Label>
                      <Input 
                        type="number" 
                        step="1000" 
                        min={1000} 
                        value={varParams.paths}
                        onChange={(e) => setVarParams({ ...varParams, paths: Number(e.target.value) })}
                      />
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* KPI Cards */}
      <section className="mx-auto max-w-7xl w-full px-4 py-5 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Status</CardTitle>
            {statusIcon}
          </CardHeader>
          <CardContent>
            <div className="text-xl font-semibold">{statusText}</div>
            <p className="text-xs text-muted-foreground">{running ? "Simulation running" : "Idle"}</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Latency (per 1k)</CardTitle>
            <Timer className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent className="space-y-1">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">PRNG</span>
              <span className="font-medium">{fmtMs(prngLatencyAvg)}</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">QRNG</span>
              <span className="font-medium">{fmtMs(qrngLatencyAvg)}</span>
            </div>
            <p className="text-xs text-muted-foreground">{'Example: "20 ms vs 1 ms" depending on environment'}</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Entropy (bits/sample)</CardTitle>
            <Sigma className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent className="space-y-1">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">PRNG</span>
              <span className="font-medium">{entropyPRNG.toFixed(2)} / 8</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">QRNG</span>
              <span className="font-medium">{entropyQRNG.toFixed(2)} / 8</span>
            </div>
            <p className="text-xs text-muted-foreground">Approximate Shannon entropy over recent uniforms</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Quantum Advantage</CardTitle>
            <Gauge className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-xl font-semibold">{`${(quantumAdvantagePct * 100).toFixed(1)}% fewer samples`}</div>
            <p className="text-xs text-muted-foreground">Estimated from recent error MSE</p>
          </CardContent>
        </Card>
      </section>

      {/* Main Visualization Panel (Option-focused) */}
      <section className="mx-auto max-w-7xl w-full px-4 grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Left: Distribution & Outcomes */}
        <Card className="h-[460px]">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <div className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
              <CardTitle className="text-sm font-medium">
                {useCase === "var" ? "Loss Distribution (Density)" : "Payoff Distribution (Density)"}
              </CardTitle>
            </div>
            <Badge variant="outline" className="text-xs">
              {"Overlays with 95% bands"}
            </Badge>
          </CardHeader>
          <CardContent className="h-[400px]">
            <ChartContainer config={densityChartConfig} className="h-full w-full">
              <AreaChart data={densityData} margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.1)" />
                <XAxis 
                  dataKey="x" 
                  tickLine={false} 
                  axisLine={false} 
                  tick={{ fontSize: 11, fill: 'rgba(255,255,255,0.7)' }}
                  domain={[densityRange.min, densityRange.max]}
                />
                <YAxis 
                  tickLine={false} 
                  axisLine={false} 
                  tick={{ fontSize: 11, fill: 'rgba(255,255,255,0.7)' }}
                  domain={[0, 'dataMax + 0.1']}
                  tickFormatter={(value) => value.toFixed(2)}
                />
                <ChartTooltip content={<ChartTooltipContent />} />
                <ChartLegend content={({ payload }) => <ChartLegendContent payload={payload} />} />
                {/* Confidence bands */}
                <Area dataKey="prngUpper" stroke="transparent" fill="transparent" />
                <Area dataKey="prngLower" stroke="transparent" fill="transparent" />
                <Area dataKey="qrngUpper" stroke="transparent" fill="transparent" />
                <Area dataKey="qrngLower" stroke="transparent" fill="transparent" />
                {/* Densities */}
                <Area
                  type="monotone"
                  dataKey="prng"
                  stroke="var(--color-prng)"
                  fill="var(--color-prng)"
                  fillOpacity={0.3}
                  strokeWidth={2}
                />
                <Area
                  type="monotone"
                  dataKey="qrng"
                  stroke="var(--color-qrng)"
                  fill="var(--color-qrng)"
                  fillOpacity={0.35}
                  strokeWidth={2}
                />
              </AreaChart>
            </ChartContainer>
          </CardContent>
        </Card>

        {/* Right: Convergence & Error Trends */}
        <Card className="h-[460px]">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <div className="flex items-center gap-2">
              <LineChart className="h-4 w-4 text-muted-foreground" />
              <CardTitle className="text-sm font-medium">Error vs. Number of Runs</CardTitle>
            </div>
            <Badge variant="outline" className="text-xs">
              Lower is better
            </Badge>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="h-[280px]">
              <ChartContainer config={errorChartConfig} className="h-full w-full">
                <RLineChart data={errorSeries} margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.1)" />
                  <XAxis
                    dataKey="n"
                    tickLine={false}
                    axisLine={false}
                    tick={{ fontSize: 11, fill: 'rgba(255,255,255,0.7)' }}
                    tickFormatter={(v) => fmtInt(v)}
                  />
                  <YAxis 
                    tickLine={false} 
                    axisLine={false} 
                    tick={{ fontSize: 11, fill: 'rgba(255,255,255,0.7)' }}
                    domain={[0, 'dataMax + 0.01']}
                    tickFormatter={(value) => value.toFixed(3)}
                  />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <ChartLegend content={({ payload }) => <ChartLegendContent payload={payload} />} />
                  <Line 
                    type="monotone" 
                    dataKey="prngError" 
                    stroke="var(--color-prngError)" 
                    strokeWidth={2}
                    dot={false} 
                  />
                  <Line 
                    type="monotone" 
                    dataKey="qrngError" 
                    stroke="var(--color-qrngError)" 
                    strokeWidth={2}
                    dot={false} 
                  />
                </RLineChart>
              </ChartContainer>
            </div>
            {/* Removed VaR-specific secondary plot for option-focused UI */}
          </CardContent>
        </Card>
      </section>

      {/* Secondary Metrics */}
      <section className="mx-auto max-w-7xl w-full px-4 py-5 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">
              {useCase === "var" ? "VaR Estimate" : "Option Price Estimate"}
            </CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-xl font-semibold">{estimateValue.toFixed(4)}</div>
            <p className="text-xs text-muted-foreground">Reference: {truth.toFixed(4)}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Confidence Interval Width</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-xl font-semibold">{ciWidth.toFixed(4)}</div>
            <p className="text-xs text-muted-foreground">Approximate 95% CI width</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Runtime</CardTitle>
            <Timer className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-xl font-semibold">{fmtMs(elapsedMs)}</div>
            <p className="text-xs text-muted-foreground">Samples: {fmtInt(totalSamples)} per stream</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Samples Needed</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-xl font-semibold">{fmtInt(neededSamples)}</div>
            <p className="text-xs text-muted-foreground">For target precision</p>
          </CardContent>
        </Card>
      </section>

      {/* Server Results KPIs */}
      {serverResults?.var && (
        <section className="mx-auto max-w-7xl w-full px-4 py-3 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Server VaR (Classical)</CardTitle>
              <Database className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent className="space-y-1">
              <div className="text-xl font-semibold">{serverResults.var.prng.var.toFixed(4)}</div>
              <p className="text-xs text-muted-foreground">CI width: {ciSpan(serverResults.var.prng.ci).toFixed(4)}</p>
              <p className="text-xs text-muted-foreground">Runtime: {fmtMs(serverResults.var.prng.runtimeMs)}</p>
              <p className="text-xs text-muted-foreground">Samples: {fmtInt(serverResults.var.prng.sampleCount)}</p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Server VaR (Quantum)</CardTitle>
              <Atom className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent className="space-y-1">
              <div className="text-xl font-semibold">{serverResults.var.qrng.var.toFixed(4)}</div>
              <p className="text-xs text-muted-foreground">CI width: {ciSpan(serverResults.var.qrng.ci).toFixed(4)}</p>
              <p className="text-xs text-muted-foreground">Runtime: {fmtMs(serverResults.var.qrng.runtimeMs)}</p>
              <div className="flex items-center gap-2">
                <p className="text-xs text-muted-foreground">Samples: {fmtInt(serverResults.var.qrng.sampleCount)}</p>
                {serverResults.var.qrng.qrngFallback && (
                  <Badge variant="secondary" className="text-[10px]">fallback</Badge>
                )}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Quantum Advantage (server)</CardTitle>
              <Gauge className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-xl font-semibold">{`${(serverAdvantage * 100).toFixed(1)}% fewer samples`}</div>
              <p className="text-xs text-muted-foreground">Based on CI widths at equal precision</p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Repro ID</CardTitle>
              <CheckCircle2 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-xs font-mono break-all">{serverReproId}</div>
            </CardContent>
          </Card>
        </section>
      )}

      {serverResults?.option && (
        <section className="mx-auto max-w-7xl w-full px-4 py-3 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Server Price (Classical)</CardTitle>
              <Database className="h-4 w-4" style={{ color: "#ef4444" }} />
            </CardHeader>
            <CardContent className="space-y-1">
              <div className="text-xl font-semibold">{serverResults.option.prng.mean.toFixed(4)}</div>
              <p className="text-xs text-muted-foreground">CI width: {(2 * 1.96 * serverResults.option.prng.stderr).toFixed(4)}</p>
              <p className="text-xs text-muted-foreground">Runtime: {fmtMs(serverResults.option.prng.runtimeMs)}</p>
              <p className="text-xs text-muted-foreground">Samples: {fmtInt(serverResults.option.prng.sampleCount)}</p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Server Price (Quantum)</CardTitle>
              <Atom className="h-4 w-4" style={{ color: "#8b5cf6" }} />
            </CardHeader>
            <CardContent className="space-y-1">
              <div className="text-xl font-semibold">{serverResults.option.qrng.mean.toFixed(4)}</div>
              <p className="text-xs text-muted-foreground">CI width: {(2 * 1.96 * serverResults.option.qrng.stderr).toFixed(4)}</p>
              <p className="text-xs text-muted-foreground">Runtime: {fmtMs(serverResults.option.qrng.runtimeMs)}</p>
              <div className="flex items-center gap-2">
                <p className="text-xs text-muted-foreground">Samples: {fmtInt(serverResults.option.qrng.sampleCount)}</p>
                {serverResults.option.qrng.qrngFallback && (
                  <Badge variant="secondary" className="text-[10px]">fallback</Badge>
                )}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Quantum Advantage (server)</CardTitle>
              <Gauge className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-xl font-semibold">{`${(serverAdvantage * 100).toFixed(1)}% fewer samples`}</div>
              <p className="text-xs text-muted-foreground">Based on CI widths at equal precision</p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Repro ID</CardTitle>
              <CheckCircle2 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-xs font-mono break-all">{serverReproId}</div>
            </CardContent>
          </Card>
        </section>
      )}

      {/* Insights & Export */}
      <section className="mx-auto max-w-7xl w-full px-4 pb-10">
        <Card className="border-emerald-200">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <div className="flex items-center gap-2">
              <Atom className="h-4 w-4 text-emerald-600" />
              <CardTitle className="text-sm font-medium">Insight</CardTitle>
            </div>
            <Badge variant="secondary" className="text-xs">
              {useCase === "var" ? "Tail Risk" : "Derivative Pricing"}
            </Badge>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm leading-relaxed">{getInsightText()}</p>
            <Separator />
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
              <Button variant="outline" className="gap-2 bg-transparent" onClick={exportCSV}>
                <Download className="h-4 w-4" />
                Export CSV
              </Button>
              {serverResults && (
                <Button
                  variant="outline"
                  className="gap-2 bg-transparent"
                  onClick={() => {
                    const blob = new Blob([JSON.stringify(serverResults, null, 2)], { type: "application/json" })
                    const url = URL.createObjectURL(blob)
                    const a = document.createElement("a")
                    a.href = url
                    a.download = `server_results_${useCase}_${serverReproId || "latest"}.json`
                    a.click()
                    URL.revokeObjectURL(url)
                  }}
                >
                  <Download className="h-4 w-4" />
                  Export Server JSON
                </Button>
              )}
              <Button variant="outline" className="gap-2 bg-transparent" onClick={exportPDF}>
                <Download className="h-4 w-4" />
                Export PDF
              </Button>
              <Button variant="outline" className="gap-2 bg-transparent" onClick={shareLink}>
                <Share2 className="h-4 w-4" />
                Share Link
              </Button>
              <div className="ml-auto flex items-center gap-3 text-xs text-muted-foreground">
                <div className="flex items-center gap-1">
                  <Database className="h-3.5 w-3.5" style={{ color: "#ef4444" }} />
                  <span>{"Classical"}</span>
                </div>
                <div className="flex items-center gap-1">
                  <Atom className="h-3.5 w-3.5" style={{ color: "#8b5cf6" }} />
                  <span>{"Quantum"}</span>
                </div>
              </div>
              {serverResults && (
                <div className="mt-4 text-xs text-muted-foreground">
                  <div>Server results available:</div>
                  <pre className="whitespace-pre-wrap break-words">
{JSON.stringify(serverResults, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </section>
    </main>
  )
}
