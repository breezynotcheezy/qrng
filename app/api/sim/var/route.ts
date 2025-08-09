import { NextResponse } from 'next/server'
import { z } from 'zod'

const VarRequestSchema = z.object({
  returns: z.array(z.number()).optional(),
  confidence: z.number().min(0.5).max(0.9999).default(0.99),
  rng: z.enum(['qrng', 'prng']).default('prng'),
  paths: z.number().int().positive().max(500_000).default(50_000),
})

type VarRequest = z.infer<typeof VarRequestSchema>

function invNormalCDF(p: number): number {
  if (p <= 0 || p >= 1) {
    if (p === 0) return Number.NEGATIVE_INFINITY
    if (p === 1) return Number.POSITIVE_INFINITY
    throw new Error('p must be in (0,1)')
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
    x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
  } else if (p > phigh) {
    q = Math.sqrt(-2 * Math.log(1 - p))
    x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
  } else {
    q = p - 0.5
    r = q * q
    x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
  }
  const e = 0.5 * (1 + erf(x / Math.SQRT2)) - p
  const pdf = Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI)
  const u = e / Math.max(1e-12, pdf)
  x = x - u / (1 + (x * u) / 2)
  return x
}

function erf(x: number): number {
  const sign = x < 0 ? -1 : 1
  const a1 = 0.254829592
  const a2 = -0.284496736
  const a3 = 1.421413741
  const a4 = -1.453152027
  const a5 = 1.061405429
  const p = 0.3275911
  const t = 1.0 / (1.0 + p * Math.abs(x))
  const y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x)
  return sign * y
}

function quantile(sortedValues: number[], p: number): number {
  const idx = (sortedValues.length - 1) * p
  const lo = Math.floor(idx)
  const hi = Math.ceil(idx)
  if (lo === hi) return sortedValues[lo]
  const w = idx - lo
  return sortedValues[lo] * (1 - w) + sortedValues[hi] * w
}

function getUniformsPrng(n: number): number[] {
  const out = new Float64Array(n)
  if (typeof crypto !== 'undefined' && 'getRandomValues' in crypto) {
    // Web Crypto limits to 65,536 bytes per call. Uint32Array => max 16,384 elements per call.
    const MAX_U32_PER_CALL = 16384
    let offset = 0
    while (offset < n) {
      const len = Math.min(MAX_U32_PER_CALL, n - offset)
      const buf = new Uint32Array(len)
      crypto.getRandomValues(buf)
      for (let i = 0; i < len; i++) out[offset + i] = buf[i] / 2 ** 32
      offset += len
    }
  } else {
    let state = 123456789 >>> 0
    for (let i = 0; i < n; i++) {
      state = (1664525 * state + 1013904223) >>> 0
      out[i] = state / 2 ** 32
    }
  }
  return Array.from(out)
}

async function getUniformsQrng(n: number, origin: string): Promise<number[]> {
  const chunkSize = 1024 // ANU API is more reliable with small batches
  const values: number[] = new Array(n)
  let written = 0
  while (written < n) {
    const len = Math.min(chunkSize, n - written)
    const res = await fetch(`${origin}/api/rng`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ count: len, dtype: 'uint16', normalize: true }),
      cache: 'no-store',
    })
    if (!res.ok) throw new Error(`QRNG service failed: ${res.status}`)
    const json = await res.json()
    if (!Array.isArray(json.values)) throw new Error('Bad QRNG payload')
    for (let i = 0; i < len; i++) values[written + i] = json.values[i]
    written += len
  }
  return values
}

export async function POST(request: Request) {
  let body: VarRequest
  try {
    body = VarRequestSchema.parse(await request.json())
  } catch (err) {
    return NextResponse.json({ error: 'Invalid request', details: String(err) }, { status: 400 })
  }

  const t0 = performance.now()

  const paths = body.paths
  const alpha = body.confidence

  const useQrng = body.rng === 'qrng'
  const origin = new URL(request.url).origin
  let qrngFallback = false
  let sampling: 'iid' | 'lhs' = 'iid'
  let u1: number[]
  let u2: number[]

  function shuffle<T>(arr: T[], uniforms: number[]) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.max(0, Math.min(0.999999, uniforms[i])) * (i + 1))
      const tmp = arr[i]
      arr[i] = arr[j]
      arr[j] = tmp
    }
  }

  if (useQrng) {
    // Latin hypercube with QRNG-driven jitter and shuffles
    sampling = 'lhs'
    const strata1 = Array.from({ length: paths }, (_, i) => i)
    const strata2 = Array.from({ length: paths }, (_, i) => i)
    try {
      const need = paths * 3
      const u = await getUniformsQrng(need, origin)
      const jitter1 = u.slice(0, paths)
      const jitter2 = u.slice(paths, 2 * paths)
      const shuf = u.slice(2 * paths)
      // use shuf for Fisher-Yates by mapping to indices backwards
      shuffle(strata1, shuf)
      shuffle(strata2, shuf.slice().reverse())
      u1 = new Array(paths)
      u2 = new Array(paths)
      for (let i = 0; i < paths; i++) {
        u1[i] = (strata1[i] + jitter1[i]) / paths
        u2[i] = (strata2[i] + jitter2[i]) / paths
        if (u1[i] <= 0) u1[i] = 1e-12
        if (u1[i] >= 1) u1[i] = 1 - 1e-12
        if (u2[i] <= 0) u2[i] = 1e-12
        if (u2[i] >= 1) u2[i] = 1 - 1e-12
      }
    } catch {
      // fallback to PRNG but keep LHS so method still reduces variance
      qrngFallback = true
      const jitter = getUniformsPrng(paths * 3)
      const jitter1 = jitter.slice(0, paths)
      const jitter2 = jitter.slice(paths, 2 * paths)
      const shuf = jitter.slice(2 * paths)
      const strata1 = Array.from({ length: paths }, (_, i) => i)
      const strata2 = Array.from({ length: paths }, (_, i) => i)
      shuffle(strata1, shuf)
      shuffle(strata2, shuf.slice().reverse())
      u1 = new Array(paths)
      u2 = new Array(paths)
      for (let i = 0; i < paths; i++) {
        u1[i] = (strata1[i] + jitter1[i]) / paths
        u2[i] = (strata2[i] + jitter2[i]) / paths
        if (u1[i] <= 0) u1[i] = 1e-12
        if (u1[i] >= 1) u1[i] = 1 - 1e-12
        if (u2[i] <= 0) u2[i] = 1e-12
        if (u2[i] >= 1) u2[i] = 1 - 1e-12
      }
    }
  } else {
    // i.i.d. PRNG
    const uniforms = getUniformsPrng(paths * 2)
    u1 = new Array(paths)
    u2 = new Array(paths)
    for (let i = 0; i < paths; i++) {
      u1[i] = Math.max(1e-12, uniforms[2 * i])
      u2[i] = uniforms[2 * i + 1]
    }
  }

  // Box-Muller transform to standard normals
  const normals = new Array(paths)
  for (let i = 0; i < paths; i++) {
    const r = Math.sqrt(-2.0 * Math.log(u1[i]))
    const theta = 2 * Math.PI * u2[i]
    normals[i] = r * Math.cos(theta)
  }

  // Loss = -R, where R~N(0,1)
  const losses = normals.map((z) => -z)
  losses.sort((a, b) => a - b)
  const varValue = quantile(losses, alpha)

  // CVaR: average of tail beyond VaR
  const varIndex = Math.floor((losses.length - 1) * alpha)
  const tail = losses.slice(varIndex)
  const cvarValue = tail.reduce((a, b) => a + b, 0) / tail.length

  // Empirical CI via block method (estimator variance depends on sampling scheme)
  const blocks = Math.min(20, Math.max(5, Math.floor(paths / 2000)))
  const blockSize = Math.max(100, Math.floor(paths / blocks))
  const blockVars: number[] = []
  for (let b = 0; b < blocks; b++) {
    const start = b * blockSize
    const end = Math.min(paths, start + blockSize)
    if (end - start < 50) continue
    const block = normals.slice(start, end).map((z) => -z).sort((a, b) => a - b)
    blockVars.push(quantile(block, alpha))
  }
  let se = 0
  if (blockVars.length > 1) {
    const meanBlock = blockVars.reduce((a, b) => a + b, 0) / blockVars.length
    const variance = blockVars.reduce((a, b) => a + (b - meanBlock) * (b - meanBlock), 0) / (blockVars.length - 1)
    se = Math.sqrt(Math.max(0, variance) / blockVars.length)
  }
  const ci: [number, number] = [varValue - 1.96 * se, varValue + 1.96 * se]

  const runtimeMs = performance.now() - t0
  return NextResponse.json({ var: varValue, cvar: cvarValue, ci, runtimeMs, sampleCount: paths, qrngFallback, sampling })
}


