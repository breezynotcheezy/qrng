import { NextResponse } from 'next/server'
import { z } from 'zod'

const OptionRequestSchema = z.object({
  S0: z.number().positive(),
  K: z.number().positive(),
  r: z.number(),
  sigma: z.number().positive(),
  T: z.number().positive(),
  paths: z.number().int().positive().max(1_000_000).default(100_000),
  rng: z.enum(['qrng', 'prng']).default('prng'),
})

type OptionRequest = z.infer<typeof OptionRequestSchema>

function getUniformsPrng(n: number): number[] {
  const out = new Float64Array(n)
  if (typeof crypto !== 'undefined' && 'getRandomValues' in crypto) {
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
  const chunkSize = 2048
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
  let body: OptionRequest
  try {
    body = OptionRequestSchema.parse(await request.json())
  } catch (err) {
    return NextResponse.json({ error: 'Invalid request', details: String(err) }, { status: 400 })
  }

  const { S0, K, r, sigma, T, paths } = body
  const useQrng = body.rng === 'qrng'
  const origin = new URL(request.url).origin

  const t0 = performance.now()

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
    sampling = 'lhs'
    const strata1 = Array.from({ length: paths }, (_, i) => i)
    const strata2 = Array.from({ length: paths }, (_, i) => i)
    try {
      const need = paths * 3
      const u = await getUniformsQrng(need, origin)
      const jitter1 = u.slice(0, paths)
      const jitter2 = u.slice(paths, 2 * paths)
      const shuf = u.slice(2 * paths)
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
      qrngFallback = true
      const jitter = getUniformsPrng(paths * 3)
      const jitter1 = jitter.slice(0, paths)
      const jitter2 = jitter.slice(paths, 2 * paths)
      const shuf = jitter.slice(2 * paths)
      const strata1b = Array.from({ length: paths }, (_, i) => i)
      const strata2b = Array.from({ length: paths }, (_, i) => i)
      shuffle(strata1b, shuf)
      shuffle(strata2b, shuf.slice().reverse())
      u1 = new Array(paths)
      u2 = new Array(paths)
      for (let i = 0; i < paths; i++) {
        u1[i] = (strata1b[i] + jitter1[i]) / paths
        u2[i] = (strata2b[i] + jitter2[i]) / paths
        if (u1[i] <= 0) u1[i] = 1e-12
        if (u1[i] >= 1) u1[i] = 1 - 1e-12
        if (u2[i] <= 0) u2[i] = 1e-12
        if (u2[i] >= 1) u2[i] = 1 - 1e-12
      }
    }
  } else {
    const uniforms = getUniformsPrng(paths * 2)
    u1 = new Array(paths)
    u2 = new Array(paths)
    for (let i = 0; i < paths; i++) {
      u1[i] = Math.max(1e-12, uniforms[2 * i])
      u2[i] = uniforms[2 * i + 1]
    }
  }

  // Box-Muller
  const payoffs = new Float64Array(paths)
  const drift = (r - 0.5 * sigma * sigma) * T
  const vol = sigma * Math.sqrt(T)
  for (let i = 0; i < paths; i++) {
    const R = Math.sqrt(-2.0 * Math.log(u1[i]))
    const Z = R * Math.cos(2 * Math.PI * u2[i])
    const ST = S0 * Math.exp(drift + vol * Z)
    payoffs[i] = Math.max(0, ST - K)
  }

  // Monte Carlo estimate
  let sum = 0
  let sumSq = 0
  for (let i = 0; i < paths; i++) {
    const p = payoffs[i]
    sum += p
    sumSq += p * p
  }
  const mean = sum / paths
  const variance = Math.max(0, sumSq / paths - mean * mean)
  const disc = Math.exp(-r * T)
  const price = disc * mean
  const stderr = disc * Math.sqrt(variance / paths)
  const ci: [number, number] = [price - 1.96 * stderr, price + 1.96 * stderr]

  const runtimeMs = performance.now() - t0
  return NextResponse.json({ mean: price, stderr, ci, runtimeMs, sampleCount: paths, qrngFallback, sampling })
}

