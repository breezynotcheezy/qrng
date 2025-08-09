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

  let uniforms: number[]
  let qrngFallback = false
  if (useQrng) {
    try {
      uniforms = await getUniformsQrng(paths * 2, origin)
    } catch {
      uniforms = getUniformsPrng(paths * 2)
      qrngFallback = true
    }
  } else {
    uniforms = getUniformsPrng(paths * 2)
  }

  // Box-Muller
  const payoffs = new Float64Array(paths)
  const drift = (r - 0.5 * sigma * sigma) * T
  const vol = sigma * Math.sqrt(T)
  for (let i = 0; i < paths; i++) {
    const u1 = Math.max(1e-12, uniforms[2 * i])
    const u2 = uniforms[2 * i + 1]
    const R = Math.sqrt(-2.0 * Math.log(u1))
    const Z = R * Math.cos(2 * Math.PI * u2)
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
  return NextResponse.json({ mean: price, stderr, ci, runtimeMs, sampleCount: paths, qrngFallback })
}

