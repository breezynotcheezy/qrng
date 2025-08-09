import { NextResponse } from 'next/server'
import { z } from 'zod'

// Request schema
const RngRequestSchema = z.object({
  count: z.number().int().positive().max(100_000),
  dtype: z.enum(['uint8', 'uint16']),
  normalize: z.boolean().optional().default(false),
})

type RngRequest = z.infer<typeof RngRequestSchema>

type AnuResponse = {
  success: boolean
  data?: number[]
  error?: string
}

// Simple in-memory cache (per server instance) with short TTL
const rngCache = new Map<string, { at: number; payload: any }>()
const RNG_TTL_MS = 30_000

function stableStringify(obj: unknown): string {
  return JSON.stringify(obj, Object.keys(obj as any).sort())
}

function fnv1a32(str: string): string {
  let h = 0x811c9dc5
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i)
    h += (h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24)
  }
  // unsigned and hex
  return (h >>> 0).toString(16)
}

function normalizeToUnitInterval(values: number[], maxValue: number): number[] {
  const denom = maxValue + 1 // to map [0, max] -> [0,1)
  return values.map((v) => v / denom)
}

function computeMean(values: number[]): number {
  if (values.length === 0) return 0
  return values.reduce((a, b) => a + b, 0) / values.length
}

export async function POST(request: Request) {
  let body: RngRequest
  try {
    const json = await request.json()
    body = RngRequestSchema.parse(json)
  } catch (err) {
    return NextResponse.json(
      { error: 'Invalid request', details: err instanceof Error ? err.message : String(err) },
      { status: 400 },
    )
  }

  const { count, dtype, normalize } = body

  // Upstream API docs: https://qrng.anu.edu.au/contact/api-documentation/
  const typeParam = dtype // 'uint8' | 'uint16'

  // Some upstreams limit max length per call; be robust by chunking here so callers
  // can request any count once without managing chunks themselves.
  const MAX_PER_CALL = 1024

  const cacheKey = stableStringify({ count, dtype, normalize })
  const now = Date.now()
  const cached = rngCache.get(cacheKey)
  if (cached && now - cached.at < RNG_TTL_MS) {
    const enriched = { ...cached.payload, reproId: fnv1a32(cacheKey), cached: true }
    return NextResponse.json(enriched)
  }

  async function fetchUpstreamChunk(len: number, attemptBaseDelay = 150): Promise<number[]> {
    const url = new URL('https://qrng.anu.edu.au/API/jsonI.php')
    url.searchParams.set('length', String(len))
    url.searchParams.set('type', typeParam)
    let lastErr: any = null
    for (let attempt = 0; attempt < 6; attempt++) {
      try {
        const controller = new AbortController()
        const t = setTimeout(() => controller.abort(), 6000)
        const res = await fetch(url.toString(), { cache: 'no-store', signal: controller.signal })
        clearTimeout(t)
        if (!res.ok) {
          lastErr = new Error(`Status ${res.status}`)
        } else {
          const upstream = (await res.json()) as AnuResponse
          if (upstream.success && Array.isArray(upstream.data)) {
            return upstream.data
          }
          lastErr = new Error('Malformed upstream payload')
        }
      } catch (e) {
        lastErr = e
      }
      // backoff
      await new Promise((r) => setTimeout(r, attemptBaseDelay * (attempt + 1)))
    }
    throw lastErr || new Error('QRNG upstream failed')
  }

  // Accumulate in chunks
  let ints: number[] = []
  try {
    let remaining = count
    while (remaining > 0) {
      const take = Math.min(MAX_PER_CALL, remaining)
      const chunk = await fetchUpstreamChunk(take)
      ints.push(...chunk)
      remaining -= take
      // small pacing to avoid rate limiting
      if (remaining > 0) await new Promise((r) => setTimeout(r, 50))
    }
  } catch (err) {
    return NextResponse.json(
      { error: 'Failed to reach upstream QRNG service', details: err instanceof Error ? err.message : String(err) },
      { status: 502 },
    )
  }

  const payload: { values: number[]; dtype: RngRequest['dtype']; normalize: boolean; entropy?: { mean: number } } = {
    values: ints,
    dtype,
    normalize,
  }

  if (normalize) {
    const maxVal = dtype === 'uint8' ? 255 : 65535
    const uniforms = normalizeToUnitInterval(ints, maxVal)
    const mean = computeMean(uniforms)
    const payload = {
      values: uniforms,
      dtype: 'float',
      normalize: true,
      entropy: { mean }, // simple sanity check: mean should be ~0.5
    }
    const enriched = { ...payload, reproId: fnv1a32(cacheKey) }
    rngCache.set(cacheKey, { at: now, payload })
    return NextResponse.json(enriched)
  }

  const enriched = { ...payload, reproId: fnv1a32(cacheKey) }
  rngCache.set(cacheKey, { at: now, payload })
  return NextResponse.json(enriched)
}


