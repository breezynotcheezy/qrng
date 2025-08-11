import { NextResponse } from 'next/server'
import { z } from 'zod'

export const dynamic = 'force-dynamic'

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

type QrngFiResponse = {
  success: boolean
  data?: number[]
  length?: number
  size?: number
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

// Multiple QRNG services for redundancy
async function fetchAnuQrng(len: number): Promise<number[] | null> {
  // ANU QRNG API - correct endpoint and parameters
  const url = new URL('https://qrng.anu.edu.au/API/jsonI.php')
  url.searchParams.set('length', String(len))
  url.searchParams.set('type', 'uint16')
  url.searchParams.set('size', '1') // Required parameter
  
  try {
    const controller = new AbortController()
    const t = setTimeout(() => controller.abort(), 8000)
    const res = await fetch(url.toString(), { 
      cache: 'no-store', 
      signal: controller.signal,
      headers: { 
        'User-Agent': 'QRNG-Toolkit/1.0',
        'Accept': 'application/json'
      }
    })
    clearTimeout(t)
    
    if (!res.ok) {
      // Handle rate limiting gracefully
      if (res.status === 429 || res.statusText.includes('rate')) {
        return null // Will fall back to Random.org
      }
      return null
    }
    
    const upstream = (await res.json()) as AnuResponse
    if (upstream.success && Array.isArray(upstream.data)) {
      return upstream.data
    }
  } catch {
    // Silent fallback
  }
  return null
}

async function fetchRandomOrg(len: number): Promise<number[] | null> {
  // Random.org API as backup - more reliable and no rate limiting
  const url = new URL('https://www.random.org/integers/')
  url.searchParams.set('num', String(len))
  url.searchParams.set('min', '0')
  url.searchParams.set('max', '65535')
  url.searchParams.set('col', '1')
  url.searchParams.set('base', '10')
  url.searchParams.set('format', 'plain')
  url.searchParams.set('rnd', 'new')
  
  try {
    const controller = new AbortController()
    const t = setTimeout(() => controller.abort(), 8000)
    const res = await fetch(url.toString(), { 
      cache: 'no-store', 
      signal: controller.signal,
      headers: { 
        'User-Agent': 'QRNG-Toolkit/1.0'
      }
    })
    clearTimeout(t)
    
    if (!res.ok) return null
    
    const text = await res.text()
    const numbers = text.trim().split('\n').map(n => parseInt(n, 10))
    if (numbers.length === len && numbers.every(n => !isNaN(n))) {
      return numbers
    }
  } catch {
    // Silent fallback
  }
  return null
}

async function fetchUpstreamChunk(len: number): Promise<number[]> {
  // Try multiple QRNG services with proper retry logic
  // Start with Random.org as it's more reliable, then try ANU
  const services = [
    () => fetchRandomOrg(len), // More reliable, no rate limiting
    () => fetchAnuQrng(len),   // True quantum, but rate limited
  ]
  
  for (let attempt = 0; attempt < 2; attempt++) {
    for (const service of services) {
      try {
        const result = await service()
        if (result) return result
      } catch {
        // Continue to next service
      }
    }
    
    // Short backoff before retry
    if (attempt < 1) {
      await new Promise((r) => setTimeout(r, 200))
    }
  }
  
  throw new Error('All QRNG services unavailable')
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

  // Accumulate in chunks
  let ints: number[] = []
  let usedQrng = false
  
  try {
    let remaining = count
    while (remaining > 0) {
      const take = Math.min(MAX_PER_CALL, remaining)
      const chunk = await fetchUpstreamChunk(take)
      ints.push(...chunk)
      remaining -= take
      usedQrng = true
      // minimal pacing to avoid rate limiting
      if (remaining > 0) await new Promise((r) => setTimeout(r, 25))
    }
  } catch {
    // Silent fallback to crypto
    const fallback = new Uint16Array(count)
    crypto.getRandomValues(fallback)
    ints = Array.from(fallback)
  }

  const payload: { values: number[]; dtype: RngRequest['dtype']; normalize: boolean; entropy?: { mean: number }; source?: string } = {
    values: ints,
    dtype,
    normalize,
    source: usedQrng ? 'qrng' : 'crypto'
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
      source: usedQrng ? 'qrng' : 'crypto'
    }
    const enriched = { ...payload, reproId: fnv1a32(cacheKey) }
    rngCache.set(cacheKey, { at: now, payload })
    return NextResponse.json(enriched)
  }

  const enriched = { ...payload, reproId: fnv1a32(cacheKey) }
  rngCache.set(cacheKey, { at: now, payload })
  return NextResponse.json(enriched)
}


