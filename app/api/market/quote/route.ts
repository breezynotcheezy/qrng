import { NextResponse } from 'next/server'

export async function GET(request: Request) {
  const url = new URL(request.url)
  const symbol = url.searchParams.get('symbol')?.toUpperCase()
  if (!symbol) {
    return NextResponse.json({ error: 'Missing symbol' }, { status: 400 })
  }
  try {
    // Yahoo Finance public quote endpoint
    const res = await fetch(`https://query1.finance.yahoo.com/v7/finance/quote?symbols=${encodeURIComponent(symbol)}`, {
      cache: 'no-store',
      headers: { 'accept': 'application/json,text/json' },
    })
    if (res.ok) {
      const json = (await res.json()) as any
      const result = json?.quoteResponse?.result?.[0]
      if (result) {
        const price = result.regularMarketPrice ?? result.postMarketPrice ?? result.preMarketPrice
        return NextResponse.json({ symbol, price, currency: result.currency, name: result.shortName || result.longName })
      }
    }
  } catch {}
  // Fallback: Stooq CSV API
  try {
    const stooq = await fetch(`https://stooq.com/q/l/?s=${encodeURIComponent(symbol)}&f=sd2t2ohlcv&h&e=csv`, {
      cache: 'no-store',
    })
    if (!stooq.ok) throw new Error('stooq error')
    const text = await stooq.text()
    // Parse CSV second line, close price in column 6 (o h l c v)
    const lines = text.trim().split(/\r?\n/)
    if (lines.length >= 2) {
      const cols = lines[1].split(',')
      const close = Number(cols[6])
      if (!Number.isNaN(close)) {
        return NextResponse.json({ symbol, price: close, currency: 'USD', name: symbol })
      }
    }
    return NextResponse.json({ error: 'Symbol not found' }, { status: 404 })
  } catch (e) {
    return NextResponse.json({ error: 'Quote fetch failed', details: String(e) }, { status: 502 })
  }
}


