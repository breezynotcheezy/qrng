"use client"

import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'

export type UseCase = 'var' | 'option'
export type RngChoice = 'qrng' | 'prng'

export type VarParams = { confidence: number; paths: number }
export type OptionParams = { symbol?: string; S0: number; K: number; r: number; sigma: number; T: number; paths: number }

export function ControlsPanel(props: {
  useCase: UseCase
  rng: RngChoice
  varParams: VarParams
  optionParams: OptionParams
  onChangeUseCase: (uc: UseCase) => void
  onChangeRng: (rng: RngChoice) => void
  onChangeVar: (p: VarParams) => void
  onChangeOption: (p: OptionParams) => void
  onRun: () => void
  running: boolean
}) {
  const { useCase, rng, varParams, optionParams, onChangeUseCase, onChangeRng, onChangeVar, onChangeOption, onRun, running } = props
  const [symbol, setSymbol] = useState(optionParams.symbol ?? 'AAPL')

  useEffect(() => {
    setSymbol(optionParams.symbol ?? 'AAPL')
  }, [optionParams.symbol])

  async function fetchQuote() {
    try {
      const res = await fetch(`/api/market/quote?symbol=${encodeURIComponent(symbol)}`)
      if (!res.ok) return
      const json = await res.json()
      if (typeof json.price === 'number') {
        onChangeOption({ ...optionParams, symbol: json.symbol, S0: json.price })
      }
    } catch {}
  }

  return (
    <div className="w-full rounded-lg border p-3 sm:p-4 bg-background/60">
      <div className="grid gap-3 sm:grid-cols-4">
        <div className="sm:col-span-4 grid gap-2 sm:grid-cols-3">
          <div>
            <Label>Ticker</Label>
            <div className="flex gap-2">
              <Input className="flex-1" value={symbol} onChange={(e) => setSymbol(e.target.value.toUpperCase())} />
              <Button variant="secondary" onClick={fetchQuote}>Get Price</Button>
            </div>
          </div>
          <div>
            <Label>S0 (spot)</Label>
            <Input type="number" step="0.01" value={optionParams.S0}
              onChange={(e) => onChangeOption({ ...optionParams, S0: Number(e.target.value) })} />
          </div>
          <div>
            <Label>K (strike)</Label>
            <Input type="number" step="0.01" value={optionParams.K}
              onChange={(e) => onChangeOption({ ...optionParams, K: Number(e.target.value) })} />
          </div>
        </div>
        <div>
          <Label>r (rate)</Label>
          <Input type="number" step="0.0001" value={optionParams.r}
            onChange={(e) => onChangeOption({ ...optionParams, r: Number(e.target.value) })} />
        </div>
        <div>
          <Label>sigma (vol)</Label>
          <Input type="number" step="0.0001" value={optionParams.sigma}
            onChange={(e) => onChangeOption({ ...optionParams, sigma: Number(e.target.value) })} />
        </div>
        <div>
          <Label>T (years)</Label>
          <Input type="number" step="0.01" value={optionParams.T}
            onChange={(e) => onChangeOption({ ...optionParams, T: Number(e.target.value) })} />
        </div>
        <div>
          <Label>Paths</Label>
          <Input type="number" step="1000" min={1000} value={optionParams.paths}
            onChange={(e) => onChangeOption({ ...optionParams, paths: Number(e.target.value) })} />
        </div>
        <div>
          <Label>RNG</Label>
          <Select value={rng} onValueChange={(v) => onChangeRng(v as RngChoice)}>
            <SelectTrigger className="w-full"><SelectValue placeholder="RNG" /></SelectTrigger>
            <SelectContent>
              <SelectGroup>
                <SelectItem value="qrng">Quantum</SelectItem>
                <SelectItem value="prng">Classical</SelectItem>
              </SelectGroup>
            </SelectContent>
          </Select>
        </div>
        <div className="sm:col-span-4">
          <Button disabled={running} onClick={onRun} className="w-full">
            {running ? 'Running…' : 'Run Pricing'}
          </Button>
        </div>
      </div>
    </div>
  )
}


