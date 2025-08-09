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
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <div className="flex items-end gap-2">
        <div className="flex-1">
          <Label>Use Case</Label>
          <Select value={useCase} onValueChange={(v) => onChangeUseCase(v as UseCase)}>
            <SelectTrigger className="w-full"><SelectValue placeholder="Use Case" /></SelectTrigger>
            <SelectContent>
              <SelectGroup>
                <SelectItem value="var">VaR</SelectItem>
                <SelectItem value="option">Option</SelectItem>
              </SelectGroup>
            </SelectContent>
          </Select>
        </div>
        <div className="flex-1">
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
        <div className="self-stretch flex items-end">
          <Button disabled={running} onClick={onRun} className="w-full">{running ? 'Running…' : 'Run'}</Button>
        </div>
      </div>

      {useCase === 'var' ? (
        <div className="grid grid-cols-2 gap-2 sm:col-span-3">
          <div>
            <Label>Confidence</Label>
            <Input type="number" step="0.001" min={0.5} max={0.9999} value={varParams.confidence}
              onChange={(e) => onChangeVar({ ...varParams, confidence: Number(e.target.value) })} />
          </div>
          <div>
            <Label>Paths</Label>
            <Input type="number" step="1000" min={1000} value={varParams.paths}
              onChange={(e) => onChangeVar({ ...varParams, paths: Number(e.target.value) })} />
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-2 sm:col-span-3">
          <div className="col-span-2 flex gap-2">
            <div className="flex-1">
              <Label>Ticker</Label>
              <Input value={symbol} onChange={(e) => setSymbol(e.target.value.toUpperCase())} />
            </div>
            <div className="self-end">
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
        </div>
      )}
    </div>
  )
}


