import React, {useState, useRef, useEffect} from 'react'
import {Play, UploadCloud, AlertCircle, CheckCircle, Settings, Github, Twitter} from 'lucide-react'
import {AreaChart, Area, LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer} from 'recharts'

const mockSpectrogram = Array.from({length: 32}).map((_, i)=>({x: i, v: Math.abs(Math.sin(i/4))* (Math.random()*0.8+0.3)}))
const mockWaveform = Array.from({length: 64}).map((_, i)=>({x: i, y: Math.sin(i/4) * (Math.random()*0.7+0.3)}))

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

function Header(){
  return (
    <header className="py-6 px-6 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-[#3b82f6] to-[#a78bfa] flex items-center justify-center shadow-lg">
          <svg className="w-7 h-7 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M12 3v18" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
        </div>
        <div>
          <h1 className="text-lg font-semibold">Tamil Deepfake Detector</h1>
          <p className="text-sm text-slate-400">AI-powered detection of synthetic Tamil speech</p>
        </div>
      </div>
      <nav className="flex items-center gap-4">
        <a className="text-slate-300 hover:text-white" href="#about">About</a>
        <a className="text-slate-300 hover:text-white" href="#model">Model</a>
        <div className="flex gap-3">
          <a aria-label="github" className="p-2 glass rounded-md" href="#"><Github size={16} /></a>
          <a aria-label="twitter" className="p-2 glass rounded-md" href="#"><Twitter size={16} /></a>
        </div>
      </nav>
    </header>
  )
}

function Hero(){
  return (
    <section className="px-6 py-16">
      <div className="max-w-6xl mx-auto grid md:grid-cols-2 gap-8 items-center">
        <div>
          <h2 className="text-4xl md:text-5xl font-extrabold leading-tight gradient-anim">🎙️ Tamil Deepfake Audio Detector</h2>
          <p className="mt-4 text-slate-300 max-w-xl">AI-Powered Detection of Synthetic Tamil Speech — fast, accurate, and designed for production.</p>
          <div className="mt-6 flex gap-4">
            <a href="#upload" className="px-5 py-3 bg-gradient-to-r from-[#60a5fa] to-[#a78bfa] rounded-lg text-black font-semibold shadow hover:scale-105 transition">Get started</a>
            <a href="#about" className="px-5 py-3 border border-slate-700 rounded-lg text-slate-200 glass">How it works</a>
          </div>
        </div>
        <div className="relative">
            <div className="h-56 glass rounded-xl p-6 glow">
            <div className="h-full flex flex-col justify-center items-center text-center">
              <div className="text-slate-300">Upload a Tamil audio file to analyze</div>
              <div className="mt-4 text-sm text-slate-400">Supports WAV, MP3, FLAC • Real-time API detection</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

function FileUploader({onFile}){
  const [hover, setHover] = useState(false)
  const inputRef = useRef()

  function handleFiles(files){
    const f = files?.[0]
    if(f) onFile(f)
  }

  return (
    <div id="upload" className="w-full">
      <div
        onDragOver={(e)=>{e.preventDefault(); setHover(true)}}
        onDragLeave={()=>setHover(false)}
        onDrop={(e)=>{e.preventDefault(); setHover(false); handleFiles(e.dataTransfer.files)}}
        className={`w-full p-8 rounded-xl glass border border-slate-700 flex flex-col items-center justify-center text-center transition ${hover? 'scale-102 shadow-lg':''}`}
      >
        <UploadCloud className="text-slate-200" />
        <div className="mt-3 text-slate-200 font-medium">Drag & drop audio file</div>
        <div className="mt-2 text-sm text-slate-400">WAV, MP3, FLAC • Up to 30MB</div>
        <div className="mt-4">
          <button onClick={()=>inputRef.current.click()} className="px-4 py-2 bg-[#3b82f6] rounded-md font-medium text-black">Browse files</button>
        </div>
        <input ref={inputRef} type="file" accept="audio/*" className="hidden" onChange={(e)=>handleFiles(e.target.files)} aria-label="Upload audio file" />
      </div>
    </div>
  )
}

function AudioPlayer({file}){
  if(!file) return null
  const url = URL.createObjectURL(file)
  return (
    <div className="mt-4 glass p-3 rounded-md flex items-center gap-4">
      <audio controls src={url} className="w-full"/>
      <div className="text-sm text-slate-300">{file.name}</div>
    </div>
  )
}

function CircularConfidence({value, fake}){
  const radius = 48
  const stroke = 10
  const normalized = Math.max(0, Math.min(100, Math.round(value)))
  const c = 2*Math.PI*radius
  const dash = c * (normalized/100)
  return (
    <div className="w-40 h-40 flex items-center justify-center">
      <svg width="120" height="120" viewBox="0 0 120 120">
        <defs>
          <linearGradient id="g1" x1="0" x2="1">
            <stop offset="0%" stopColor="#60a5fa" />
            <stop offset="100%" stopColor={fake? '#ef4444' : '#22c55e'} />
          </linearGradient>
        </defs>
        <g transform="translate(60,60)">
          <circle r={radius} stroke="#0f172a" strokeWidth={stroke} fill="none" />
          <circle r={radius} stroke="url(#g1)" strokeWidth={stroke} strokeLinecap="round" fill="none" strokeDasharray={`${dash} ${c-dash}`} transform="rotate(-90)" style={{transition:'stroke-dasharray 900ms ease'}} />
          <text x="0" y="6" fill="#e6eef8" fontSize="18" fontWeight="700" textAnchor="middle">{normalized}%</text>
        </g>
      </svg>
    </div>
  )
}

function ResultCard({result, confidence, processingTime}){
  if(!result) return null
  const fake = result === 'FAKE'
  return (
    <div className="mt-6 p-6 rounded-xl glass flex gap-6 items-center">
      <div>
        <div className={`px-4 py-2 rounded-full text-sm font-semibold ${fake? 'bg-red-600/20 text-red-300':'bg-green-600/20 text-green-300'}`}>{result} {fake? '⚠️':'✅'}</div>
        <div className="mt-4 text-slate-300">Confidence</div>
      </div>
      <CircularConfidence value={confidence} fake={fake} />
      <div className="flex-1">
        <div className="text-slate-400">Processing time: <span className="text-slate-200">{processingTime.toFixed(2)}s</span></div>
        <div className={`mt-4 p-4 rounded-md ${fake? 'bg-red-900/20':'bg-green-900/10'}`}>
          <div className="text-sm text-slate-300">{fake? 'This audio appears to be AI-generated (synthetic Tamil speech).' : 'This audio appears to be authentic human speech.'} Model confidence: {confidence}%</div>
        </div>
      </div>
    </div>
  )
}

function SpectrogramChart(){
  return (
    <div className="h-48 glass rounded-xl p-4">
      <div className="text-sm text-slate-300 mb-2">Mel Spectrogram (mock)</div>
      <ResponsiveContainer width="100%" height={120}>
        <AreaChart data={mockSpectrogram}>
          <defs>
            <linearGradient id="colorV" x1="0" x2="0" y1="0" y2="1">
              <stop offset="5%" stopColor="#a78bfa" stopOpacity={0.9}/>
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1}/>
            </linearGradient>
          </defs>
          <Area type="monotone" dataKey="v" stroke="#60a5fa" fillOpacity={1} fill="url(#colorV)" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

function WaveformDisplay(){
  return (
    <div className="h-36 glass rounded-xl p-4 mt-4">
      <div className="text-sm text-slate-300 mb-2">Waveform (mock)</div>
      <ResponsiveContainer width="100%" height={120}>
        <LineChart data={mockWaveform}>
          <Line stroke="#ec4899" dataKey="y" dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

function StatsCards({file, audioInfo}){
  const duration = audioInfo?.duration || '—'
  const sampleRate = audioInfo?.sample_rate || '—'
  const size = audioInfo?.file_size ? audioInfo.file_size + ' KB' : (file? (Math.max(10, Math.round(file.size/1024)) + ' KB') : '—')
  return (
    <div className="grid grid-cols-3 gap-4 mt-4">
      <div className="glass p-3 rounded-md text-center">
        <div className="text-xs text-slate-400">Duration</div>
        <div className="text-lg font-semibold">{duration}{duration !== '—' && 's'}</div>
      </div>
      <div className="glass p-3 rounded-md text-center">
        <div className="text-xs text-slate-400">Sample Rate</div>
        <div className="text-lg font-semibold">{sampleRate !== '—' ? sampleRate + 'Hz' : sampleRate}</div>
      </div>
      <div className="glass p-3 rounded-md text-center">
        <div className="text-xs text-slate-400">File Size</div>
        <div className="text-lg font-semibold">{size}</div>
      </div>
    </div>
  )
}

function ModelInfoCard(){
  const [acc, setAcc] = useState(0)
  useEffect(()=>{
    let i=0
    const t = setInterval(()=>{
      i+=1
      setAcc(prev=> Math.min(94.5, prev + (94.5/20)))
      if(i>20) clearInterval(t)
    },80)
    return ()=>clearInterval(t)
  },[])
  return (
    <aside id="model" className="mt-6 glass p-4 rounded-xl">
      <h3 className="font-semibold">Model Info</h3>
      <div className="mt-3 text-sm text-slate-300">Architecture: CNN-based audio classifier</div>
      <div className="mt-3 grid grid-cols-2 gap-2">
        <div className="p-3 glass rounded-md text-center">
          <div className="text-xs text-slate-400">Accuracy</div>
          <div className="text-lg font-semibold">{acc.toFixed(1)}%</div>
        </div>
        <div className="p-3 glass rounded-md text-center">
          <div className="text-xs text-slate-400">Precision</div>
          <div className="text-lg font-semibold">0.93</div>
        </div>
        <div className="p-3 glass rounded-md text-center">
          <div className="text-xs text-slate-400">Recall</div>
          <div className="text-lg font-semibold">0.95</div>
        </div>
        <div className="p-3 glass rounded-md text-center">
          <div className="text-xs text-slate-400">F1-score</div>
          <div className="text-lg font-semibold">0.94</div>
        </div>
      </div>
      <div className="mt-4 text-sm text-slate-400">Trained on 6GB Tamil audio data • Demo mock metrics</div>
    </aside>
  )
}

export default function App(){
  const [file, setFile] = useState(null)
  const [processing, setProcessing] = useState(false)
  const [result, setResult] = useState(null)
  const [confidence, setConfidence] = useState(0)
  const [error, setError] = useState(null)
  const [audioInfo, setAudioInfo] = useState(null)
  const [processingTime, setProcessingTime] = useState(0)

  function handleFile(f){
    setFile(f)
    setResult(null)
    setConfidence(0)
    setError(null)
    setAudioInfo(null)
  }

  async function analyze(){
    if(!file) return
    setProcessing(true)
    setResult(null)
    setConfidence(0)
    setError(null)
    
    try {
      const formData = new FormData()
      formData.append('file', file)
      
      const startTime = performance.now()
      const response = await fetch(`${API_BASE_URL}/api/predict`, {
        method: 'POST',
        body: formData,
        headers: {
          'Accept': 'application/json'
        }
      })
      const endTime = performance.now()
      
      if (!response.ok) {
        const errData = await response.json()
        throw new Error(errData.error || `API error: ${response.status}`)
      }
      
      const data = await response.json()
      
      if (data.success) {
        setResult(data.prediction)
        setConfidence(data.confidence)
        setAudioInfo(data.audio_info)
        setProcessingTime(data.processing_time_seconds || (endTime - startTime) / 1000)
      } else {
        setError(data.error || 'Prediction failed')
      }
    } catch (err) {
      console.error('API error:', err)
      setError(err.message || 'Failed to connect to API. Make sure the backend is running on port 5000.')
    } finally {
      setProcessing(false)
    }
  }

  return (
    <div className="min-h-screen font-sans bg-gradient-to-b from-[#0a0e1a] via-[#0f172a] to-[#1e293b] text-slate-100">
      <div className="max-w-7xl mx-auto">
        <Header />
        <Hero />

        <main className="px-6 pb-16">
          <div className="grid md:grid-cols-3 gap-6">
            <div className="md:col-span-2">
              <div className="glass p-6 rounded-xl">
                <h4 className="font-semibold">Upload & Analyze</h4>
                <p className="text-sm text-slate-400">Drop a Tamil audio file to start detection</p>
                <div className="mt-4">
                  <FileUploader onFile={handleFile} />
                  <AudioPlayer file={file} />
                  
                  {error && (
                    <div className="mt-4 p-4 rounded-md bg-red-900/20 border border-red-700/50 flex gap-3">
                      <AlertCircle size={20} className="text-red-400 flex-shrink-0 mt-0.5" />
                      <div>
                        <div className="font-semibold text-red-300">Error</div>
                        <div className="text-sm text-red-200">{error}</div>
                      </div>
                    </div>
                  )}
                  
                  <div className="mt-4 flex items-center gap-3">
                    <button onClick={analyze} disabled={!file || processing} className="px-4 py-3 bg-gradient-to-r from-[#60a5fa] to-[#a78bfa] rounded-md text-black font-semibold disabled:opacity-50 hover:scale-105 transition">
                      {processing? 'Analyzing...' : 'Analyze'}
                    </button>
                    {processing && <div className="text-slate-400">Processing <span className="ml-2 inline-block animate-pulse">●</span></div>}
                  </div>
                  
                  <ResultCard result={result} confidence={confidence} processingTime={processingTime} />
                  
                  {result && (
                    <>
                      <SpectrogramChart />
                      <WaveformDisplay />
                      <StatsCards file={file} audioInfo={audioInfo} />
                    </>
                  )}
                </div>
              </div>
            </div>
            <aside>
              <ModelInfoCard />
              <div className="mt-6 glass p-4 rounded-xl">
                <h4 className="font-semibold">Sample Predictions</h4>
                <div className="mt-3 text-sm text-slate-300">87.3% REAL • 92.1% FAKE</div>
                <div className="mt-4">
                  <button className="w-full p-3 rounded-md bg-[#111827] border border-slate-700 text-sm">View dataset</button>
                </div>
              </div>
            </aside>
          </div>

          <section id="about" className="mt-10 glass p-6 rounded-xl">
            <h3 className="font-semibold">How it works</h3>
            <ol className="mt-3 list-decimal list-inside text-slate-300">
              <li>Upload audio</li>
              <li>Extract features (mel spectrograms, MFCCs)</li>
              <li>Run CNN classifier to detect synthetic patterns</li>
              <li>Show results, visualizations, and confidence score</li>
            </ol>
            <div className="mt-4 text-sm text-slate-400">Tech: React + Flask + PyTorch • Backend runs at localhost:5000</div>
          </section>
        </main>

        <footer className="p-6 text-center text-slate-400">
          <div>© {new Date().getFullYear()} Tamil Deepfake Detector • Built with ❤️</div>
        </footer>
      </div>
    </div>
  )
}
