import React, { useState, useEffect } from 'react';
import { Shield, Brain, AlertTriangle, TrendingUp, Zap, Info, RefreshCw, BarChart3 } from 'lucide-react';

const App = () => {
  // --- ÉTATS DE SIMULATION ---
  const [aiScores, setAiScores] = useState([85, 88, 82]);
  const [drawdown, setDrawdown] = useState(0);
  const [consecutiveLosses, setConsecutiveLosses] = useState(0);
  const [marketVolatility, setMarketVolatility] = useState(2.0); // ATR %
  
  // --- RÉSULTATS DE LA LOGIQUE TITAN v4.5 ---
  const [decision, setDecision] = useState({ 
    action: 'WAIT', 
    reason: '', 
    sl: 0, 
    tp: 0, 
    riskScaling: 1.0,
    dispersion: 0 
  });

  const calculateLogic = () => {
    // 1. Calcul de la Dispersion (Standard Deviation simple)
    const avg = aiScores.reduce((a, b) => a + b) / 3;
    const squareDiffs = aiScores.map(s => Math.pow(s - avg, 2));
    const dispersion = Math.sqrt(squareDiffs.reduce((a, b) => a + b) / 3);

    // 2. Risk Scaling
    let riskScaling = drawdown <= -0.05 ? 0.5 : 1.0;

    // 3. Cooldown logic
    let isCooldown = consecutiveLosses >= 3;

    // 4. Adaptive SL/TP (v4.5 Formula)
    const convictionFactor = (avg - 80) / 20;
    let tp = 0.06 + (convictionFactor * 0.04);
    let sl = 0.03 - (convictionFactor * 0.01);
    
    // Clamping (Audit v4.4)
    tp = Math.min(Math.max(tp, 0.04), 0.12);
    sl = Math.min(Math.max(sl, 0.015), 0.05);

    // 5. Final Decision
    let action = 'GO';
    let reason = 'Signal validé par le consensus';

    if (isCooldown) {
      action = 'COOLDOWN';
      reason = 'Pause forcée (3+ pertes consécutives)';
    } else if (dispersion > 25) {
      action = 'REJECTED';
      reason = 'Dispersion IA trop élevée (Désaccord)';
    } else if (avg < 80) {
      action = 'REJECTED';
      reason = 'Consensus insuffisant (< 80)';
    } else if (drawdown <= -0.10) {
      action = 'KILL_SWITCH';
      reason = 'Drawdown critique atteint (-10%)';
    }

    setDecision({ action, reason, sl, tp, riskScaling, dispersion, avg });
  };

  useEffect(() => {
    calculateLogic();
  }, [aiScores, drawdown, consecutiveLosses]);

  const updateScore = (index, val) => {
    const newScores = [...aiScores];
    newScores[index] = parseInt(val);
    setAiScores(newScores);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 md:p-8 font-sans">
      <div className="max-w-6xl mx-auto">
        
        {/* HEADER */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
          <div>
            <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400">
              Titan v4.5 Sentinel-Elite
            </h1>
            <p className="text-slate-400 mt-1">Visualiseur de Logique et Réactions IA</p>
          </div>
          <div className="flex items-center gap-3 bg-slate-900 border border-slate-800 p-2 rounded-lg">
            <div className={`h-3 w-3 rounded-full ${decision.action === 'GO' ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`}></div>
            <span className="text-sm font-mono uppercase tracking-wider">Status: {decision.action}</span>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* INPUT PANEL */}
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6 shadow-xl backdrop-blur-sm">
              <h2 className="flex items-center gap-2 text-lg font-semibold mb-6">
                <Brain className="text-blue-400 w-5 h-5" /> Colisée des IA
              </h2>
              
              {aiScores.map((score, i) => (
                <div key={i} className="mb-6">
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-slate-400">Modèle {i+1} {i === 0 ? "(Claude)" : i === 1 ? "(GPT)" : "(Gemini)"}</span>
                    <span className="font-mono text-blue-400">{score}/100</span>
                  </div>
                  <input 
                    type="range" min="0" max="100" value={score} 
                    onChange={(e) => updateScore(i, e.target.value)}
                    className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                  />
                </div>
              ))}

              <div className="pt-4 border-t border-slate-800">
                <h2 className="flex items-center gap-2 text-lg font-semibold mb-6">
                  <Shield className="text-emerald-400 w-5 h-5" /> Gouvernance
                </h2>
                
                <div className="mb-6">
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-slate-400">Drawdown Portefeuille</span>
                    <span className={`font-mono ${drawdown < 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                      {(drawdown * 100).toFixed(1)}%
                    </span>
                  </div>
                  <input 
                    type="range" min="-0.15" max="0.05" step="0.005" value={drawdown} 
                    onChange={(e) => setDrawdown(parseFloat(e.target.value))}
                    className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                  />
                </div>

                <div className="flex items-center justify-between bg-slate-800/50 p-3 rounded-lg border border-slate-700">
                  <span className="text-sm text-slate-300">Pertes Consécutives</span>
                  <div className="flex gap-2">
                    {[1, 2, 3].map(n => (
                      <button 
                        key={n}
                        onClick={() => setConsecutiveLosses(n === consecutiveLosses ? n-1 : n)}
                        className={`w-8 h-8 rounded flex items-center justify-center transition-all ${consecutiveLosses >= n ? 'bg-red-500 text-white' : 'bg-slate-700 text-slate-500'}`}
                      >
                        {n}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* REACTION DISPLAY */}
          <div className="lg:col-span-2 space-y-6">
            
            {/* DECISION CARD */}
            <div className={`p-8 rounded-xl border-2 transition-all ${
              decision.action === 'GO' ? 'bg-emerald-500/10 border-emerald-500/50 shadow-emerald-500/10' :
              decision.action === 'COOLDOWN' ? 'bg-amber-500/10 border-amber-500/50 shadow-amber-500/10' :
              'bg-red-500/10 border-red-500/50 shadow-red-500/10'
            }`}>
              <div className="flex items-start justify-between">
                <div>
                  <span className="text-xs font-bold uppercase tracking-[0.2em] opacity-60 mb-2 block">Verdict Sentinel</span>
                  <h3 className={`text-4xl font-black mb-4 ${
                    decision.action === 'GO' ? 'text-emerald-400' : 
                    decision.action === 'COOLDOWN' ? 'text-amber-400' : 
                    'text-red-400'
                  }`}>
                    {decision.action === 'GO' ? 'EXECUTE TRADE' : 
                     decision.action === 'COOLDOWN' ? 'SYSTEM PAUSED' : 
                     decision.action === 'KILL_SWITCH' ? 'EMERGENCY STOP' : 'ORDER REJECTED'}
                  </h3>
                  <div className="flex items-center gap-2 text-slate-300 italic">
                    <Info size={16} />
                    {decision.reason}
                  </div>
                </div>
                <div className="hidden md:block">
                  {decision.action === 'GO' ? <Zap size={48} className="text-emerald-400" /> : <Shield size={48} className="text-red-400" />}
                </div>
              </div>

              {decision.action === 'GO' && (
                <div className="grid grid-cols-3 gap-4 mt-8 pt-8 border-t border-slate-700/50">
                  <div className="text-center">
                    <span className="text-xs text-slate-400 uppercase block mb-1">Take Profit</span>
                    <span className="text-2xl font-mono text-emerald-400">+{(decision.tp * 100).toFixed(1)}%</span>
                  </div>
                  <div className="text-center">
                    <span className="text-xs text-slate-400 uppercase block mb-1">Stop Loss</span>
                    <span className="text-2xl font-mono text-red-400">-{(decision.sl * 100).toFixed(1)}%</span>
                  </div>
                  <div className="text-center">
                    <span className="text-xs text-slate-400 uppercase block mb-1">Risk Scaling</span>
                    <span className="text-2xl font-mono text-blue-400">x{decision.riskScaling.toFixed(1)}</span>
                  </div>
                </div>
              )}
            </div>

            {/* METRICS GRID */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl">
                <div className="flex justify-between items-center mb-4">
                  <span className="text-slate-400 text-sm">Consensus IA</span>
                  <BarChart3 size={18} className="text-blue-400" />
                </div>
                <div className="text-3xl font-mono font-bold text-white mb-2">{decision.avg?.toFixed(1)}%</div>
                <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                  <div className="bg-blue-500 h-full transition-all duration-500" style={{width: `${decision.avg}%`}}></div>
                </div>
              </div>

              <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl">
                <div className="flex justify-between items-center mb-4">
                  <span className="text-slate-400 text-sm">Dispersion (Sigma)</span>
                  <RefreshCw size={18} className="text-purple-400" />
                </div>
                <div className={`text-3xl font-mono font-bold mb-2 ${decision.dispersion > 25 ? 'text-red-400' : 'text-purple-400'}`}>
                  {decision.dispersion?.toFixed(2)}
                </div>
                <div className="text-xs text-slate-500">Seuil critique : 25.00</div>
              </div>
            </div>

            {/* LIVE LOG MOCK */}
            <div className="bg-black/40 border border-slate-800 rounded-xl p-4 font-mono text-xs overflow-hidden">
              <div className="flex items-center gap-2 mb-2 text-slate-500">
                <div className="h-2 w-2 rounded-full bg-slate-600"></div>
                <span>DAEMON_LOG_STREAM</span>
              </div>
              <div className="space-y-1">
                <div className="text-slate-500">[22:45:01] Scan cycle initiated...</div>
                <div className="text-slate-500">[22:45:03] Found candidate: NVDA (EPS Surprise: 12.4%)</div>
                <div className="text-blue-400">[22:45:05] AI Consensus: {decision.avg?.toFixed(1)}% | Sigma: {decision.dispersion?.toFixed(2)}</div>
                <div className={decision.action === 'GO' ? 'text-emerald-400' : 'text-red-400'}>
                  [22:45:06] Decision: {decision.action} - {decision.reason}
                </div>
                {decision.action === 'GO' && (
                  <div className="text-amber-400 text-xs mt-2 p-2 bg-amber-400/5 rounded border border-amber-400/20">
                    &gt; EXÉCUTION: Bracket Order envoyé (Risk: {decision.riskScaling * 1}%)
                  </div>
                )}
              </div>
            </div>

          </div>
        </div>

        <footer className="mt-12 pt-8 border-t border-slate-900 text-center text-slate-500 text-sm">
          Système Titan v4.5 "Sentinel-Elite" • Protection du Capital par Algorithmes Adaptatifs
        </footer>

      </div>
    </div>
  );
};

export default App;
