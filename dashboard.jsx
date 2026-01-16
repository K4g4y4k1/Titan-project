import React, { useState, useEffect, useCallback } from 'react';
import { 
  Shield, 
  Activity, 
  TrendingUp, 
  AlertTriangle, 
  Zap, 
  Database, 
  Cpu, 
  Clock, 
  ArrowUpRight, 
  ArrowDownRight, 
  Power, 
  Terminal, 
  RefreshCcw, 
  Layers, 
  Settings, 
  Globe, 
  X, 
  EyeOff,
  ChevronRight,
  LogOut,
  Server,
  Key,
  ShieldAlert,
  ZapOff,
  Scale,
  Play,
  BrainCircuit,
  History,
  Target
} from 'lucide-react';

// --- CONFIGURATION VISUELLE ---
const COLORS = {
  bg: 'bg-slate-950',
  card: 'bg-slate-900/40',
  border: 'border-slate-800/60',
  accent: 'text-blue-500',
  success: 'text-emerald-400',
  danger: 'text-rose-500',
  warning: 'text-amber-400'
};

const APP_VERSION = "6.6.6-Ultimate-Semantic"; // v6.6.6 : Correction du bug de lecture Shadow

const App = () => {
  const [view, setView] = useState('login'); 
  const [apiUrl, setApiUrl] = useState('http://13.60.227.45:8080');
  const [authToken, setAuthToken] = useState('');
  const [isConnecting, setIsConnecting] = useState(false);
  const [isResuming, setIsResuming] = useState(false);
  const [loginError, setLoginError] = useState('');

  const [data, setData] = useState(null);
  const [logs, setLogs] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [showSettings, setShowSettings] = useState(false);

  // Mise à jour du flux de logs système
  const updateLogs = useCallback((msg) => {
    if (!msg) return;
    setLogs(prev => {
      const newEntry = { time: new Date().toLocaleTimeString(), msg };
      if (prev[0]?.msg === msg) return prev;
      return [newEntry, ...prev].slice(0, 25);
    });
  }, []);

  // Récupération des données via l'API (Contrat BRD v1.2)
  const fetchData = useCallback(async () => {
    if (view !== 'dashboard') return;
    try {
      const cleanUrl = apiUrl.replace(/\/$/, "");
      const headers = { 
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authToken.trim()}`
      };
      
      const response = await fetch(`${cleanUrl}/status`, { 
        method: 'GET', 
        headers,
        signal: AbortSignal.timeout(5000)
      });

      if (response.ok) {
        const json = await response.json();
        
        // Surveillance des changements d'état pour l'Audit Trail
        if (data && json.state !== data.state) {
            updateLogs(`SYSTEM_STATE_CHANGE: ${json.state}`);
        }
        if (json.state === "HALTED" && (!data || data.state !== "HALTED")) {
            updateLogs(`CRITICAL: System Halted! Reason: ${json.halt_reason || "Safety Triggered"}`);
        }
        if (json.audit && json.audit !== data?.audit) {
            updateLogs(json.audit);
        }

        setData(json);
        setIsConnected(true);
      } else {
        setIsConnected(false);
      }
    } catch (err) {
      setIsConnected(false);
    }
    setLastUpdate(new Date());
  }, [apiUrl, authToken, view, updateLogs, data]);

  useEffect(() => {
    if (view === 'dashboard') {
      fetchData();
      const interval = setInterval(fetchData, 5000);
      return () => clearInterval(interval);
    }
  }, [fetchData, view]);

  // Initialisation du lien Sentinel
  const handleConnect = async (e) => {
    if (e) e.preventDefault();
    setIsConnecting(true);
    setLoginError('');
    try {
      const cleanUrl = apiUrl.replace(/\/$/, "");
      const headers = { 
        'Content-Type': 'application/json', 
        'Authorization': `Bearer ${authToken.trim()}` 
      };
      const response = await fetch(`${cleanUrl}/status`, { method: 'GET', headers });
      if (response.ok) {
        const json = await response.json();
        setData(json);
        setIsConnected(true);
        setView('dashboard');
        updateLogs(`Liaison établie avec Titan Node ${json.version || "Unknown"}`);
      } else {
        setLoginError(`Accès refusé (Code ${response.status})`);
      }
    } catch (err) {
      setLoginError("Serveur Titan injoignable (Vérifiez IP/CORS)");
    } finally { setIsConnecting(false); }
  };

  const handleResume = async () => {
    if (!window.confirm("Voulez-vous vraiment réactiver le trading automatique ?")) return;
    setIsResuming(true);
    try {
      const cleanUrl = apiUrl.replace(/\/$/, "");
      const headers = { 'Authorization': `Bearer ${authToken.trim()}` };
      const response = await fetch(`${cleanUrl}/resume`, { method: 'POST', headers });
      if (response.ok) {
        updateLogs("MANUAL_RESUME: Commande acceptée.");
        fetchData();
      }
    } catch (err) {
      updateLogs("ERROR: Échec de la commande de reprise.");
    } finally { setIsResuming(false); }
  };

  const handleLogout = () => {
    setView('login');
    setIsConnected(false);
    setData(null);
    setAuthToken('');
    setLogs([]);
  };

  if (view === 'login') {
    return (
      <div className={`min-h-screen ${COLORS.bg} flex items-center justify-center p-4 font-sans`}>
        <div className="w-full max-w-[440px] z-10 animate-in fade-in zoom-in-95 duration-500">
          <div className="text-center mb-8">
            <div className="inline-flex bg-blue-600 p-4 rounded-3xl shadow-2xl mb-6">
              <Shield size={48} className="text-white" />
            </div>
            <h1 className="text-3xl font-black text-white tracking-tighter uppercase italic">
              Titan <span className="text-blue-500">Sentinel</span>
            </h1>
            <p className="text-slate-500 text-[10px] font-black uppercase tracking-[0.4em] mt-2">Core Access v{APP_VERSION}</p>
          </div>
          <form onSubmit={handleConnect} className="bg-slate-900/40 border border-slate-800 p-8 rounded-[2rem] backdrop-blur-2xl shadow-2xl space-y-6">
            <div className="space-y-4">
              <div className="relative group">
                <Server className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500 group-focus-within:text-blue-500 transition-colors" size={18} />
                <input type="text" value={apiUrl} onChange={(e) => setApiUrl(e.target.value)} className="w-full bg-slate-950 border border-slate-800 rounded-2xl py-4 pl-12 pr-6 text-sm text-white focus:border-blue-500 outline-none transition-all placeholder:text-slate-700" placeholder="API Endpoint" />
              </div>
              <div className="relative group">
                <Key className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500 group-focus-within:text-emerald-500 transition-colors" size={18} />
                <input type="password" value={authToken} onChange={(e) => setAuthToken(e.target.value)} className="w-full bg-slate-950 border border-slate-800 rounded-2xl py-4 pl-12 pr-6 text-sm text-white focus:border-emerald-500 outline-none transition-all placeholder:text-slate-700" placeholder="Bearer Token" />
              </div>
            </div>
            {loginError && <div className="text-rose-500 text-[10px] font-bold uppercase text-center">{loginError}</div>}
            <button type="submit" disabled={isConnecting} className="w-full bg-blue-600 hover:bg-blue-500 text-white font-black py-4 rounded-2xl transition-all flex items-center justify-center gap-3 uppercase tracking-widest text-xs shadow-xl shadow-blue-900/20">
              {isConnecting ? <RefreshCcw className="animate-spin" size={18} /> : "Initialiser Liaison"}
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className={`min-h-screen ${COLORS.bg} text-slate-200 font-sans selection:bg-blue-500/30`}>
      {/* --- BANNER D'ALERTE HALTED --- */}
      {data?.state === 'HALTED' && (
        <div className="bg-rose-600 text-white px-6 py-4 flex items-center justify-between sticky top-0 z-[60] shadow-2xl">
            <div className="flex items-center gap-4">
                <ShieldAlert size={24} className="animate-pulse" />
                <div className="flex flex-col">
                    <span className="font-black uppercase tracking-tighter text-sm italic">Système Interrompu</span>
                    <span className="text-[10px] font-bold uppercase tracking-widest opacity-80">{data.halt_reason || "Violation de sécurité"}</span>
                </div>
            </div>
            <button onClick={handleResume} disabled={isResuming} className="bg-white text-rose-600 px-6 py-2 rounded-full text-[10px] font-black uppercase hover:bg-slate-100 transition-all flex items-center gap-2">
                {isResuming ? <RefreshCcw size={14} className="animate-spin" /> : <Play size={14} fill="currentColor" />}
                Reprendre Trading
            </button>
        </div>
      )}

      {/* --- NAVIGATION PRINCIPALE --- */}
      <nav className="border-b border-slate-800/80 bg-slate-950/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-[1600px] mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="bg-blue-600 p-2 rounded-xl shadow-lg shadow-blue-500/20">
              <Shield size={24} className="text-white" />
            </div>
            <div>
              <span className="font-black text-2xl tracking-tighter text-white uppercase italic">Titan <span className="text-blue-500">Sentinel</span></span>
              <div className="flex items-center gap-2 text-[9px] uppercase tracking-[0.2em] text-slate-500 font-bold mt-1">
                <span className={`flex h-2 w-2 rounded-full ${data?.state === 'HALTED' ? 'bg-rose-500 animate-ping' : isConnected ? 'bg-emerald-500' : 'bg-slate-700'}`}></span>
                {data?.state || 'OFFLINE'} // {data?.version || 'V6.6'}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-6">
            <div className="hidden md:flex flex-col items-end px-4 border-r border-slate-800">
                <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest italic">Neural Engine</span>
                <span className="text-xs font-bold text-blue-400 uppercase">{(data?.model || data?.ai_model || "Syncing...").split('/').pop()}</span>
            </div>
            <button onClick={() => setShowSettings(true)} className="bg-slate-800/50 p-3 rounded-2xl border border-slate-700 hover:border-blue-500 transition-all">
              <Settings size={20} />
            </button>
          </div>
        </div>
      </nav>

      <main className="max-w-[1600px] mx-auto p-4 md:p-8 space-y-8">
        
        {/* --- SECTION DES KPI (CORRIGÉE v1.2) --- */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <StatCard 
            label="Equity Actual" 
            value={data ? `$${data.equity?.current?.toLocaleString()}` : '---'} 
            subValue={data?.equity ? `${data.equity.pnl_pct >= 0 ? '+' : ''}${data.equity.pnl_pct}% Session` : '---'} 
            icon={<TrendingUp size={24} />} 
            isSuccess={data?.equity?.pnl_pct >= 0} 
            isDanger={data?.equity?.pnl_pct <= -1.5} 
          />
          <StatCard 
            label="Live Exposure" 
            value={data?.positions?.live ?? 0} 
            subValue={`Active Nodes: ${data?.positions?.live ?? 0}/3`} 
            icon={<Activity size={24} />} 
            isWarning={data?.positions?.live >= 3} 
          />
          {/* FIX: On utilise shadow.open.length pour le compteur principal (Positions Actives) */}
          <StatCard 
            label="Shadow Monitoring" 
            value={data?.positions?.shadow_open ?? data?.shadow?.open?.length ?? 0} 
            subValue={data?.shadow?.stats ? `${data.shadow.stats.winrate}% WR | ${data.shadow.stats.total} Closed` : 'Scanning...'} 
            icon={<EyeOff size={24} />} 
            isSuccess={(data?.positions?.shadow_open ?? data?.shadow?.open?.length ?? 0) > 0} 
          />
          <StatCard 
            label="Sentinel Safety" 
            value={data?.safety?.consecutive_sl ?? 0} 
            subValue={data?.safety?.market_stress ? "Stress Market" : "Nominal Sequence"} 
            icon={<Shield size={24} />} 
            isDanger={data?.safety?.consecutive_sl >= 2 || data?.safety?.market_stress} 
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* --- CONTENU PRINCIPAL (GAUCHE) --- */}
          <div className="lg:col-span-8 space-y-8">
            
            {/* DERNIER TRADE DB (INSIGHTS) */}
            {data?.trades?.last && (
                <section className="bg-blue-600/5 border-2 border-blue-500/20 rounded-[2.5rem] p-8 shadow-2xl relative overflow-hidden group animate-in slide-in-from-left duration-700">
                    <div className="absolute top-0 right-0 p-8 opacity-5 group-hover:opacity-10 transition-opacity">
                        <History size={120} />
                    </div>
                    <div className="flex justify-between items-start mb-6">
                        <div className="flex flex-col gap-1">
                            <span className="text-[10px] font-black text-blue-500 uppercase tracking-widest">Dernière Action Persistée</span>
                            <h3 className="text-3xl font-black text-white italic tracking-tighter uppercase flex items-center gap-3">
                                {data.trades.last.symbol}
                                <span className={`text-[10px] px-3 py-1 rounded-full font-black tracking-widest ${data.trades.last.mode === 'LIVE' ? 'bg-rose-500/20 text-rose-500' : 'bg-blue-500/20 text-blue-500'}`}>
                                    {data.trades.last.mode}
                                </span>
                            </h3>
                        </div>
                        <div className="text-right">
                             <span className="text-[10px] font-black text-slate-500 uppercase block mb-1">Conviction IA</span>
                             <span className="text-xl font-black text-emerald-400">{data.trades.last.confidence}%</span>
                        </div>
                    </div>
                    <div className="bg-slate-950/50 rounded-2xl p-6 border border-slate-800/50">
                        <p className="text-xs text-slate-400 italic leading-relaxed">
                            "{data.trades.last.thesis || "Analyse comportementale en cours..."}"
                        </p>
                        <div className="mt-4 flex flex-wrap gap-8 border-t border-slate-800 pt-4">
                             <div className="flex flex-col">
                                 <span className="text-[9px] font-bold text-slate-600 uppercase">Entrée</span>
                                 <span className="text-xs font-mono font-bold text-slate-300">${data.trades.last.entry_price}</span>
                             </div>
                             <div className="flex flex-col">
                                 <span className="text-[9px] font-bold text-slate-600 uppercase">Statut</span>
                                 <span className={`text-xs font-bold uppercase ${data.trades.last.status === 'OPEN' ? 'text-blue-400 animate-pulse' : 'text-slate-500'}`}>{data.trades.last.status}</span>
                             </div>
                             <div className="flex flex-col ml-auto">
                                 <span className="text-[9px] font-bold text-slate-600 uppercase italic">Execution Time</span>
                                 <span className="text-xs font-mono text-slate-500">{data.trades.last.entry_time}</span>
                             </div>
                        </div>
                    </div>
                </section>
            )}

            {/* MONITORING SHADOW (TERMINAL ACTIF) */}
            <section className="bg-slate-900/40 border-2 border-slate-800 rounded-[2.5rem] overflow-hidden shadow-2xl transition-all">
                <div className="px-8 py-6 border-b border-slate-800 bg-slate-950/30 flex justify-between items-center">
                    <h3 className="text-sm font-black uppercase tracking-tighter italic text-emerald-400 flex items-center gap-3">
                        <EyeOff size={20} /> Shadow Terminal Active
                    </h3>
                    <div className="flex items-center gap-4">
                        <div className="flex items-center gap-2 px-3 py-1 bg-emerald-500/10 border border-emerald-500/20 rounded-full">
                            <span className="flex h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                            <span className="text-[10px] font-black text-emerald-500 uppercase tracking-widest">Monitoring_On</span>
                        </div>
                    </div>
                </div>
                <div className="p-0 overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                        <thead className="bg-slate-950/50 text-[10px] font-black uppercase text-slate-500 tracking-widest">
                            <tr>
                                <th className="px-8 py-4">Symbol</th>
                                <th className="px-8 py-4 text-center">Neural Confidence</th>
                                <th className="px-8 py-4">Protection (TP/SL)</th>
                                <th className="px-8 py-4 text-right italic">Duration</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800 font-mono text-xs">
                            {(data?.shadow?.open || []).map((pos, i) => (
                                <tr key={i} className="group hover:bg-emerald-500/5 transition-all">
                                    <td className="px-8 py-5">
                                        <div className="flex flex-col">
                                            <span className="text-lg font-black text-white">{pos.symbol}</span>
                                            <span className="text-[9px] text-slate-600 uppercase font-bold tracking-tighter">SIM_ENTRY @ ${pos.entry_price}</span>
                                        </div>
                                    </td>
                                    <td className="px-8 py-5 text-center">
                                        <div className="inline-flex flex-col items-center">
                                            <span className="text-emerald-500 font-black">{pos.confidence}%</span>
                                            <div className="w-12 bg-slate-800 h-1 rounded-full mt-1 overflow-hidden">
                                                <div className="bg-emerald-500 h-full" style={{width: `${pos.confidence}%`}}></div>
                                            </div>
                                        </div>
                                    </td>
                                    <td className="px-8 py-5">
                                        <div className="flex flex-col gap-0.5 font-bold">
                                            <span className="text-emerald-500/70">TP: ${pos.tp}</span>
                                            <span className="text-rose-500/70">SL: ${pos.sl}</span>
                                        </div>
                                    </td>
                                    <td className="px-8 py-5 text-right">
                                        <span className="text-slate-500 font-bold">{pos.duration_min} min</span>
                                    </td>
                                </tr>
                            ))}
                            {(data?.shadow?.open || []).length === 0 && (
                                <tr>
                                    <td colSpan="4" className="px-8 py-16 text-center text-slate-700 italic font-black uppercase tracking-[0.2em] opacity-30">
                                        No active shadow positions. Scanning for institutional picks.
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </section>
          </div>

          {/* --- SIDEBAR (DROITE) --- */}
          <div className="lg:col-span-4 space-y-8">
            
            {/* FLUX DE DÉCISIONS IA (CERVEAU) */}
            <section className="bg-slate-950 border-2 border-slate-800 rounded-[2.5rem] p-8 shadow-2xl h-[520px] flex flex-col relative overflow-hidden group">
                <div className="flex justify-between items-center mb-6 border-b border-slate-900 pb-4">
                    <h3 className="text-[10px] font-black uppercase tracking-[0.3em] text-slate-500 flex items-center gap-3 italic">
                        <BrainCircuit size={16} className="text-blue-500" /> Neural Decision Stream
                    </h3>
                </div>
                <div className="flex-1 overflow-y-auto space-y-5 scrollbar-hide">
                    {(data?.decisions?.recent || []).map((dec, i) => (
                        <div key={i} className="bg-slate-900/40 border border-slate-800/50 rounded-2xl p-4 animate-in slide-in-from-bottom-2 duration-300 hover:border-blue-500/30 transition-colors">
                            <div className="flex justify-between items-center mb-2">
                                <span className="text-lg font-black text-white italic tracking-tighter">{dec.symbol}</span>
                                <span className="text-[10px] font-mono text-slate-600">{dec.time}</span>
                            </div>
                            <div className="flex items-center gap-2 mb-2">
                                <span className={`text-[8px] font-black px-2 py-0.5 rounded border uppercase tracking-widest ${dec.mode === 'LIVE' ? 'border-rose-500/30 text-rose-400' : 'border-blue-500/30 text-blue-400'}`}>
                                    {dec.mode}
                                </span>
                                <span className="text-[10px] font-bold text-emerald-400 opacity-80">{dec.confidence}% Conviction</span>
                            </div>
                            <p className="text-[10px] text-slate-500 italic leading-relaxed line-clamp-3">
                                "{dec.thesis}"
                            </p>
                        </div>
                    ))}
                    {(data?.decisions?.recent || []).length === 0 && (
                         <div className="text-center py-20 text-slate-800 font-black uppercase text-[10px] tracking-widest opacity-30 flex flex-col items-center gap-4">
                            <RefreshCcw size={20} className="animate-spin text-slate-800" />
                            Awaiting AI Decision Logs...
                         </div>
                    )}
                </div>
                <div className="mt-4 pt-4 border-t border-slate-900 flex justify-between items-center">
                     <span className="text-[8px] font-black text-slate-700 uppercase tracking-widest italic flex items-center gap-1">
                        <Database size={8} /> Sync with Persistent DB
                     </span>
                     <div className="flex h-1.5 w-1.5 rounded-full bg-blue-500 animate-pulse"></div>
                </div>
            </section>

            {/* AUDIT STREAM (UI LOGS) */}
            <section className="bg-slate-900/40 border border-slate-800 rounded-[2.5rem] overflow-hidden">
                <div className="px-6 py-4 border-b border-slate-800 bg-slate-950/20 flex justify-between items-center">
                    <h3 className="text-[9px] font-black uppercase tracking-widest text-slate-500 flex items-center gap-2 italic">
                        <Terminal size={14} className="text-emerald-500" /> System Audit (UI)
                    </h3>
                </div>
                <div className="p-6 h-[200px] overflow-y-auto space-y-3 font-mono text-[10px] scrollbar-hide">
                    {logs.map((log, i) => (
                        <div key={i} className="flex gap-3 border-l-2 border-slate-800 pl-3 py-1 opacity-60 hover:opacity-100 transition-opacity">
                            <span className="text-slate-600 shrink-0 font-bold">[{log.time}]</span>
                            <span className={log.msg.includes('CRITICAL') || log.msg.includes('HALTED') ? 'text-rose-500 font-black' : 'text-slate-400'}>{log.msg}</span>
                        </div>
                    ))}
                </div>
            </section>
          </div>
        </div>
      </main>

      {/* --- MODAL DE CONFIGURATION --- */}
      {showSettings && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-slate-950/95 backdrop-blur-xl p-4">
          <div className="bg-slate-900 border-2 border-slate-800 w-full max-w-md rounded-[2.5rem] p-8 shadow-2xl animate-in slide-in-from-bottom-4 duration-300">
            <div className="flex justify-between items-center mb-10">
              <h2 className="text-xl font-black flex items-center gap-3 uppercase tracking-tighter italic"><Settings size={24} className="text-blue-500" /> Configuration Session</h2>
              <button onClick={() => setShowSettings(false)} className="text-slate-500 hover:text-white p-2 bg-slate-800 rounded-xl transition-colors"><X size={20} /></button>
            </div>
            <div className="space-y-6 text-left">
                <ConfigField label="Endpoint Actif" value={apiUrl} />
                <ConfigField label="Version Moteur" value={data?.version || APP_VERSION} />
                <ConfigField label="ID Volatile" value={authToken ? "Active Bearer" : "No Auth"} />
                <button onClick={handleLogout} className="w-full bg-rose-600/10 hover:bg-rose-600 text-rose-500 hover:text-white border border-rose-500/20 font-black py-5 rounded-2xl transition-all flex items-center justify-center gap-3 uppercase tracking-widest text-xs shadow-xl active:scale-95">
                  <LogOut size={18} /> Terminer Session Volatile
                </button>
            </div>
          </div>
        </div>
      )}

      {/* --- PIED DE PAGE --- */}
      <footer className="border-t border-slate-900 py-4 px-8 bg-slate-950/80 backdrop-blur-md">
        <div className="max-w-[1600px] mx-auto flex justify-between items-center text-[9px] font-mono text-slate-700 uppercase tracking-[0.3em] font-black">
          <span className="flex items-center gap-2"><Globe size={10}/> Node Liaison: {apiUrl.replace('http://', '').replace('https://', '')}</span>
          <div className="flex items-center gap-4">
             <div className="flex items-center gap-2">
                <div className={`h-1.5 w-1.5 rounded-full ${isConnected ? 'bg-blue-500 shadow-lg shadow-blue-500' : 'bg-rose-500'}`}></div>
                DB_PULSE: {lastUpdate.toLocaleTimeString()}
             </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

// --- COMPOSANTS ATOMIQUES ---

const StatCard = ({ label, value, subValue, icon, isSuccess, isWarning, isDanger }) => (
  <div className={`bg-slate-900/40 border-2 p-6 rounded-[2rem] transition-all duration-500 relative overflow-hidden group ${isDanger ? 'border-rose-500/50 shadow-xl shadow-rose-900/10' : isWarning ? 'border-amber-500/50 shadow-xl shadow-amber-900/10' : 'border-slate-800/60 hover:border-blue-500/50'}`}>
    <div className="flex justify-between items-start mb-4">
      <div className={`p-3 rounded-2xl border border-slate-700/50 transition-colors ${isDanger ? 'bg-rose-500/20 text-rose-400' : 'bg-slate-800/50 text-slate-400 group-hover:text-blue-400'}`}>{icon}</div>
      <div className={`text-[9px] font-black px-4 py-1.5 rounded-full border tracking-widest uppercase italic transition-all ${isSuccess ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' : isWarning ? 'bg-amber-500/10 text-amber-400 border-amber-500/20' : isDanger ? 'bg-rose-500/10 text-rose-400 border-rose-500/20' : 'bg-slate-800 text-slate-600 border-slate-700'}`}>{subValue}</div>
    </div>
    <div className="flex flex-col relative z-10">
      <span className="text-[10px] font-black text-slate-600 uppercase tracking-[0.2em] mb-1.5">{label}</span>
      <span className="text-3xl font-black text-white tracking-tighter italic uppercase group-hover:scale-105 transition-transform origin-left duration-500">{value}</span>
    </div>
    <div className="absolute -right-4 -bottom-4 opacity-[0.02] group-hover:opacity-[0.05] transition-opacity duration-700 pointer-events-none">
        {icon && React.cloneElement(icon, { size: 100 })}
    </div>
  </div>
);

const ConfigField = ({ label, value }) => (
    <div className="p-5 bg-slate-950 rounded-2xl border border-slate-800 font-mono text-[10px] text-slate-400">
        <span className="text-slate-600 uppercase font-black text-[9px] tracking-widest block mb-2 underline decoration-slate-900">{label}</span>
        <span className="text-blue-400 break-all font-bold tracking-tight">{value}</span>
    </div>
);

const SafetyMetric = ({ label, value, limit, percent, color }) => {
    const colors = { rose: 'bg-rose-500 shadow-rose-500/50', amber: 'bg-amber-500 shadow-amber-500/50', blue: 'bg-blue-500 shadow-blue-500/50' };
    return (
        <div className="bg-slate-950/50 border border-slate-800 rounded-[1.5rem] p-6 space-y-4 hover:border-slate-700 transition-colors shadow-inner">
            <div className="flex justify-between items-center"><span className="text-[9px] font-black text-slate-600 uppercase tracking-[0.2em]">{label}</span><span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">Limite {limit}</span></div>
            <div className="text-2xl font-black text-white tracking-tighter italic uppercase">{value}</div>
            <div className="w-full bg-slate-900 h-1.5 rounded-full overflow-hidden"><div className={`${colors[color]} h-full transition-all duration-1000`} style={{ width: `${Math.min(percent, 100)}%` }}></div></div>
        </div>
    );
};

export default App;
