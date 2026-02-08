'use client'

import React, { useState, useEffect, useMemo } from 'react';
import { 
  Activity, 
  ShieldCheck, 
  Database, 
  Cpu, 
  UserCheck, 
  ImageIcon, 
  AlertTriangle, 
  BrainCircuit, 
  FileSearch, 
  CheckCircle2, 
  Clock,
  ArrowRight,
  ChevronRight,
  Layers,
  Terminal,
  Info,
  Code,
  Zap,
  Globe,
  Lock,
  Server,
  FileText,
  BarChart3,
  Search,
  AlertOctagon
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { api, Post, AnalysisResult } from './api';

// Type definitions
interface Phase {
  phase_id: number;
  name: string;
  iconComponent: string;
  status: string;
  purpose: string;
  inputs: any;
  processing_steps: string[];
  outputs: any;
  passes_output_to: number[];
}

// Dataset will be loaded from API

// Account ID to Name mapping (can be extended or fetched from API)
const ACCOUNT_NAME_MAP: Record<string, { name: string; screen_name?: string; description?: string }> = {
  'acc_19701628': {
    name: 'BBC',
    screen_name: 'BBC',
    description: 'The BBC is the world\'s leading public service broadcaster'
  },
  'acc_14075928': {
    name: 'The Onion',
    screen_name: 'TheOnion',
    description: 'America\'s Finest News Source'
  }
};

// Icon components mapping
const iconComponents: { [key: string]: React.ComponentType<{ className?: string }> } = {
  Database,
  Activity,
  UserCheck,
  ImageIcon,
  AlertTriangle,
  BrainCircuit,
  FileSearch,
  ShieldCheck,
};

const getProjectData = () => ({
  "project_metadata": {
    "project_name": "GenAI Social Media Credibility Analyzer",
    "project_type": "research_prototype",
    "intended_use": ["college research paper", "patent working model", "local system demonstration"],
    "execution_environment": "local_machine",
    "design_goals": ["multimodal credibility analysis", "explainable AI", "agent-based reasoning"]
  },
  "current_execution_mode": {
    "mode": "single_dataset_testing",
    "active_inputs": ["processed_social_media_dataset"]
  },
  "global_data_schema": {
    "post_id": "string", "platform": "string", "text": "string", "image_path": "string | null", "urls": "array[string]",
    "hashtags": "array[string]", "likes": "integer", "retweets": "integer", "timestamp": "YYYY-MM-DD",
    "account": { "account_id": "string", "account_age_days": "integer", "verified": "boolean", "historical_post_count": "integer" }
  },
  "phases": [
    { phase_id: 1, name: "Data Ingestion & Normalization", iconComponent: "Database", status: "completed", purpose: "Convert raw or platform-specific data into a unified schema.", inputs: { current: "raw_or_preprocessed_dataset" }, processing_steps: ["field_extraction", "schema_mapping", "format_standardization", "data_validation"], outputs: { normalized_posts: "array[global_data_schema]" }, passes_output_to: [2, 3, 4] },
    { phase_id: 2, name: "Textual Intelligence (NLP)", iconComponent: "Activity", status: "pending", purpose: "Extract linguistic, emotional, and semantic credibility signals.", inputs: { from_phase_1: ["post_id", "text", "hashtags", "urls"] }, processing_steps: ["sentiment_analysis", "emotion_detection", "clickbait_detection", "claim_extraction"], outputs: { nlp_signals: { sentiment: "string", emotion: "string", clickbait: "boolean", extracted_claim: "string" } }, passes_output_to: [5, 6] },
    { phase_id: 3, name: "Source & Account Analysis", iconComponent: "UserCheck", status: "pending", purpose: "Assess credibility based on account behavior.", inputs: { from_phase_1: ["account_age", "verified", "history"] }, processing_steps: ["account_trust_scoring", "domain_reliability_evaluation", "behavioral_heuristic_analysis"], outputs: { source_signals: { account_trust_score: "float", source_reliability_score: "float" } }, passes_output_to: [5, 6] },
    { phase_id: 4, name: "Image & Visual Analysis", iconComponent: "ImageIcon", status: "pending", purpose: "Detect visual manipulation and AI generation.", inputs: { from_phase_1: ["image_path"] }, processing_steps: ["ocr_text_extraction", "image_tampering_detection", "ai_generated_image_estimation"], outputs: { image_signals: { ocr_text: "string", image_tampered: "boolean", ai_generated_prob: "float" } }, passes_output_to: [2, 5, 6] },
    { phase_id: 5, name: "Misinformation Modeling", iconComponent: "AlertTriangle", status: "pending", purpose: "Fuse multimodal signals to estimate misinformation likelihood.", inputs: { sources: ["nlp_signals", "source_signals", "image_signals"] }, processing_steps: ["fake_news_classification", "signal_fusion", "score_computation"], outputs: { misinformation_assessment: { credibility_score: "float", risk_category: "string" } }, passes_output_to: [6] },
    { phase_id: 6, name: "Agentic Reasoning", iconComponent: "BrainCircuit", status: "pending", purpose: "Perform multi-agent evaluation and resolve conflicts.", inputs: { sources: ["signals", "assessment"] }, processing_steps: ["text_agent_eval", "image_agent_eval", "conflict_resolution"], outputs: { final_decision: { final_score: "float", reasoning_trace: "array" } }, passes_output_to: [7] },
    { phase_id: 7, name: "Explainability & Feedback", iconComponent: "FileSearch", status: "pending", purpose: "Generate human-readable explanations and warnings.", inputs: { from_phase_6: "final_decision" }, processing_steps: ["explanation_generation", "warning_label_assignment"], outputs: { user_facing_result: { credibility_score: "float", warning_label: "string", explanation: "array" } }, passes_output_to: [8] },
    { phase_id: 8, name: "Evaluation & Validation", iconComponent: "ShieldCheck", status: "pending", purpose: "Evaluate system behavior and document performance.", inputs: { from_phase_7: "user_facing_result" }, processing_steps: ["metric_computation", "error_logging"], outputs: { evaluation_report: { metrics: ["accuracy", "f1_score"] } }, passes_output_to: [] }
  ],
  "future_extensibility": {
    "platforms": ["facebook", "instagram", "youtube"],
    "features": ["live_processing", "real_time_alerts", "horizontal_scaling"]
  }
});

const projectData = getProjectData();

const JSONTree = ({ data }: { data: any }) => (
  <div className="font-mono text-xs leading-relaxed text-indigo-200/80">
    {Object.entries(data).map(([key, value]) => (
      <div key={key} className="ml-4">
        <span className="text-indigo-400">{key}:</span>{' '}
        {typeof value === 'object' && value !== null ? (
          Array.isArray(value) ? (
            <span className="text-slate-400">[{value.length} items]</span>
          ) : (
            <span className="text-slate-400">{'{...}'}</span>
          )
        ) : (
          <span className="text-emerald-400">"{value?.toString()}"</span>
        )}
      </div>
    ))}
  </div>
);

export default function App() {
  const [activePhase, setActivePhase] = useState<string | number>('overview'); // 'overview', 'dataset', 'results', or phase_id (number)
  const [isSimulating, setIsSimulating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [showSchema, setShowSchema] = useState(false);
  const [selectedPost, setSelectedPost] = useState<Post | null>(null);
  const [dataset, setDataset] = useState<Post[]>([]);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [apiConnected, setApiConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedHandle, setSelectedHandle] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'handles' | 'tweets'>('handles');

  // Load dataset on mount
  useEffect(() => {
    const loadDataset = async () => {
      try {
        const data = await api.getDataset();
        setDataset(data);
      } catch (err) {
        console.error('Failed to load dataset:', err);
        setError('Failed to load dataset. Make sure the API server is running.');
      }
    };
    loadDataset();
  }, []);

  // Check API health on mount
  useEffect(() => {
    const checkHealth = async () => {
      const healthy = await api.checkHealth();
      setApiConnected(healthy);
      if (!healthy) {
        setError('API server is not connected. Please start the backend server.');
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, []);

  // Helper function to render icon component
  const renderIcon = (iconComponentName: string, className: string = "w-5 h-5") => {
    const IconComponent = iconComponents[iconComponentName];
    return IconComponent ? <IconComponent className={className} /> : null;
  };

  // Extract handle/username from account data, URLs, or account_id
  const extractHandle = (post: Post): string => {
    // Priority 1: Use account name or screen_name if available in post data
    if (post.account.screen_name) {
      return `@${post.account.screen_name}`;
    }
    if (post.account.name) {
      return post.account.name;
    }
    
    // Priority 2: Look up account name from mapping table
    const accountId = post.account.account_id;
    if (accountId && ACCOUNT_NAME_MAP[accountId]) {
      const accountInfo = ACCOUNT_NAME_MAP[accountId];
      if (accountInfo.screen_name) {
        return `@${accountInfo.screen_name}`;
      }
      return accountInfo.name;
    }
    
    // Priority 3: Try to extract from URLs
    for (const url of post.urls) {
      if (url.includes('theonion.com')) return '@TheOnion';
      if (url.includes('bbc.co.uk') || url.includes('bbc.com')) return '@BBC';
      if (url.includes('twitter.com/')) {
        const match = url.match(/twitter\.com\/([^\/\?]+)/);
        if (match) return `@${match[1]}`;
      }
    }
    
    // Priority 4: Use account_id as fallback (but format it nicely)
    if (accountId) {
      if (accountId.startsWith('acc_')) {
        // Extract numeric part and show as ID
        return accountId;
      }
      return accountId;
    }
    
    // Final fallback
    return 'Unknown Account';
  };

  // Get account display info (name, description, etc.)
  const getAccountInfo = (post: Post) => {
    const accountId = post.account.account_id;
    
    // Check if account info is in post data
    if (post.account.name || post.account.screen_name) {
      return {
        name: post.account.name || post.account.screen_name || '',
        screen_name: post.account.screen_name || post.account.name || '',
        description: post.account.description || ''
      };
    }
    
    // Check mapping table
    if (accountId && ACCOUNT_NAME_MAP[accountId]) {
      return ACCOUNT_NAME_MAP[accountId];
    }
    
    // Fallback
    return {
      name: accountId || 'Unknown',
      screen_name: accountId || 'Unknown',
      description: ''
    };
  };

  // Group posts by handle/account
  const groupedByHandle = useMemo(() => {
    const groups: Record<string, { handle: string; account: Post['account']; posts: Post[] }> = {};
    
    dataset.forEach(post => {
      const handle = extractHandle(post);
      if (!groups[handle]) {
        groups[handle] = {
          handle,
          account: post.account,
          posts: []
        };
      }
      groups[handle].posts.push(post);
    });

    // Sort posts by timestamp (newest first)
    Object.values(groups).forEach(group => {
      group.posts.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
    });

    return groups;
  }, [dataset]);

  // Get posts for selected handle
  const handlePosts = selectedHandle ? groupedByHandle[selectedHandle]?.posts || [] : [];

  // Get image URL helper
  const getImageUrl = (imagePath: string | null): string | null => {
    if (!imagePath) return null;
    // Images are in data/media/image/ folder
    // Try multiple possible paths
    const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    
    // If it's already a full URL, return as is
    if (imagePath.startsWith('http://') || imagePath.startsWith('https://')) {
      return imagePath;
    }
    
    // Try direct image path first (if API serves from root)
    const directPath = imagePath.startsWith('data/') ? imagePath : `data/media/image/${imagePath}`;
    
    // Return API endpoint for images
    return `${apiBase}/api/images/${encodeURIComponent(directPath)}`;
  };

  // Image Display Component
  const ImageDisplay = ({ imageUrl, imagePath }: { imageUrl: string; imagePath: string | null }) => {
    const [isLoading, setIsLoading] = useState(true);
    const [hasError, setHasError] = useState(false);

    return (
      <div className="mb-4 rounded-lg overflow-hidden border border-slate-800 bg-slate-950/50 group/image">
        <div className="relative w-full bg-slate-900/50 flex items-center justify-center min-h-[200px] max-h-[500px]">
          {isLoading && !hasError && (
            <div className="absolute inset-0 bg-slate-900/80 flex items-center justify-center z-10">
              <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
            </div>
          )}
          {hasError ? (
            <div className="w-full h-full flex items-center justify-center text-slate-500 p-8">
              <div className="text-center">
                <ImageIcon className="w-12 h-12 mx-auto mb-2 text-slate-600" />
                <p className="text-xs text-slate-500">Image not available</p>
                {imagePath && (
                  <p className="text-xs text-slate-600 mt-1 font-mono">{imagePath}</p>
                )}
              </div>
            </div>
          ) : (
            <img 
              src={imageUrl} 
              alt="Tweet media"
              className="w-full h-auto max-h-[500px] object-contain transition-transform group-hover/image:scale-[1.02]"
              onError={() => {
                setIsLoading(false);
                setHasError(true);
              }}
              onLoad={() => {
                setIsLoading(false);
              }}
            />
          )}
        </div>
        {imagePath && !hasError && (
          <div className="px-3 py-2 bg-slate-900/30 border-t border-slate-800">
            <p className="text-xs text-slate-500 font-mono truncate flex items-center gap-2">
              <ImageIcon className="w-3 h-3" />
              {imagePath}
            </p>
          </div>
        )}
      </div>
    );
  };

  // Simulation logic - now calls real API
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isSimulating && selectedPost) {
      // Call API for real analysis
      const runAnalysis = async () => {
        setIsLoading(true);
        setError(null);
        try {
          const result = await api.analyzePost(selectedPost);
          setAnalysisResult(result);
          
          // Simulate progress through phases
          let currentPhase = 1;
          interval = setInterval(() => {
            currentPhase++;
            setProgress(currentPhase);
            setActivePhase(currentPhase);
            
            if (currentPhase >= 8) {
              setIsSimulating(false);
              setIsLoading(false);
              setActivePhase('results');
              clearInterval(interval);
            }
          }, 2000); // 2 seconds per phase
        } catch (err: any) {
          setIsSimulating(false);
          setIsLoading(false);
          setError(err.message || 'Analysis failed');
          console.error('Analysis error:', err);
        }
      };
      
      runAnalysis();
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isSimulating, selectedPost]);

  const resetSimulation = () => {
    setProgress(0);
    setActivePhase('overview');
    setIsSimulating(false);
    setSelectedPost(null);
    setViewMode('handles');
    setSelectedHandle(null);
  };

  const handleStartSimulation = async () => {
    if (!selectedPost) {
      setActivePhase('dataset'); // Redirect to dataset if no post selected
      return;
    }
    if (!apiConnected) {
      setError('API server is not connected. Please start the backend server.');
      return;
    }
    setActivePhase(1);
    setIsSimulating(true);
    setProgress(1);
    setAnalysisResult(null);
    setError(null);
  };

  const getActivePhaseData = () => {
    if (typeof activePhase !== 'number') return null;
    return projectData.phases.find(p => p.phase_id === activePhase);
  };

  const activeData = getActivePhaseData();

  // Get results from analysis or generate placeholder
  const results = analysisResult?.user_facing_output || null;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-indigo-500/30 overflow-hidden">
      {/* Background Ambience */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-indigo-900/10 blur-[150px] rounded-full" />
        <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] bg-fuchsia-900/10 blur-[150px] rounded-full" />
        <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 brightness-100 contrast-150 mix-blend-overlay"></div>
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-6 py-6 h-screen flex flex-col">
        {/* Header */}
        <header className="flex-none flex items-center justify-between border-b border-slate-800/60 pb-6 mb-6">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <BrainCircuit className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
                {projectData.project_metadata.project_name}
              </h1>
              <div className="flex items-center gap-3 mt-1 text-xs text-slate-500">
                <span className="px-2 py-0.5 bg-slate-900 border border-slate-800 rounded text-slate-400">
                  {projectData.project_metadata.project_type}
                </span>
                <span>•</span>
                <span className="flex items-center gap-1">
                  <Server className="w-3 h-3" /> {projectData.project_metadata.execution_environment}
                </span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
             <button 
              onClick={() => setShowSchema(!showSchema)}
              className={`p-2.5 rounded-lg border transition-all ${showSchema ? 'bg-indigo-500/10 border-indigo-500/50 text-indigo-400' : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-white'}`}
              title="View Data Schema"
            >
              <Code className="w-5 h-5" />
            </button>
            <div className="h-8 w-px bg-slate-800 mx-1"></div>
            
            <button 
              onClick={handleStartSimulation}
              disabled={isSimulating}
              className={`px-6 py-2.5 rounded-lg font-semibold text-sm transition-all flex items-center gap-2 ${
                isSimulating 
                ? 'bg-amber-500/10 text-amber-500 border border-amber-500/20 cursor-wait' 
                : !selectedPost 
                  ? 'bg-slate-800 text-slate-400 border border-slate-700 hover:bg-slate-700'
                  : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-500/20'
              }`}
            >
              {isSimulating ? <Cpu className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
              {isSimulating ? 'Processing...' : !selectedPost ? 'Select Input First' : 'Run Simulation'}
            </button>

            <button 
              onClick={resetSimulation}
              className="p-2.5 rounded-lg border border-slate-800 hover:bg-slate-900 hover:text-white text-slate-500 transition-colors"
              title="Reset"
            >
              <Clock className="w-5 h-5" />
            </button>
          </div>
        </header>

        {/* Main Layout */}
        <div className="flex-1 flex gap-8 min-h-0">
          
          {/* Left Sidebar: Navigation */}
          <div className="w-80 flex-none overflow-y-auto pr-2 custom-scrollbar">
            
            {/* General Navigation */}
            <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4 flex items-center gap-2">
              <Layers className="w-3 h-3" /> Views
            </h2>
            <div className="space-y-2 mb-8">
              <button onClick={() => setActivePhase('overview')} className={`w-full text-left p-3 rounded-xl border transition-all flex items-center gap-3 ${activePhase === 'overview' ? 'bg-slate-900 border-indigo-500/50 text-white' : 'border-transparent text-slate-400 hover:text-white'}`}>
                <Activity className="w-4 h-4" /> System Overview
              </button>
              <button onClick={() => setActivePhase('dataset')} className={`w-full text-left p-3 rounded-xl border transition-all flex items-center gap-3 ${activePhase === 'dataset' ? 'bg-slate-900 border-indigo-500/50 text-white' : 'border-transparent text-slate-400 hover:text-white'}`}>
                <Database className="w-4 h-4" /> Input Data Explorer
              </button>
              <button 
                onClick={() => progress >= 8 && setActivePhase('results')} 
                disabled={progress < 8}
                className={`w-full text-left p-3 rounded-xl border transition-all flex items-center gap-3 ${activePhase === 'results' ? 'bg-slate-900 border-indigo-500/50 text-white' : progress < 8 ? 'border-transparent text-slate-600 cursor-not-allowed' : 'border-transparent text-emerald-400 hover:bg-emerald-500/10'}`}
              >
                <BarChart3 className="w-4 h-4" /> Analysis Report
                {progress >= 8 && <CheckCircle2 className="w-3 h-3 ml-auto" />}
              </button>
            </div>

            <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4 flex items-center gap-2">
              <Cpu className="w-3 h-3" /> Pipeline Stages
            </h2>
            <div className="space-y-2">
              {projectData.phases.map((phase) => (
                <button
                  key={phase.phase_id}
                  onClick={() => typeof activePhase === 'number' && setActivePhase(phase.phase_id)}
                  className={`w-full text-left p-3 rounded-xl border transition-all flex items-center gap-3 group relative overflow-hidden ${
                    activePhase === phase.phase_id
                      ? 'bg-slate-900 border-indigo-500/50 shadow-lg shadow-indigo-500/5'
                      : 'bg-transparent border-slate-800/40'
                  } ${typeof activePhase !== 'number' ? 'opacity-50 cursor-default' : 'hover:border-slate-700 hover:bg-slate-900/30'}`}
                >
                  {/* Progress Bar Background for Phase */}
                  {phase.phase_id === progress && isSimulating && (
                    <motion.div 
                      layoutId="phase-active-bg"
                      className="absolute inset-0 bg-indigo-500/5 z-0" 
                      initial={{ width: "0%" }}
                      animate={{ width: "100%" }}
                      transition={{ duration: 1.0, ease: "linear" }}
                    />
                  )}

                  <div className={`relative z-10 p-2 rounded-lg transition-colors ${
                    activePhase === phase.phase_id ? 'bg-indigo-500 text-white' : 'bg-slate-800 text-slate-500'
                  }`}>
                    {renderIcon(phase.iconComponent, "w-5 h-5")}
                  </div>
                  <div className="flex-1 relative z-10">
                    <div className="flex items-center justify-between">
                      <span className={`text-sm font-medium ${activePhase === phase.phase_id ? 'text-white' : 'text-slate-400'}`}>
                        {phase.name}
                      </span>
                      {phase.phase_id < progress && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />}
                    </div>
                  </div>
                </button>
              ))}
            </div>

            {selectedPost && (
               <div className="mt-8 p-4 rounded-xl border border-indigo-500/30 bg-indigo-500/10">
                 <h3 className="text-xs font-semibold text-indigo-300 mb-2 flex items-center gap-2">
                   <FileText className="w-3 h-3" /> Active Input
                 </h3>
                 <p className="text-xs text-indigo-200 line-clamp-2 italic opacity-80">
                   "{selectedPost.text}"
                 </p>
                 <div className="mt-2 text-[10px] text-indigo-400 font-mono">
                   ID: {selectedPost.post_id}
                 </div>
               </div>
            )}
          </div>

          {/* Center Content Area */}
          <div className="flex-1 flex flex-col min-h-0 relative">
            
            {/* Schema Overlay */}
            <AnimatePresence>
              {showSchema && (
                <motion.div 
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  className="absolute right-0 top-0 bottom-0 w-80 bg-slate-900 border-l border-slate-800 shadow-2xl z-50 p-6 overflow-y-auto"
                >
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="font-bold text-white flex items-center gap-2">
                      <Database className="w-4 h-4 text-indigo-400" /> Global Schema
                    </h3>
                    <button onClick={() => setShowSchema(false)} className="text-slate-500 hover:text-white">
                      <ChevronRight className="w-5 h-5" />
                    </button>
                  </div>
                  <pre className="text-[10px] text-slate-300 font-mono whitespace-pre-wrap leading-relaxed">
                    {JSON.stringify(projectData.global_data_schema, null, 2)}
                  </pre>
                </motion.div>
              )}
            </AnimatePresence>

            <div className="flex-1 overflow-y-auto custom-scrollbar pr-2">
              <AnimatePresence mode="wait">
                
                {/* View: System Overview */}
                {activePhase === 'overview' && (
                  <motion.div
                    key="overview"
                    initial={{ opacity: 0, scale: 0.98 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.98 }}
                    transition={{ duration: 0.2 }}
                    className="space-y-6"
                  >
                    <div className="p-8 bg-gradient-to-br from-indigo-900/20 via-slate-900/50 to-slate-900/50 rounded-2xl border border-slate-800">
                      <h2 className="text-2xl font-bold text-white mb-4">Research Prototype Dashboard</h2>
                      <p className="text-slate-400 max-w-3xl leading-relaxed">
                        Welcome to the control center for the GenAI Social Media Credibility Analyzer. 
                        This system uses a multi-agentic pipeline to ingest social media posts, analyze 
                        textual and visual content for manipulation, and perform reasoning to detect misinformation.
                      </p>
                      <div className="mt-6 flex gap-4">
                        <button onClick={() => setActivePhase('dataset')} className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-2">
                          <Database className="w-4 h-4" /> Select Input Data
                        </button>
                      </div>
                    </div>

                    {/* Goals Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                       {projectData.project_metadata.design_goals.map((goal, idx) => (
                         <div key={idx} className="p-4 bg-slate-900/40 border border-slate-800/60 rounded-xl hover:border-indigo-500/30 transition-colors">
                           <div className="w-8 h-8 rounded-lg bg-indigo-500/10 flex items-center justify-center text-indigo-400 mb-3">
                             {idx % 2 === 0 ? <Zap className="w-4 h-4" /> : <ShieldCheck className="w-4 h-4" />}
                           </div>
                           <h3 className="text-sm font-medium text-slate-200 capitalize">{goal.replace(/_/g, ' ')}</h3>
                         </div>
                       ))}
                    </div>
                  </motion.div>
                )}

                {/* View: Data Explorer */}
                {activePhase === 'dataset' && (
                  <motion.div
                    key="dataset"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="space-y-6"
                  >
                    {/* Header with Navigation */}
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        {viewMode === 'tweets' && (
                          <button
                            onClick={() => {
                              setViewMode('handles');
                              setSelectedHandle(null);
                            }}
                            className="p-2 rounded-lg border border-slate-800 hover:bg-slate-900 text-slate-400 hover:text-white transition-colors"
                            title="Back to handles"
                          >
                            <ArrowRight className="w-5 h-5 rotate-180" />
                          </button>
                        )}
                        <h2 className="text-xl font-bold text-white flex items-center gap-3">
                          <Database className="w-6 h-6 text-indigo-400" /> 
                          {viewMode === 'handles' ? 'Social Media Handles' : selectedHandle}
                        </h2>
                      </div>
                      <div className="flex items-center gap-3">
                        {!apiConnected && (
                          <span className="text-xs text-amber-500 flex items-center gap-1">
                            <AlertOctagon className="w-3 h-3" /> API Disconnected
                          </span>
                        )}
                        <span className="text-sm text-slate-500">
                          {viewMode === 'handles' 
                            ? `${Object.keys(groupedByHandle).length} handles, ${dataset.length} total posts`
                            : `${handlePosts.length} tweets`
                          }
                        </span>
                      </div>
                    </div>

                    {error && (
                      <div className="p-4 bg-amber-500/10 border border-amber-500/30 rounded-xl text-amber-400 text-sm">
                        {error}
                      </div>
                    )}

                    {/* Handles View */}
                    {viewMode === 'handles' && (
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {dataset.length === 0 ? (
                          <div className="col-span-full p-8 text-center text-slate-500">
                            {isLoading ? 'Loading dataset...' : 'No posts available. Make sure the API server is running.'}
                          </div>
                        ) : (
                          Object.values(groupedByHandle).map((group) => {
                            const totalLikes = group.posts.reduce((sum, p) => sum + p.likes, 0);
                            const totalRetweets = group.posts.reduce((sum, p) => sum + p.retweets, 0);
                            const avgLikes = Math.round(totalLikes / group.posts.length);
                            const hasImages = group.posts.some(p => p.image_path);
                            const accountInfo = getAccountInfo(group.posts[0]); // Get account info from first post
                            
                            return (
                              <motion.div
                                key={group.handle}
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                onClick={() => {
                                  setSelectedHandle(group.handle);
                                  setViewMode('tweets');
                                }}
                                className="p-6 rounded-xl border border-slate-800 bg-slate-900/50 hover:border-indigo-500/50 hover:bg-slate-900 cursor-pointer transition-all group"
                              >
                                <div className="flex items-start justify-between mb-4">
                                  <div className="flex items-center gap-3">
                                    <div className="w-12 h-12 rounded-full bg-indigo-500/20 flex items-center justify-center text-indigo-400 group-hover:bg-indigo-500/30 transition-colors">
                                      {group.account.verified ? (
                                        <ShieldCheck className="w-6 h-6" />
                                      ) : (
                                        <UserCheck className="w-6 h-6" />
                                      )}
                                    </div>
                                    <div className="flex-1 min-w-0">
                                      <h3 className="text-lg font-bold text-white flex items-center gap-2">
                                        {accountInfo.name || group.handle}
                                        {group.account.verified && (
                                          <span className="text-xs text-blue-400" title="Verified">✓</span>
                                        )}
                                      </h3>
                                      <p className="text-xs text-slate-400">
                                        {accountInfo.screen_name ? `@${accountInfo.screen_name}` : group.handle}
                                      </p>
                                      {accountInfo.description && (
                                        <p className="text-xs text-slate-500 mt-1 line-clamp-2">
                                          {accountInfo.description}
                                        </p>
                                      )}
                                      <p className="text-xs text-slate-600 font-mono mt-1">{group.account.account_id}</p>
                                    </div>
                                  </div>
                                </div>
                                
                                <div className="space-y-3">
                                  <div className="flex items-center justify-between text-sm">
                                    <span className="text-slate-400">Posts</span>
                                    <span className="text-white font-semibold">{group.posts.length}</span>
                                  </div>
                                  <div className="flex items-center justify-between text-sm">
                                    <span className="text-slate-400">Avg Likes</span>
                                    <span className="text-white font-semibold">{avgLikes.toLocaleString()}</span>
                                  </div>
                                  <div className="flex items-center justify-between text-sm">
                                    <span className="text-slate-400">Account Age</span>
                                    <span className="text-white font-semibold">{Math.round(group.account.account_age_days / 365)} years</span>
                                  </div>
                                  <div className="flex items-center gap-2 pt-2 border-t border-slate-800">
                                    {hasImages && (
                                      <span className="text-xs px-2 py-1 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 flex items-center gap-1">
                                        <ImageIcon className="w-3 h-3" /> Has Images
                                      </span>
                                    )}
                                    <span className="text-xs px-2 py-1 rounded bg-slate-800 text-slate-400">
                                      {group.account.historical_post_count.toLocaleString()} total posts
                                    </span>
                                  </div>
                                </div>
                                
                                <div className="mt-4 pt-4 border-t border-slate-800 flex items-center justify-between">
                                  <span className="text-xs text-slate-500">Click to view tweets</span>
                                  <ChevronRight className="w-4 h-4 text-slate-500 group-hover:text-indigo-400 group-hover:translate-x-1 transition-all" />
                                </div>
                              </motion.div>
                            );
                          })
                        )}
                      </div>
                    )}

                    {/* Tweets View */}
                    {viewMode === 'tweets' && (
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {handlePosts.length === 0 ? (
                          <div className="col-span-full p-8 text-center text-slate-500">
                            No tweets available for this handle.
                          </div>
                        ) : (
                          handlePosts.map((post) => {
                            const imageUrl = getImageUrl(post.image_path);
                            
                            return (
                              <motion.div
                                key={post.post_id}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                onClick={() => setSelectedPost(post)}
                                className={`rounded-xl border overflow-hidden cursor-pointer transition-all ${
                                  selectedPost?.post_id === post.post_id 
                                    ? 'bg-indigo-900/20 border-indigo-500 ring-2 ring-indigo-500/50 shadow-lg shadow-indigo-500/10' 
                                    : 'bg-slate-900/50 border-slate-800 hover:border-slate-700 hover:bg-slate-900'
                                }`}
                              >
                                {/* Tweet Header */}
                                <div className="p-4 border-b border-slate-800">
                                  <div className="flex items-start justify-between mb-2">
                                    <div className="flex items-center gap-2">
                                      <span className="text-xs font-mono text-slate-500">{post.post_id}</span>
                                      <span className="px-2 py-0.5 rounded text-[10px] uppercase font-bold bg-sky-500/10 text-sky-400 border border-sky-500/20">
                                        {post.platform}
                                      </span>
                                      {post.account.verified && (
                                        <span className="text-xs text-blue-400" title="Verified Account">✓</span>
                                      )}
                                    </div>
                                    <span className="text-xs text-slate-500">{new Date(post.timestamp).toLocaleDateString()}</span>
                                  </div>
                                </div>

                                {/* Tweet Content */}
                                <div className="p-4">
                                  <p className="text-slate-200 text-sm leading-relaxed mb-4 whitespace-pre-wrap">
                                    {post.text}
                                  </p>

                                  {/* Image Display - Enhanced */}
                                  {imageUrl && (
                                    <ImageDisplay imageUrl={imageUrl} imagePath={post.image_path} />
                                  )}
                                  
                                  {/* Show indicator if image exists but URL couldn't be generated */}
                                  {post.image_path && !imageUrl && (
                                    <div className="mb-4 p-3 rounded-lg border border-amber-500/30 bg-amber-500/10 flex items-center gap-2">
                                      <ImageIcon className="w-4 h-4 text-amber-400" />
                                      <span className="text-xs text-amber-400">Image available but cannot be loaded</span>
                                    </div>
                                  )}

                                  {/* URLs */}
                                  {post.urls.length > 0 && (
                                    <div className="mb-4 space-y-1">
                                      {post.urls.map((url, idx) => (
                                        <a
                                          key={idx}
                                          href={url}
                                          target="_blank"
                                          rel="noopener noreferrer"
                                          onClick={(e) => e.stopPropagation()}
                                          className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1 break-all"
                                        >
                                          <Globe className="w-3 h-3 flex-shrink-0" />
                                          {url}
                                        </a>
                                      ))}
                                    </div>
                                  )}

                                  {/* Hashtags */}
                                  {post.hashtags.length > 0 && (
                                    <div className="mb-4 flex flex-wrap gap-1">
                                      {post.hashtags.map((tag, idx) => (
                                        <span key={idx} className="text-xs px-2 py-1 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                                          #{tag}
                                        </span>
                                      ))}
                                    </div>
                                  )}

                                  {/* Engagement Metrics */}
                                  <div className="flex items-center gap-6 text-xs text-slate-500 pt-3 border-t border-slate-800">
                                    <span className="flex items-center gap-1">
                                      <Zap className="w-3 h-3" /> {post.likes.toLocaleString()} likes
                                    </span>
                                    <span className="flex items-center gap-1">
                                      <ArrowRight className="w-3 h-3" /> {post.retweets.toLocaleString()} retweets
                                    </span>
                                    {post.image_path && (
                                      <span className="flex items-center gap-1 text-indigo-400">
                                        <ImageIcon className="w-3 h-3" /> Image
                                      </span>
                                    )}
                                  </div>
                                </div>

                                {/* Action Button */}
                                {selectedPost?.post_id === post.post_id && (
                                  <div className="p-4 border-t border-indigo-500/20 bg-indigo-500/5 flex justify-end">
                                    <button 
                                      onClick={(e) => { 
                                        e.stopPropagation(); 
                                        handleStartSimulation(); 
                                      }}
                                      className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
                                    >
                                      <Zap className="w-4 h-4" />
                                      Run Analysis
                                    </button>
                                  </div>
                                )}
                              </motion.div>
                            );
                          })
                        )}
                      </div>
                    )}
                  </motion.div>
                )}

                {/* View: Pipeline Details (Active during Simulation) */}
                {typeof activePhase === 'number' && activeData && (
                  <motion.div
                    key="phase-detail"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="bg-slate-900/50 border border-slate-800 rounded-2xl p-8 backdrop-blur-sm"
                  >
                    {/* Phase Header */}
                    <div className="flex items-start justify-between mb-8 border-b border-slate-800/50 pb-8">
                      <div className="flex items-center gap-5">
                        <div className="w-16 h-16 bg-gradient-to-br from-indigo-500/20 to-purple-500/20 rounded-2xl flex items-center justify-center text-indigo-400 ring-1 ring-white/10 shadow-2xl">
                          {renderIcon(activeData.iconComponent, "w-8 h-8")}
                        </div>
                        <div>
                          <h2 className="text-3xl font-bold text-white tracking-tight">{activeData.name}</h2>
                          <p className="text-slate-400 mt-2 max-w-2xl text-sm leading-relaxed">{activeData.purpose}</p>
                        </div>
                      </div>
                      <div className="text-right">
                         <div className="text-xs text-slate-500 uppercase tracking-widest mb-1">Processing Input</div>
                         <div className="font-mono text-indigo-400">{selectedPost?.post_id || 'N/A'}</div>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
                      <div className="space-y-6">
                        <div>
                          <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4 flex items-center gap-2">
                            <Cpu className="w-3.5 h-3.5" /> Processing Steps
                          </h4>
                          <div className="space-y-2">
                            {activeData.processing_steps.map((step, i) => (
                              <motion.div 
                                key={i} 
                                initial={{ opacity: 0, x: -10 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: i * 0.1 }}
                                className="flex items-center gap-3 text-sm text-slate-300 bg-slate-800/40 p-3 rounded-lg border border-slate-800/50 hover:bg-slate-800/60 transition-colors"
                              >
                                <div className="w-6 h-6 rounded bg-indigo-500/20 flex items-center justify-center text-indigo-400 text-xs font-mono">
                                  {i + 1}
                                </div>
                                <span className="capitalize">{step.replace(/_/g, ' ')}</span>
                              </motion.div>
                            ))}
                          </div>
                        </div>
                      </div>

                      <div className="bg-slate-950 rounded-xl border border-slate-800 overflow-hidden flex flex-col">
                        <div className="bg-slate-900/50 px-4 py-3 border-b border-slate-800 flex items-center justify-between">
                          <span className="text-xs font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
                            <Terminal className="w-3.5 h-3.5" /> Structured Output Schema
                          </span>
                        </div>
                        <div className="p-4 overflow-x-auto flex-1">
                          <pre className="text-xs font-mono leading-relaxed">
                            <JSONTree data={activeData.outputs} />
                          </pre>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}

                {/* View: Results Report */}
                {activePhase === 'results' && (
                  <motion.div
                    key="results"
                    initial={{ opacity: 0, scale: 0.98 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.98 }}
                    className="grid grid-cols-1 lg:grid-cols-3 gap-8"
                  >
                    {results ? (
                      <>
                        {/* Score Card */}
                        <div className="lg:col-span-1 space-y-6">
                          <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 text-center">
                             <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-4">Credibility Score</h3>
                             <div className="relative w-40 h-40 mx-auto mb-4 flex items-center justify-center">
                                <svg className="w-full h-full transform -rotate-90">
                                  <circle cx="80" cy="80" r="70" stroke="currentColor" strokeWidth="10" fill="transparent" className="text-slate-800" />
                                  <circle cx="80" cy="80" r="70" stroke="currentColor" strokeWidth="10" fill="transparent" className={results.credibility_score > 0.7 ? "text-emerald-500" : results.credibility_score > 0.4 ? "text-amber-500" : "text-red-500"} strokeDasharray={440} strokeDashoffset={440 - (440 * results.credibility_score)} />
                                </svg>
                                <span className="absolute text-4xl font-bold text-white">{(results.credibility_score * 100).toFixed(0)}%</span>
                             </div>
                             <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-bold border ${results.credibility_score > 0.7 ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" : results.credibility_score > 0.4 ? "bg-amber-500/10 text-amber-400 border-amber-500/20" : "bg-red-500/10 text-red-400 border-red-500/20"}`}>
                               {results.warning_label}
                             </div>
                          </div>

                          <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
                             <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-4">Input Context</h3>
                             <p className="text-sm text-slate-300 italic mb-4">"{selectedPost?.text}"</p>
                             <div className="space-y-2 text-xs text-slate-500">
                               <div className="flex justify-between"><span>Source:</span> <span className="text-slate-300">{selectedPost?.platform}</span></div>
                               <div className="flex justify-between"><span>Date:</span> <span className="text-slate-300">{selectedPost?.timestamp}</span></div>
                               <div className="flex justify-between"><span>Likes:</span> <span className="text-slate-300">{selectedPost?.likes}</span></div>
                               {analysisResult && (
                                 <>
                                   <div className="flex justify-between"><span>Sentiment:</span> <span className="text-slate-300">{analysisResult.nlp_signals.sentiment}</span></div>
                                   <div className="flex justify-between"><span>Emotion:</span> <span className="text-slate-300">{analysisResult.nlp_signals.emotion}</span></div>
                                   <div className="flex justify-between"><span>Clickbait:</span> <span className="text-slate-300">{analysisResult.nlp_signals.clickbait ? 'Yes' : 'No'}</span></div>
                                 </>
                               )}
                             </div>
                          </div>
                        </div>

                        {/* Detailed Analysis */}
                        <div className="lg:col-span-2 space-y-6">
                          <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-8">
                            <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                              <BrainCircuit className="w-6 h-6 text-indigo-400" /> Agentic Reasoning Trace
                            </h2>
                            
                            <div className="space-y-4">
                               {results.explanation.map((exp: string, i: number) => (
                                 <div key={i} className="flex gap-4 p-4 rounded-xl bg-slate-950/50 border border-slate-800/50">
                                   <div className="w-8 h-8 rounded-full bg-indigo-500/20 flex-shrink-0 flex items-center justify-center text-indigo-400 text-xs font-bold">
                                     {i+1}
                                   </div>
                                   <div>
                                     <p className="text-slate-300 text-sm leading-relaxed">{exp}</p>
                                   </div>
                                 </div>
                               ))}
                            </div>
                          </div>

                          {analysisResult && (
                            <div className="grid grid-cols-2 gap-4">
                              <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/30">
                                <h4 className="text-xs font-bold text-slate-500 uppercase mb-2">NLP Sentiment</h4>
                                <div className="text-lg font-mono text-slate-200">{analysisResult.nlp_signals.sentiment}</div>
                                <div className="text-xs text-slate-500 mt-1">Emotion: {analysisResult.nlp_signals.emotion}</div>
                              </div>
                              <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/30">
                                <h4 className="text-xs font-bold text-slate-500 uppercase mb-2">Source Trust</h4>
                                <div className="text-lg font-mono text-slate-200">{(analysisResult.source_signals.account_trust_score * 100).toFixed(0)}%</div>
                                <div className="text-xs text-slate-500 mt-1">Risk: {analysisResult.source_signals.behavioral_risk_flag ? 'High' : 'Low'}</div>
                              </div>
                            </div>
                          )}
                        </div>
                      </>
                    ) : (
                      <div className="col-span-3 p-8 text-center text-slate-500">
                        {isLoading ? 'Analyzing post...' : 'No results available. Run an analysis first.'}
                      </div>
                    )}
                  </motion.div>
                )}

              </AnimatePresence>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
