"use client";

import React, { useState, useRef } from "react";
import { 
  UploadCloud, 
  FileImage, 
  CheckCircle, 
  AlertTriangle, 
  AlertCircle, 
  RefreshCw, 
  XCircle, 
  Sparkles,
  Search,
  ScanLine,
  Crop,
  Sun,
  Image as ImageIcon
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";

type AppState = 'idle' | 'preview' | 'analyzing' | 'result' | 'error' | 'non_document';

interface ApiResponse {
  success: boolean;
  is_document: boolean;
  document_confidence: number;
  message?: string;
  quality?: string;
  quality_confidence?: number;
  ocr_suitable?: boolean;
  detected_issue?: string | null;
  recommendation: string;
}

const QUALITY_DISPLAY_MAP: Record<string, { label: string; icon: React.ReactNode; color: string; bg: string; border: string }> = {
  "clear": { 
    label: "CLEAR", 
    icon: <CheckCircle className="w-5 h-5" />, 
    color: "text-emerald-700", 
    bg: "bg-emerald-50",
    border: "border-emerald-200" 
  },
  "blurry": { 
    label: "BLURRY", 
    icon: <AlertCircle className="w-5 h-5" />, 
    color: "text-red-700", 
    bg: "bg-red-50",
    border: "border-red-200"
  },
  "low_light": { 
    label: "LOW-LIGHT", 
    icon: <Sun className="w-5 h-5" />, 
    color: "text-amber-700", 
    bg: "bg-amber-50",
    border: "border-amber-200"
  },
  "skewed": { 
    label: "SKEWED", 
    icon: <RefreshCw className="w-5 h-5" />, 
    color: "text-orange-700", 
    bg: "bg-orange-50",
    border: "border-orange-200"
  },
  "partially_cropped": { 
    label: "PARTIALLY CROPPED", 
    icon: <Crop className="w-5 h-5" />, 
    color: "text-rose-700", 
    bg: "bg-rose-50",
    border: "border-rose-200"
  },
  "glare": { 
    label: "GLARE", 
    icon: <Sparkles className="w-5 h-5" />, 
    color: "text-amber-600", 
    bg: "bg-amber-50",
    border: "border-amber-200"
  },
};

export default function DocumentQualityApp() {
  const [appState, setAppState] = useState<AppState>('idle');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const processFile = (file: File) => {
    if (!file.type.includes('image/')) {
      setErrorMessage('Please upload an image file (JPG, PNG).');
      setAppState('error');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setErrorMessage('File size exceeds 10 MB limit.');
      setAppState('error');
      return;
    }
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setAppState('preview');
  };

  const startAnalysis = async () => {
    if (!selectedFile) return;
    
    setAppState('analyzing');
    setProgress(0);
    
    const interval = setInterval(() => {
      setProgress(prev => (prev < 85 ? prev + 5 : prev));
    }, 150);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      
      const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
      
      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        body: formData,
      });
      
      clearInterval(interval);
      setProgress(100);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail || `Server error (${response.status})`);
      }
      
      const data: ApiResponse = await response.json();
      setResult(data);
      
      setTimeout(() => {
        if (!data.is_document) {
          setAppState('non_document');
        } else {
          setAppState('result');
        }
      }, 400); // Small delay for smooth progress bar finish
      
    } catch (err: any) {
      clearInterval(interval);
      setErrorMessage(err.message || "Failed to connect to the server.");
      setAppState('error');
    }
  };

  const resetApp = () => {
    setAppState('idle');
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setResult(null);
    setProgress(0);
    setErrorMessage("");
    setIsDragging(false);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans selection:bg-indigo-100 selection:text-indigo-900">
      {/* HEADER */}
      <header className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2 text-indigo-600 font-bold text-lg tracking-tight cursor-pointer" onClick={resetApp}>
            <ScanLine className="w-6 h-6" />
            <span>Smart Document Quality</span>
          </div>
          <nav className="hidden md:flex gap-6 text-sm font-medium text-slate-600">
            <a href="#how-it-works" className="hover:text-indigo-600 transition-colors">How it works</a>
            <a href="#quality-checks" className="hover:text-indigo-600 transition-colors">Quality Checks</a>
          </nav>
        </div>
      </header>

      {/* HERO SECTION */}
      <section className="pt-20 pb-12 px-6 text-center animate-in fade-in slide-in-from-bottom-4 duration-700">
        <div className="max-w-3xl mx-auto space-y-6">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-50 border border-indigo-100 text-indigo-700 text-xs font-semibold uppercase tracking-wider mx-auto">
            <Sparkles className="w-3.5 h-3.5" />
            AI-Powered Document Quality Analysis
          </div>
          <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight text-slate-900 leading-tight">
            Make Every Document <br className="hidden md:block"/> OCR-Ready.
          </h1>
          <p className="text-lg md:text-xl text-slate-600 max-w-2xl mx-auto leading-relaxed">
            Analyze document image quality before OCR processing. Detect blur, low light, skew, cropping, and glare in seconds.
          </p>
        </div>
      </section>

      {/* MAIN UPLOAD & RESULT AREA */}
      <section className="max-w-4xl mx-auto px-6 pb-24">
        <div className="bg-white rounded-3xl shadow-xl shadow-slate-200/50 border border-slate-200 overflow-hidden transition-all duration-300">
          
          {/* IDLE / PREVIEW STATE */}
          {(appState === 'idle' || appState === 'preview') && (
            <div className="p-8 md:p-12">
              <div 
                className={`relative border-2 border-dashed rounded-2xl p-10 text-center transition-all duration-300 ${
                  isDragging ? 'border-indigo-500 bg-indigo-50 scale-[1.01]' : 'border-slate-300 bg-slate-50 hover:bg-slate-100/50 hover:border-slate-400'
                }`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                {!selectedFile ? (
                  <div className="flex flex-col items-center justify-center space-y-4 py-8">
                    <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center shadow-sm border border-slate-200 text-indigo-600 mb-2">
                      <UploadCloud className="w-8 h-8" />
                    </div>
                    <div className="space-y-1">
                      <h3 className="text-xl font-semibold text-slate-800">Drop your document here</h3>
                      <p className="text-slate-500">or choose an image from your device</p>
                    </div>
                    <div className="pt-4 flex gap-4 text-xs font-medium text-slate-400 uppercase tracking-widest">
                      <span>JPG • JPEG • PNG</span>
                      <span>Maximum: 10 MB</span>
                    </div>
                    <Button 
                      onClick={() => fileInputRef.current?.click()} 
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    />
                  </div>
                ) : (
                  <div className="space-y-6 animate-in fade-in duration-500">
                    <div className="relative mx-auto w-full max-w-md aspect-[4/3] rounded-xl overflow-hidden border border-slate-200 shadow-sm bg-white flex items-center justify-center group">
                      <img src={previewUrl!} alt="Preview" className="w-full h-full object-contain" />
                      <div className="absolute inset-0 bg-slate-900/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center backdrop-blur-sm">
                        <Button variant="secondary" onClick={(e) => { e.stopPropagation(); resetApp(); }}>
                          Remove / Change
                        </Button>
                      </div>
                    </div>
                    <div className="flex items-center justify-center gap-4 text-sm text-slate-600 font-medium bg-white px-4 py-2 rounded-full border border-slate-200 shadow-sm mx-auto w-fit">
                      <ImageIcon className="w-4 h-4 text-slate-400" />
                      <span className="truncate max-w-[200px]">{selectedFile.name}</span>
                      <span className="text-slate-400">•</span>
                      <span>{formatFileSize(selectedFile.size)}</span>
                    </div>
                    <div className="pt-4">
                      <Button size="lg" className="w-full max-w-sm text-base h-14 rounded-xl shadow-lg shadow-indigo-200 bg-indigo-600 hover:bg-indigo-700 transition-all hover:-translate-y-1" onClick={startAnalysis}>
                        Analyze Document
                      </Button>
                    </div>
                  </div>
                )}
                
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  className="hidden" 
                  accept=".jpg,.jpeg,.png,image/jpeg,image/png" 
                  onChange={handleFileSelect} 
                />
              </div>
            </div>
          )}

          {/* ANALYZING STATE */}
          {appState === 'analyzing' && (
            <div className="p-16 text-center space-y-8 animate-in zoom-in-95 duration-300">
              <div className="mx-auto w-20 h-20 bg-indigo-50 rounded-full flex items-center justify-center text-indigo-600">
                <Search className="w-10 h-10 animate-pulse" />
              </div>
              <div className="space-y-2">
                <h3 className="text-2xl font-bold text-slate-800">Analyzing Document</h3>
                <p className="text-slate-500">Running advanced vision pipeline...</p>
              </div>
              <div className="max-w-md mx-auto">
                <Progress value={progress} className="h-3 rounded-full bg-slate-100" />
              </div>
            </div>
          )}

          {/* ERROR STATE */}
          {appState === 'error' && (
            <div className="p-16 text-center space-y-6 animate-in fade-in duration-300">
              <div className="mx-auto w-20 h-20 bg-red-50 rounded-full flex items-center justify-center text-red-500 border border-red-100">
                <AlertTriangle className="w-10 h-10" />
              </div>
              <div className="space-y-2">
                <h3 className="text-2xl font-bold text-slate-800">Analysis Failed</h3>
                <p className="text-slate-500 max-w-md mx-auto">{errorMessage}</p>
              </div>
              <Button size="lg" variant="outline" onClick={resetApp} className="mt-4">
                Try Again
              </Button>
            </div>
          )}

          {/* NON-DOCUMENT STATE */}
          {appState === 'non_document' && result && (
            <div className="p-8 md:p-12 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="text-center space-y-6 max-w-lg mx-auto">
                <div className="mx-auto w-24 h-24 bg-rose-50 rounded-full flex items-center justify-center text-rose-500 border border-rose-100 mb-6">
                  <XCircle className="w-12 h-12" />
                </div>
                <h2 className="text-3xl font-bold text-slate-900">Not a document</h2>
                <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 text-left space-y-4">
                  <p className="text-lg font-medium text-slate-800">{result.message}</p>
                  <div className="border-t border-slate-200 pt-4">
                    <p className="text-sm font-bold text-slate-500 uppercase tracking-wider mb-2">Recommendation</p>
                    <p className="text-slate-700">{result.recommendation}</p>
                  </div>
                </div>
                <Button size="lg" className="w-full h-14 rounded-xl text-base" onClick={resetApp}>
                  Analyze Another Document
                </Button>
              </div>
            </div>
          )}

          {/* RESULT STATE */}
          {appState === 'result' && result && (
            <div className="animate-in fade-in duration-500 flex flex-col md:flex-row">
              {/* Left Column: Image */}
              <div className="w-full md:w-5/12 bg-slate-50 border-b md:border-b-0 md:border-r border-slate-200 p-8 flex flex-col justify-center items-center relative">
                 <div className="absolute top-4 left-4 bg-white/80 backdrop-blur text-xs font-semibold px-3 py-1.5 rounded-md border border-slate-200 text-slate-600 shadow-sm z-10 flex items-center gap-2">
                    <FileImage className="w-3.5 h-3.5" /> Original Image
                 </div>
                 <div className="w-full aspect-[3/4] relative rounded-xl overflow-hidden shadow-md bg-white border border-slate-200">
                    <img src={previewUrl!} alt="Analyzed Document" className="w-full h-full object-cover md:object-contain" />
                 </div>
              </div>
              
              {/* Right Column: Data */}
              <div className="w-full md:w-7/12 p-8 md:p-10 flex flex-col justify-between">
                <div className="space-y-8">
                  {/* Suitability Banner */}
                  <div className={`p-5 rounded-2xl border flex items-start gap-4 ${result.ocr_suitable ? 'bg-emerald-50 border-emerald-200 text-emerald-900' : 'bg-amber-50 border-amber-200 text-amber-900'}`}>
                    <div className="mt-0.5">
                      {result.ocr_suitable ? <CheckCircle className="w-6 h-6 text-emerald-600" /> : <AlertTriangle className="w-6 h-6 text-amber-600" />}
                    </div>
                    <div>
                      <h3 className="font-bold text-lg mb-1">
                        {result.ocr_suitable ? 'Suitable for further OCR processing' : 'Not suitable for further OCR processing'}
                      </h3>
                      <p className={`text-sm ${result.ocr_suitable ? 'text-emerald-700' : 'text-amber-800'}`}>
                        {result.recommendation}
                      </p>
                    </div>
                  </div>

                  {/* Quality Metrics */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className={`p-6 rounded-2xl border ${QUALITY_DISPLAY_MAP[result.quality || "clear"]?.bg} ${QUALITY_DISPLAY_MAP[result.quality || "clear"]?.border}`}>
                      <p className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">Quality Class</p>
                      <div className={`flex items-center gap-2 font-bold text-xl ${QUALITY_DISPLAY_MAP[result.quality || "clear"]?.color}`}>
                        {QUALITY_DISPLAY_MAP[result.quality || "clear"]?.icon}
                        {QUALITY_DISPLAY_MAP[result.quality || "clear"]?.label}
                      </div>
                    </div>
                    
                    <div className="p-6 rounded-2xl border border-slate-200 bg-white">
                      <p className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">Confidence</p>
                      <div className="font-bold text-2xl text-slate-800">
                        {((result.quality_confidence || 0) * 100).toFixed(1)}<span className="text-lg text-slate-400">%</span>
                      </div>
                    </div>
                  </div>

                  {/* Detected Issue (if any) */}
                  {!result.ocr_suitable && result.detected_issue && (
                    <div className="p-6 rounded-2xl border border-slate-200 bg-white flex items-center justify-between">
                      <p className="text-sm font-bold uppercase tracking-wider text-slate-500">Detected Issue</p>
                      <Badge variant="outline" className="text-sm font-semibold uppercase px-3 py-1 bg-slate-50">
                        {result.detected_issue}
                      </Badge>
                    </div>
                  )}
                </div>

                <div className="mt-12 flex justify-end">
                  <Button onClick={resetApp} size="lg" className="h-12 px-6 rounded-xl bg-slate-900 hover:bg-slate-800 transition-all">
                    Analyze Another Document
                  </Button>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* HOW IT WORKS */}
      <section id="how-it-works" className="py-24 bg-white border-t border-slate-200">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-slate-900 mb-4">How it works</h2>
            <p className="text-slate-500 text-lg">A simple three-step workflow to ensure data quality.</p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8 md:gap-12 relative">
            <div className="hidden md:block absolute top-12 left-[15%] right-[15%] h-0.5 bg-slate-100 -z-10"></div>
            
            <div className="bg-white p-8 rounded-3xl border border-slate-100 shadow-sm text-center relative hover:-translate-y-2 transition-transform duration-300">
              <div className="w-16 h-16 bg-indigo-50 text-indigo-600 rounded-2xl flex items-center justify-center text-xl font-black mx-auto mb-6 shadow-inner">01</div>
              <h3 className="text-xl font-bold text-slate-900 mb-3 uppercase tracking-wide">Upload</h3>
              <p className="text-slate-600 leading-relaxed">Upload a scanned or photographed document securely from your device.</p>
            </div>
            
            <div className="bg-white p-8 rounded-3xl border border-slate-100 shadow-sm text-center relative hover:-translate-y-2 transition-transform duration-300">
              <div className="w-16 h-16 bg-indigo-50 text-indigo-600 rounded-2xl flex items-center justify-center text-xl font-black mx-auto mb-6 shadow-inner">02</div>
              <h3 className="text-xl font-bold text-slate-900 mb-3 uppercase tracking-wide">Analyze</h3>
              <p className="text-slate-600 leading-relaxed">Our vision pipeline validates the image and classifies its visual quality.</p>
            </div>
            
            <div className="bg-white p-8 rounded-3xl border border-slate-100 shadow-sm text-center relative hover:-translate-y-2 transition-transform duration-300">
              <div className="w-16 h-16 bg-indigo-50 text-indigo-600 rounded-2xl flex items-center justify-center text-xl font-black mx-auto mb-6 shadow-inner">03</div>
              <h3 className="text-xl font-bold text-slate-900 mb-3 uppercase tracking-wide">Check</h3>
              <p className="text-slate-600 leading-relaxed">Get an instant quality result and an automated OCR suitability recommendation.</p>
            </div>
          </div>
        </div>
      </section>

      {/* QUALITY CHECKS */}
      <section id="quality-checks" className="py-24 bg-slate-50">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-slate-900 mb-4">Six quality checks</h2>
            <p className="text-slate-500 text-lg">We classify documents into discrete visual defect categories.</p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Clear */}
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex items-start gap-4 hover:shadow-md transition-shadow">
              <div className="p-3 bg-emerald-50 text-emerald-600 rounded-xl">
                <CheckCircle className="w-6 h-6" />
              </div>
              <div>
                <h4 className="font-bold text-slate-900 mb-1">Clear</h4>
                <p className="text-sm text-slate-500">Ready for OCR</p>
              </div>
            </div>
            
            {/* Blurry */}
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex items-start gap-4 hover:shadow-md transition-shadow">
              <div className="p-3 bg-red-50 text-red-600 rounded-xl">
                <AlertCircle className="w-6 h-6" />
              </div>
              <div>
                <h4 className="font-bold text-slate-900 mb-1">Blurry</h4>
                <p className="text-sm text-slate-500">Focus is insufficient</p>
              </div>
            </div>

            {/* Low Light */}
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex items-start gap-4 hover:shadow-md transition-shadow">
              <div className="p-3 bg-amber-50 text-amber-600 rounded-xl">
                <Sun className="w-6 h-6" />
              </div>
              <div>
                <h4 className="font-bold text-slate-900 mb-1">Low Light</h4>
                <p className="text-sm text-slate-500">Insufficient illumination</p>
              </div>
            </div>

            {/* Skewed */}
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex items-start gap-4 hover:shadow-md transition-shadow">
              <div className="p-3 bg-orange-50 text-orange-600 rounded-xl">
                <RefreshCw className="w-6 h-6" />
              </div>
              <div>
                <h4 className="font-bold text-slate-900 mb-1">Skewed</h4>
                <p className="text-sm text-slate-500">Document angle detected</p>
              </div>
            </div>

            {/* Partially Cropped */}
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex items-start gap-4 hover:shadow-md transition-shadow">
              <div className="p-3 bg-rose-50 text-rose-600 rounded-xl">
                <Crop className="w-6 h-6" />
              </div>
              <div>
                <h4 className="font-bold text-slate-900 mb-1">Partially Cropped</h4>
                <p className="text-sm text-slate-500">Document edges are missing</p>
              </div>
            </div>

            {/* Glare */}
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex items-start gap-4 hover:shadow-md transition-shadow">
              <div className="p-3 bg-amber-50 text-amber-500 rounded-xl">
                <Sparkles className="w-6 h-6" />
              </div>
              <div>
                <h4 className="font-bold text-slate-900 mb-1">Glare</h4>
                <p className="text-sm text-slate-500">Reflection affects the page</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="py-12 bg-white border-t border-slate-200">
        <div className="max-w-7xl mx-auto px-6 flex flex-col items-center justify-center text-center space-y-4">
          <div className="flex items-center gap-2 text-slate-900 font-bold text-xl">
            <ScanLine className="w-6 h-6 text-indigo-600" />
            <span>Smart Document Quality</span>
          </div>
          <p className="text-slate-500">AI-powered document image quality assessment.</p>
        </div>
      </footer>
    </div>
  );
}
