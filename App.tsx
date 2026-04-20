import React, { useState, useRef, useCallback, useEffect } from 'react';
import { GoogleGenAI, LiveSession, LiveServerMessage, Modality, Blob, Type } from '@google/genai';
import { Speaker, TranscriptEntry, SessionStatus, AppMode } from './types';
import { encode, decode, decodeAudioData } from './utils/audio';
import { MicIcon, StopIcon, LoadingSpinner, SparklesIcon, PlayIcon, DocumentCheckIcon, UploadIcon } from './components/Icons';

// --- Constantes de configuración ---
const INPUT_SAMPLE_RATE = 16000;
const OUTPUT_SAMPLE_RATE = 24000;

const LISTENING_TOPICS = [
    'recent breakthroughs in AI', 'the ethical implications of gene editing', 'exploring life on Mars',
    'impact of social media', 'renewable energy debate', 'future of self-driving cars',
    'how quantum computing works', 'film and TV reviews', 'history of Jazz', 'famous paintings',
    'traveling in Southeast Asia', 'managing expatriate life', 'minimalism and simple living',
    'public speaking techniques', 'learning styles', 'stress management', 'benefits of bilingualism',
];

// MEJORA: Instrucción reforzada para evitar interrupciones (Paciencia de 3 segundos)
const CONVERSATION_TUTOR_SYSTEM_INSTRUCTION = `You are a friendly, concise English language tutor. The user is a {LEVEL} level English learner.
Topic: "{TOPIC}".
Goal: Act like a real tutor, be natural, and keep responses brief (1-3 sentences).
EXTREMELY IMPORTANT: The user will pause frequently to think. You MUST wait for at least 3 seconds of silence before you reply. Do not interrupt.
Response Structure:
1. Correction (If needed): "Correction: [Corrected sentence]".
2. Separator: "||".
3. Conversational Content: Short response (1-3 sentences).`;

const LISTENING_SYSTEM_INSTRUCTION = `Generate an engaging listening comprehension exercise for level {LEVEL}.
1. Create a dialogue (250-400 words) with speakers [Alice]: and [Bob]:.
2. Provide 3-4 multiple-choice questions.`;

const WRITING_CORRECTOR_SYSTEM_INSTRUCTION = `You are an expert English examiner for Cambridge Assessment. Correct the user's text based on level {LEVEL}.
Output JSON: { "score": "1-5", "summary": "...", "corrections": [{"original": "...", "improved": "...", "explanation": "..."}], "improvedVersion": "..." }`;

// Herramientas auxiliares
function createPcmBlob(data: Float32Array): Blob {
    const int16 = new Int16Array(data.length);
    for (let i = 0; i < data.length; i++) int16[i] = data[i] * 32768;
    return { data: encode(new Uint8Array(int16.buffer)), mimeType: `audio/pcm;rate=${INPUT_SAMPLE_RATE}` };
}

// MEJORA: Compresor de imágenes para que los pantallazos no saturen la red
async function compressImage(file: File): Promise<{data: string, type: string}> {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = (e) => {
            const img = new Image();
            img.src = e.target?.result as string;
            img.onload = () => {
                const canvas = document.createElement('canvas');
                const MAX_SIZE = 1200;
                let w = img.width, h = img.height;
                if (w > h && w > MAX_SIZE) { h *= MAX_SIZE / w; w = MAX_SIZE; }
                else if (h > MAX_SIZE) { w *= MAX_SIZE / h; h = MAX_SIZE; }
                canvas.width = w; canvas.height = h;
                canvas.getContext('2d')?.drawImage(img, 0, 0, w, h);
                resolve({ data: canvas.toDataURL('image/jpeg', 0.8).split(',')[1], type: 'image/jpeg' });
            };
        };
    });
}

// Selector de nivel (Tu diseño original)
const LevelSelector: React.FC<{ level: string; setLevel: (level: string) => void; disabled: boolean; }> = ({ level, setLevel, disabled }) => (
    <div className="flex items-center gap-2">
        <span className="text-slate-400 font-medium">Target:</span>
        <select value={level} onChange={(e) => setLevel(e.target.value)} disabled={disabled} className="bg-slate-700 text-slate-100 rounded-lg px-3 py-1 border border-slate-600 focus:ring-2 focus:ring-cyan-500 focus:outline-none">
            <option value="A2">A2</option><option value="B1">B1</option><option value="B2">B2</option><option value="C1">C1</option><option value="C2">C2</option>
        </select>
    </div>
);

export default function App() {
    const [mode, setMode] = useState<AppMode>(AppMode.CONVERSATION);
    const [topic, setTopic] = useState('');
    const [level, setLevel] = useState('B1');
    const [status, setStatus] = useState<SessionStatus>(SessionStatus.INACTIVE);
    const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
    const [isTutorReplying, setIsTutorReplying] = useState(false);
    const [liveUserTranscript, setLiveUserTranscript] = useState('');

    const [exercise, setExercise] = useState<any>(null);
    const [isGenerating, setIsGenerating] = useState(false);
    const [listeningAudioBuffer, setListeningAudioBuffer] = useState<AudioBuffer | null>(null);
    const [isListeningAudioPlaying, setIsListeningAudioPlaying] = useState(false);

    const [writingInput, setWritingInput] = useState('');
    const [writingResult, setWritingResult] = useState<any>(null);
    const [isCorrecting, setIsCorrecting] = useState(false);
    const [uploadedFile, setUploadedFile] = useState<any>(null);

    const sessionPromiseRef = useRef<Promise<LiveSession> | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const audioWorkletNodeRef = useRef<AudioWorkletNode | null>(null);
    const audioContextsRef = useRef<{ input?: AudioContext; output?: AudioContext }>({});
    const audioPlaybackQueueRef = useRef<{ nextStartTime: number, sources: Set<AudioBufferSourceNode> }>({ nextStartTime: 0, sources: new Set() });
    const listeningPlaybackRef = useRef<{ source: AudioBufferSourceNode | null, startTime: number, pausedAt: number }>({ source: null, startTime: 0, pausedAt: 0 });
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const getOutputAudioContext = useCallback(() => {
        let context = audioContextsRef.current.output;
        if (!context || context.state === 'closed') {
            context = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: OUTPUT_SAMPLE_RATE });
            audioContextsRef.current.output = context;
        }
        return context;
    }, []);

    const cleanupSessionResources = useCallback(() => {
        streamRef.current?.getTracks().forEach(track => track.stop());
        streamRef.current = null;
        audioWorkletNodeRef.current?.disconnect();
        audioWorkletNodeRef.current = null;
        audioPlaybackQueueRef.current.sources.forEach(s => { try { s.stop(); } catch (e) {} });
        audioPlaybackQueueRef.current.sources.clear();
        audioPlaybackQueueRef.current.nextStartTime = 0;
        if (audioContextsRef.current.input) { audioContextsRef.current.input.close(); delete audioContextsRef.current.input; }
    }, []);

    const stopSession = useCallback(async () => {
        if (sessionPromiseRef.current) { try { (await sessionPromiseRef.current).close(); } catch (e) {} sessionPromiseRef.current = null; }
        cleanupSessionResources();
        setIsTutorReplying(false); setLiveUserTranscript(''); setStatus(SessionStatus.INACTIVE);
    }, [cleanupSessionResources]);

    const startSession = useCallback(async () => {
        const apiKey = import.meta.env.VITE_GEMINI_API_KEY;
        if (!topic.trim() || !apiKey) return;
        setTranscript([]); setStatus(SessionStatus.CONNECTING);
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            streamRef.current = stream;
            const ai = new GoogleGenAI({ apiKey });
            const outCtx = getOutputAudioContext();
            const inCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: INPUT_SAMPLE_RATE });
            await inCtx.audioWorklet.addModule('/audio-processor.js');
            audioContextsRef.current.input = inCtx;

            sessionPromiseRef.current = ai.live.connect({
                model: 'gemini-2.5-flash-native-audio-preview-09-2025',
                config: {
                    responseModalities: [Modality.AUDIO],
                    inputAudioTranscription: {}, outputAudioTranscription: {},
                    speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } } },
                    systemInstruction: CONVERSATION_TUTOR_SYSTEM_INSTRUCTION.replace('{TOPIC}', topic).replace('{LEVEL}', level),
                },
                callbacks: {
                    onopen: () => {
                        setStatus(SessionStatus.ACTIVE);
                        const source = inCtx.createMediaStreamSource(stream);
                        const worklet = new AudioWorkletNode(inCtx, 'audio-processor');
                        worklet.port.onmessage = (e) => sessionPromiseRef.current?.then(s => s.sendRealtimeInput({ media: createPcmBlob(e.data) }));
                        source.connect(worklet); worklet.connect(inCtx.destination);
                        audioWorkletNodeRef.current = worklet;
                    },
                    onmessage: async (msg: LiveServerMessage) => {
                        if (msg.serverContent?.inputTranscription) setLiveUserTranscript(prev => prev + msg.serverContent!.inputTranscription!.text);
                        if (msg.serverContent?.turnComplete) {
                            setTranscript(prev => [...prev, { speaker: Speaker.USER, text: liveUserTranscript }]);
                            setLiveUserTranscript('');
                        }
                        const b64 = msg.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
                        if (b64) {
                            const start = Math.max(audioPlaybackQueueRef.current.nextStartTime, outCtx.currentTime);
                            const buffer = await decodeAudioData(decode(b64), outCtx, OUTPUT_SAMPLE_RATE, 1);
                            const node = outCtx.createBufferSource();
                            node.buffer = buffer; node.connect(outCtx.destination);
                            node.onended = () => audioPlaybackQueueRef.current.sources.delete(node);
                            node.start(start);
                            audioPlaybackQueueRef.current.nextStartTime = start + buffer.duration;
                            audioPlaybackQueueRef.current.sources.add(node);
                        }
                    },
                    onerror: stopSession,
                },
            });
        } catch (e) { stopSession(); }
    }, [topic, level, stopSession, getOutputAudioContext, liveUserTranscript]);

    // MEJORA: Manejo de pegado de imágenes (Capturas de pantalla)
    const handlePaste = async (e: React.ClipboardEvent) => {
        const item = e.clipboardData.items[0];
        if (item?.type.includes('image')) {
            const file = item.getAsFile();
            if (file) {
                e.preventDefault();
                setUploadedFile(await compressImage(file));
            }
        }
    };

    const runWritingCorrection = async () => {
        const apiKey = import.meta.env.VITE_GEMINI_API_KEY;
        if (!apiKey || (!writingInput.trim() && !uploadedFile)) return;
        setIsCorrecting(true); setWritingResult(null);
        try {
            const ai = new GoogleGenAI({ apiKey });
            const parts: any[] = [{ text: `Correct this text at level ${level}` }];
            if (writingInput) parts.push({ text: writingInput });
            if (uploadedFile) parts.push({ inlineData: { data: uploadedFile.data, mimeType: uploadedFile.type } });

            const res = await ai.models.generateContent({
                model: 'gemini-2.0-flash',
                contents: [{ role: 'user', parts }],
                config: { systemInstruction: WRITING_CORRECTOR_SYSTEM_INSTRUCTION.replace('{LEVEL}', level), responseMimeType: "application/json" }
            });
            setWritingResult(JSON.parse(res.text.replace(/```json|```/g, '').trim()));
        } catch (e) { alert("Error analizando escritura."); } finally { setIsCorrecting(false); }
    };

    const generateListeningExercise = async () => {
        const apiKey = import.meta.env.VITE_GEMINI_API_KEY;
        if (!apiKey) return;
        setIsGenerating(true); setExercise(null); setListeningAudioBuffer(null);
        listeningPlaybackRef.current = { source: null, startTime: 0, pausedAt: 0 };
        setIsListeningAudioPlaying(false);
        try {
            const ai = new GoogleGenAI({ apiKey });
            const topic = LISTENING_TOPICS[Math.floor(Math.random() * LISTENING_TOPICS.length)];
            const res = await ai.models.generateContent({
                model: 'gemini-2.0-flash',
                contents: `Create exercise level ${level} about ${topic}`,
                config: { systemInstruction: LISTENING_SYSTEM_INSTRUCTION.replace('{LEVEL}', level), responseMimeType: "application/json" }
            });
            const json = JSON.parse(res.text.replace(/```json|```/g, '').trim());
            setExercise(json);
            const tts = await ai.models.generateContent({
                model: "gemini-2.5-flash-preview-tts",
                contents: [{ parts: [{ text: json.transcript }] }],
                config: { responseModalities: [Modality.AUDIO] }
            });
            const b64 = tts.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
            if (b64) setListeningAudioBuffer(await decodeAudioData(decode(b64), getOutputAudioContext(), OUTPUT_SAMPLE_RATE, 1));
        } catch (e) { alert("Error al generar listening."); } finally { setIsGenerating(false); }
    };

    return (
        <div className="h-screen w-screen bg-slate-950 text-slate-100 flex flex-col font-sans overflow-hidden">
            {/* Cabecera (Tu diseño original) */}
            <header className="p-4 border-b border-slate-800 flex justify-between items-center bg-slate-900/50">
                <h1 className="text-2xl font-black tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-500 to-indigo-500">
                    ENGLISH MASTERY AI
                </h1>
                <div className="flex bg-slate-800 rounded-full p-1 border border-slate-700">
                    {[AppMode.CONVERSATION, AppMode.LISTENING, AppMode.WRITING].map(m => (
                        <button key={m} onClick={() => setMode(m)} className={`px-4 py-1.5 rounded-full text-sm font-bold transition-all ${mode === m ? 'bg-cyan-600 shadow-lg' : 'text-slate-400 hover:text-white'}`}>
                            {m}
                        </button>
                    ))}
                </div>
            </header>

            <main className="flex-grow flex flex-col p-4 gap-4 overflow-hidden max-w-6xl mx-auto w-full">
                {mode === AppMode.CONVERSATION && (
                    <div className="flex-grow flex flex-col overflow-hidden gap-4">
                        <div className="flex-grow bg-slate-900/50 rounded-3xl border border-slate-800 p-6 overflow-y-auto space-y-4">
                            {transcript.map((e, i) => (
                                <div key={i} className={`flex ${e.speaker === Speaker.USER ? 'justify-end' : 'justify-start'}`}>
                                    <div className={`max-w-[85%] p-4 rounded-2xl shadow-sm ${e.speaker === Speaker.USER ? 'bg-indigo-600 text-white rounded-br-none' : 'bg-slate-800 border border-slate-700 rounded-bl-none'}`}>
                                        <p className="leading-relaxed">{e.text}</p>
                                    </div>
                                </div>
                            ))}
                            {liveUserTranscript && <div className="text-right italic text-slate-500">{liveUserTranscript}</div>}
                            <div ref={messagesEndRef} />
                        </div>
                        <div className="bg-slate-900 border border-slate-800 p-4 rounded-3xl shadow-2xl flex flex-col md:flex-row items-center gap-4">
                            <input placeholder="Topic..." className="flex-grow bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-cyan-500" value={topic} onChange={e => setTopic(e.target.value)} />
                            <LevelSelector level={level} setLevel={setLevel} disabled={status !== SessionStatus.INACTIVE} />
                            <button onClick={status === SessionStatus.INACTIVE ? startSession : stopSession} className={`px-8 py-3 rounded-xl font-bold flex items-center gap-2 transition-all ${status === SessionStatus.INACTIVE ? 'bg-cyan-600' : 'bg-rose-600'}`}>
                                {status === SessionStatus.INACTIVE ? <MicIcon className="w-5 h-5" /> : <StopIcon className="w-5 h-5" />} {status === SessionStatus.INACTIVE ? 'Start' : 'Stop'}
                            </button>
                        </div>
                    </div>
                )}

                {mode === AppMode.WRITING && (
                    <div className="flex-grow flex flex-col md:flex-row gap-4 overflow-hidden">
                        <div className="flex-1 flex flex-col gap-4">
                            <div className="flex-grow bg-slate-900/50 rounded-3xl border border-slate-800 p-6 flex flex-col">
                                <h2 className="text-xl font-bold mb-4 flex items-center gap-2"><DocumentCheckIcon className="w-6 h-6 text-cyan-400" /> Writing Workspace</h2>
                                <textarea className="flex-grow bg-transparent border-none outline-none resize-none text-lg text-slate-300 placeholder:text-slate-600 leading-relaxed" placeholder="Paste your essay or screenshot here..." value={writingInput} onChange={e => setWritingInput(e.target.value)} onPaste={handlePaste} />
                                <div className="mt-4 flex items-center gap-4 pt-4 border-t border-slate-800">
                                    <label className="flex items-center gap-2 text-sm text-slate-400 cursor-pointer hover:text-cyan-400 transition-colors">
                                        <UploadIcon className="w-5 h-5" />
                                        <span>{uploadedFile ? '✓ Attached' : 'Upload'}</span>
                                        <input type="file" className="hidden" accept="image/*,.pdf" onChange={async e => setUploadedFile(await compressImage(e.target.files![0]))} />
                                    </label>
                                    <div className="flex-grow" />
                                    <LevelSelector level={level} setLevel={setLevel} disabled={isCorrecting} />
                                    <button onClick={runWritingCorrection} disabled={isCorrecting} className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 px-6 py-2 rounded-xl font-bold flex items-center gap-2 transition-all">
                                        {isCorrecting ? <LoadingSpinner className="w-5 h-5" /> : <SparklesIcon className="w-5 h-5" />} Analyze
                                    </button>
                                </div>
                            </div>
                        </div>
                        <div className="flex-1 bg-slate-900/50 rounded-3xl border border-slate-800 p-6 overflow-y-auto">
                            {writingResult && (
                                <div className="space-y-6">
                                    <div className="flex justify-between items-center bg-slate-800 p-4 rounded-2xl border border-slate-700">
                                        <div className="text-sm font-bold uppercase text-slate-400">Score</div>
                                        <div className="text-3xl font-black text-cyan-400">{writingResult.score}/5</div>
                                    </div>
                                    <div>
                                        <h3 className="font-bold text-slate-200 mb-2">Summary</h3>
                                        <p className="text-slate-400 text-sm leading-relaxed">{writingResult.summary}</p>
                                    </div>
                                    <div className="space-y-3">
                                        {writingResult.corrections.map((c: any, i: number) => (
                                            <div key={i} className="bg-slate-800/50 p-3 rounded-xl border-l-4 border-yellow-500">
                                                <p className="text-xs text-rose-400 line-through mb-1">{c.original}</p>
                                                <p className="text-sm text-emerald-400 font-medium">{c.improved}</p>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {mode === AppMode.LISTENING && (
                    <div className="flex-grow flex flex-col bg-slate-900/50 rounded-3xl border border-slate-800 p-8 overflow-y-auto">
                        {!exercise && !isGenerating && (
                            <div className="m-auto text-center space-y-4 max-w-md">
                                <PlayIcon className="w-20 h-20 mx-auto text-cyan-500/20" />
                                <h2 className="text-2xl font-bold">Listening Comprehension</h2>
                                <p className="text-slate-500">Generate professional dialogues and practice.</p>
                                <div className="flex items-center justify-center gap-4 py-4">
                                    <LevelSelector level={level} setLevel={setLevel} disabled={isGenerating} />
                                    <button onClick={generateListeningExercise} className="bg-purple-600 px-6 py-2 rounded-xl font-bold">Generate</button>
                                </div>
                            </div>
                        )}
                        {isGenerating && <div className="m-auto text-center animate-pulse"><LoadingSpinner className="w-12 h-12 mx-auto mb-4" /> Generating...</div>}
                        {exercise && (
                            <div className="space-y-8 max-w-2xl mx-auto w-full">
                                <div className="bg-slate-800 p-6 rounded-3xl border border-slate-700 flex items-center justify-between">
                                    <h3 className="text-lg font-bold">Exercise Audio</h3>
                                    <button onClick={async () => {
                                        const ctx = getOutputAudioContext();
                                        if (isListeningAudioPlaying) {
                                            listeningPlaybackRef.current.source?.stop();
                                            setIsListeningAudioPlaying(false);
                                        } else if (listeningAudioBuffer) {
                                            if (ctx.state === 'suspended') await ctx.resume();
                                            const s = ctx.createBufferSource();
                                            s.buffer = listeningAudioBuffer; s.connect(ctx.destination);
                                            s.onended = () => setIsListeningAudioPlaying(false);
                                            s.start(); listeningPlaybackRef.current.source = s;
                                            setIsListeningAudioPlaying(true);
                                        }
                                    }} disabled={!listeningAudioBuffer} className="bg-cyan-600 p-4 rounded-full transition-transform">
                                        {isListeningAudioPlaying ? <StopIcon className="w-8 h-8" /> : <PlayIcon className="w-8 h-8" />}
                                    </button>
                                </div>
                                <div className="bg-slate-800/50 p-6 rounded-3xl border border-slate-800 whitespace-pre-wrap">{exercise.questions}</div>
                            </div>
                        )}
                    </div>
                )}
            </main>
        </div>
    );
}
