import React, { useState, useRef, useCallback, useEffect } from 'react';
import { GoogleGenAI, LiveSession, LiveServerMessage, Modality, Blob, Type } from '@google/genai';
import { Speaker, TranscriptEntry, SessionStatus, AppMode } from './types';
import { encode, decode, decodeAudioData } from './utils/audio';
import { MicIcon, StopIcon, LoadingSpinner, SparklesIcon, PlayIcon, DocumentCheckIcon, UploadIcon } from './components/Icons';

// --- Helper Functions & Constants ---
const INPUT_SAMPLE_RATE = 16000;
const OUTPUT_SAMPLE_RATE = 24000;
const SCRIPT_PROCESSOR_BUFFER_SIZE = 4096;

const LISTENING_TOPICS = [
    'recent breakthroughs in AI',
    'the ethical implications of gene editing',
    'exploring life on Mars',
    'impact of social media',
    'renewable energy debate',
    'future of self-driving cars',
    'how quantum computing works',
    'film and TV reviews',
    'history of Jazz',
    'famous paintings',
    'traveling in Southeast Asia',
    'managing expatriate life',
    'minimalism and simple living',
    'public speaking techniques',
    'learning styles',
    'stress management',
    'benefits of bilingualism',
];

const CONVERSATION_TUTOR_SYSTEM_INSTRUCTION = `You are a friendly, concise English language tutor. The user is a {LEVEL} level English learner.
Topic: "{TOPIC}".

Primary Goal:
- Act like a real tutor: be encouraging and natural, but keep your responses brief (1-3 sentences maximum).
- Do not overwhelm the user with long explanations. Focus on keeping the conversation moving.
- Provide a correction if they make a mistake, then continue the chat briefly.

Response Structure:
1. Correction (If needed): Start with "Correction: [Corrected sentence]".
2. Separator: Add "||".
3. Conversational Content: Your short, natural response (1-3 sentences).

Example:
User: "I study English for 2 years."
Tutor: "Correction: I have been studying English for two years.||That's a great milestone! Two years is usually when students start feeling more confident. What do you find most difficult about learning it?"`;

const LISTENING_SYSTEM_INSTRUCTION = `Generate an engaging listening comprehension exercise for level {LEVEL}.
1. Create a realistic dialogue (250-400 words).
2. Use exactly two speakers with tags like [Alice]: and [Bob]:.
3. Provide 3-4 multiple-choice questions.`;

const WRITING_CORRECTOR_SYSTEM_INSTRUCTION = `You are an expert English examiner for Cambridge Assessment. Correct the user's text based on their target level ({LEVEL}).

Assessment Criteria:
1. Content: Did they cover all points?
2. Communicative Achievement: Is the tone appropriate?
3. Organization: Is there a logical flow?
4. Language: Accuracy and range of grammar and vocabulary.

Output format (JSON):
{
  "score": "A score from 1-5 (Cambridge scale)",
  "summary": "Overall feedback",
  "corrections": [
    {"original": "...", "improved": "...", "explanation": "..."}
  ],
  "improvedVersion": "The full text rewritten professionally at {LEVEL} level."
}`;

function createPcmBlob(data: Float32Array): Blob {
    const l = data.length;
    const int16 = new Int16Array(l);
    for (let i = 0; i < l; i++) {
        int16[i] = data[i] * 32768;
    }
    return {
        data: encode(new Uint8Array(int16.buffer)),
        mimeType: `audio/pcm;rate=${INPUT_SAMPLE_RATE}`,
    };
}

async function fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => {
            const base64String = (reader.result as string).split(',')[1];
            resolve(base64String);
        };
        reader.onerror = (error) => reject(error);
    });
}

// --- UI Components ---
const LevelSelector: React.FC<{
    level: string;
    setLevel: (level: string) => void;
    disabled: boolean;
}> = ({ level, setLevel, disabled }) => (
    <div className="flex items-center gap-2">
        <span className="text-slate-400 font-medium">Target:</span>
        <select
            value={level}
            onChange={(e) => setLevel(e.target.value)}
            disabled={disabled}
            className="bg-slate-700 text-slate-100 rounded-lg px-3 py-1 border border-slate-600 focus:ring-2 focus:ring-cyan-500 focus:outline-none"
        >
            <option value="A2">A2 (Elementary)</option>
            <option value="B1">B1 (Intermediate)</option>
            <option value="B2">B2 (Upper Intermediate)</option>
            <option value="C1">C1 (Advanced)</option>
            <option value="C2">C2 (Proficiency)</option>
        </select>
    </div>
);

// --- Main App Component ---

export default function App() {
    const [mode, setMode] = useState<AppMode>(AppMode.CONVERSATION);

    // Conversation state
    const [topic, setTopic] = useState<string>('');
    const [level, setLevel] = useState<string>('B1');
    const [status, setStatus] = useState<SessionStatus>(SessionStatus.INACTIVE);
    const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
    const [isTutorReplying, setIsTutorReplying] = useState<boolean>(false);
    const [liveUserTranscript, setLiveUserTranscript] = useState<string>('');

    // Listening state
    const [listeningLevel, setListeningLevel] = useState<string>('B1');
    const [isGenerating, setIsGenerating] = useState<boolean>(false);
    const [exercise, setExercise] = useState<{ transcript: string, questions: string } | null>(null);
    const [listeningAudioBuffer, setListeningAudioBuffer] = useState<AudioBuffer | null>(null);
    const [isListeningAudioPlaying, setIsListeningAudioPlaying] = useState<boolean>(false);

    // Writing state
    const [writingInput, setWritingInput] = useState<string>('');
    const [writingLevel, setWritingLevel] = useState<string>('B2');
    const [isCorrecting, setIsCorrecting] = useState<boolean>(false);
    const [writingResult, setWritingResult] = useState<any | null>(null);
    const [uploadedFile, setUploadedFile] = useState<{ data: string, type: string } | null>(null);

    const sessionPromiseRef = useRef<Promise<LiveSession> | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const audioProcessorRef = useRef<ScriptProcessorNode | null>(null);
    const audioContextsRef = useRef<{ input?: AudioContext; output?: AudioContext }>({});
    const audioPlaybackQueueRef = useRef<{ nextStartTime: number, sources: Set<AudioBufferSourceNode> }>({ nextStartTime: 0, sources: new Set() });
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const currentInputTranscription = useRef<string>('');
    const currentOutputTranscription = useRef<string>('');

    // Auto-scroll logic
    const scrollToBottom = useCallback(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, []);

    useEffect(() => {
        if (mode === AppMode.CONVERSATION) {
            scrollToBottom();
        }
    }, [transcript, liveUserTranscript, isTutorReplying, mode, scrollToBottom]);

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
        audioProcessorRef.current?.disconnect();
        audioProcessorRef.current = null;
        audioPlaybackQueueRef.current.sources.forEach(source => { try { source.stop(); } catch (e) {} });
        audioPlaybackQueueRef.current.sources.clear();
        audioPlaybackQueueRef.current.nextStartTime = 0;
        if (audioContextsRef.current.input) {
            audioContextsRef.current.input.close().catch(console.error);
            delete audioContextsRef.current.input;
        }
    }, []);

    const stopSession = useCallback(async () => {
        if (sessionPromiseRef.current) {
            try { (await sessionPromiseRef.current).close(); } catch (e) {}
            sessionPromiseRef.current = null;
        }
        cleanupSessionResources();
        setIsTutorReplying(false);
        setLiveUserTranscript('');
        setStatus(SessionStatus.INACTIVE);
    }, [cleanupSessionResources]);

    const startSession = useCallback(async () => {
        if (!topic.trim()) return alert('Please enter a topic.');
        const apiKey = process.env.API_KEY;
        if (!apiKey) return setStatus(SessionStatus.ERROR);

        setTranscript([]);
        setStatus(SessionStatus.CONNECTING);
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            streamRef.current = stream;
            const ai = new GoogleGenAI({ apiKey });
            const outCtx = getOutputAudioContext();
            const inCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: INPUT_SAMPLE_RATE });
            audioContextsRef.current.input = inCtx;

            sessionPromiseRef.current = ai.live.connect({
                model: 'gemini-2.5-flash-native-audio-preview-09-2025',
                config: {
                    responseModalities: [Modality.AUDIO],
                    inputAudioTranscription: {},
                    outputAudioTranscription: {},
                    speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } } },
                    systemInstruction: CONVERSATION_TUTOR_SYSTEM_INSTRUCTION.replace('{TOPIC}', topic).replace('{LEVEL}', level),
                },
                callbacks: {
                    onopen: () => {
                        setStatus(SessionStatus.ACTIVE);
                        const source = inCtx.createMediaStreamSource(stream);
                        const scriptProcessor = inCtx.createScriptProcessor(SCRIPT_PROCESSOR_BUFFER_SIZE, 1, 1);
                        scriptProcessor.onaudioprocess = (e) => {
                            const inputData = e.inputBuffer.getChannelData(0);
                            sessionPromiseRef.current?.then(s => s.sendRealtimeInput({ media: createPcmBlob(inputData) }));
                        };
                        source.connect(scriptProcessor);
                        scriptProcessor.connect(inCtx.destination);
                        audioProcessorRef.current = scriptProcessor;
                    },
                    onmessage: async (msg: LiveServerMessage) => {
                        if (msg.serverContent?.inputTranscription) {
                            currentInputTranscription.current += msg.serverContent.inputTranscription.text;
                            setLiveUserTranscript(currentInputTranscription.current);
                        }
                        if (msg.serverContent?.outputTranscription) {
                            setIsTutorReplying(true);
                            currentOutputTranscription.current += msg.serverContent.outputTranscription.text;
                        }
                        if (msg.serverContent?.turnComplete) {
                            const input = currentInputTranscription.current.trim();
                            const output = currentOutputTranscription.current.trim();
                            currentInputTranscription.current = ''; currentOutputTranscription.current = '';
                            setLiveUserTranscript(''); setIsTutorReplying(false);

                            setTranscript(prev => {
                                const next = [...prev];
                                if (input) next.push({ speaker: Speaker.USER, text: input });
                                if (output) {
                                    const hasCorr = output.includes("||");
                                    const correction = hasCorr ? output.split("||")[0].replace("Correction:", "").trim() : undefined;
                                    const text = hasCorr ? output.split("||")[1].trim() : output;
                                    next.push({ speaker: Speaker.TUTOR, text, correction });
                                }
                                return next;
                            });
                        }
                        const b64 = msg.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
                        if (b64) {
                            const start = Math.max(audioPlaybackQueueRef.current.nextStartTime, outCtx.currentTime);
                            const buffer = await decodeAudioData(decode(b64), outCtx, OUTPUT_SAMPLE_RATE, 1);
                            const node = outCtx.createBufferSource();
                            node.buffer = buffer;
                            node.connect(outCtx.destination);
                            node.onended = () => audioPlaybackQueueRef.current.sources.delete(node);
                            node.start(start);
                            audioPlaybackQueueRef.current.nextStartTime = start + buffer.duration;
                            audioPlaybackQueueRef.current.sources.add(node);
                        }
                    },
                    onerror: () => stopSession(),
                },
            });
        } catch (e) { stopSession(); }
    }, [topic, level, stopSession, getOutputAudioContext]);

    const handleWritingFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        const base64 = await fileToBase64(file);
        setUploadedFile({ data: base64, type: file.type });
    };

    const runWritingCorrection = async () => {
        const apiKey = process.env.API_KEY;
        if (!apiKey || (!writingInput.trim() && !uploadedFile)) return;
        setIsCorrecting(true);
        setWritingResult(null);

        try {
            const ai = new GoogleGenAI({ apiKey });
            const parts: any[] = [{ text: `Correct this English text based on level ${writingLevel}. Consider Cambridge criteria.` }];
            if (writingInput) parts.push({ text: `Content: ${writingInput}` });
            if (uploadedFile) {
                parts.push({
                    inlineData: { data: uploadedFile.data, mimeType: uploadedFile.type }
                });
            }

            const res = await ai.models.generateContent({
                model: 'gemini-3-flash-preview',
                contents: { parts },
                config: {
                    systemInstruction: WRITING_CORRECTOR_SYSTEM_INSTRUCTION.replace('{LEVEL}', writingLevel),
                    responseMimeType: "application/json",
                    responseSchema: {
                        type: Type.OBJECT,
                        properties: {
                            score: { type: Type.STRING },
                            summary: { type: Type.STRING },
                            corrections: {
                                type: Type.ARRAY,
                                items: {
                                    type: Type.OBJECT,
                                    properties: {
                                        original: { type: Type.STRING },
                                        improved: { type: Type.STRING },
                                        explanation: { type: Type.STRING }
                                    }
                                }
                            },
                            improvedVersion: { type: Type.STRING }
                        },
                        required: ['score', 'summary', 'corrections', 'improvedVersion']
                    }
                }
            });
            setWritingResult(JSON.parse(res.text));
        } catch (e) { alert("Correction failed."); } finally { setIsCorrecting(false); }
    };

    const generateListeningExercise = async () => {
        const apiKey = process.env.API_KEY;
        if (!apiKey) return;
        setIsGenerating(true); setExercise(null); setListeningAudioBuffer(null);
        try {
            const ai = new GoogleGenAI({ apiKey });
            const topic = LISTENING_TOPICS[Math.floor(Math.random() * LISTENING_TOPICS.length)];
            const res = await ai.models.generateContent({
                model: 'gemini-3-flash-preview',
                contents: `Create listening exercise for level ${listeningLevel} on ${topic}`,
                config: {
                    systemInstruction: LISTENING_SYSTEM_INSTRUCTION.replace('{LEVEL}', listeningLevel),
                    responseMimeType: "application/json",
                    responseSchema: {
                        type: Type.OBJECT,
                        properties: {
                            transcript: { type: Type.STRING },
                            questions: { type: Type.STRING }
                        }
                    }
                }
            });
            const json = JSON.parse(res.text);
            setExercise(json);

            const tts = await ai.models.generateContent({
                model: "gemini-2.5-flash-preview-tts",
                contents: [{ parts: [{ text: json.transcript }] }],
                config: {
                    responseModalities: [Modality.AUDIO],
                    speechConfig: {
                        multiSpeakerVoiceConfig: {
                            speakerVoiceConfigs: [
                                { speaker: 'Alice', voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } } },
                                { speaker: 'Bob', voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Puck' } } }
                            ]
                        }
                    }
                }
            });
            const b64 = tts.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
            if (b64) setListeningAudioBuffer(await decodeAudioData(decode(b64), getOutputAudioContext(), OUTPUT_SAMPLE_RATE, 1));
        } catch (e) { alert("Generation failed."); } finally { setIsGenerating(false); }
    };

    return (
        <div className="h-screen w-screen bg-slate-950 text-slate-100 flex flex-col font-sans overflow-hidden">
            <header className="p-4 border-b border-slate-800 flex justify-between items-center bg-slate-900/50">
                <h1 className="text-2xl font-black tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-500 to-indigo-500">
                    ENGLISH MASTERY AI
                </h1>
                <div className="flex bg-slate-800 rounded-full p-1 border border-slate-700">
                    {[AppMode.CONVERSATION, AppMode.LISTENING, AppMode.WRITING].map(m => (
                        <button key={m} onClick={() => setMode(m)} className={`px-4 py-1.5 rounded-full text-sm font-bold transition-all ${mode === m ? 'bg-cyan-600 shadow-lg' : 'text-slate-400 hover:text-white'}`}>
                            {m.charAt(0) + m.slice(1).toLowerCase()}
                        </button>
                    ))}
                </div>
            </header>

            <main className="flex-grow flex flex-col p-4 gap-4 overflow-hidden max-w-6xl mx-auto w-full">
                {mode === AppMode.CONVERSATION && (
                    <div className="flex-grow flex flex-col overflow-hidden gap-4">
                        <div className="flex-grow bg-slate-900/50 rounded-3xl border border-slate-800 p-6 overflow-y-auto space-y-4">
                            {transcript.length === 0 && (
                                <div className="h-full flex flex-col items-center justify-center text-slate-500 text-center p-8">
                                    <MicIcon className="w-16 h-16 mb-4 opacity-20" />
                                    <p className="text-xl font-medium">Start a conversation to improve your fluency.</p>
                                    <p className="text-sm mt-2">I will provide audio input and correct your grammar as we talk.</p>
                                </div>
                            )}
                            {transcript.map((e, i) => (
                                <div key={i} className={`flex ${e.speaker === Speaker.USER ? 'justify-end' : 'justify-start'}`}>
                                    <div className={`max-w-[85%] p-4 rounded-2xl shadow-sm ${e.speaker === Speaker.USER ? 'bg-indigo-600 text-white rounded-br-none' : 'bg-slate-800 border border-slate-700 rounded-bl-none'}`}>
                                        {e.correction && (
                                            <div className="mb-2 p-2 bg-yellow-400/10 border border-yellow-400/20 rounded-lg text-xs">
                                                <span className="font-bold text-yellow-400">Tutor's Correction: </span>
                                                <span className="italic">"{e.correction}"</span>
                                            </div>
                                        )}
                                        <p className="leading-relaxed">{e.text}</p>
                                    </div>
                                </div>
                            ))}
                            {liveUserTranscript && (
                                <div className="flex justify-end opacity-50">
                                    <div className="bg-indigo-600/50 p-4 rounded-2xl rounded-br-none italic">{liveUserTranscript}...</div>
                                </div>
                            )}
                            {isTutorReplying && (
                                <div className="flex justify-start">
                                    <div className="bg-slate-800 p-4 rounded-2xl rounded-bl-none flex items-center gap-2">
                                        <div className="flex space-x-1">
                                            <div className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce"></div>
                                            <div className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce [animation-delay:0.2s]"></div>
                                            <div className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce [animation-delay:0.4s]"></div>
                                        </div>
                                    </div>
                                </div>
                            )}
                            {/* Hidden element to mark the end of the conversation */}
                            <div ref={messagesEndRef} className="h-1 w-full" />
                        </div>
                        <div className="bg-slate-900 border border-slate-800 p-4 rounded-3xl shadow-2xl flex flex-col md:flex-row items-center gap-4">
                            <input
                                placeholder="Conversation Topic (e.g. Travel, Jobs, Movies)"
                                className="flex-grow bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-cyan-500"
                                value={topic} onChange={e => setTopic(e.target.value)} disabled={status !== SessionStatus.INACTIVE}
                            />
                            <LevelSelector level={level} setLevel={setLevel} disabled={status !== SessionStatus.INACTIVE} />
                            {status === SessionStatus.INACTIVE ? (
                                <button onClick={startSession} className="bg-cyan-600 hover:bg-cyan-500 text-white px-8 py-3 rounded-xl font-bold flex items-center gap-2 transition-all">
                                    <MicIcon className="w-5 h-5" /> Start
                                </button>
                            ) : (
                                <button onClick={stopSession} className="bg-rose-600 hover:bg-rose-500 text-white px-8 py-3 rounded-xl font-bold flex items-center gap-2 transition-all">
                                    <StopIcon className="w-5 h-5" /> Stop
                                </button>
                            )}
                        </div>
                    </div>
                )}

                {mode === AppMode.WRITING && (
                    <div className="flex-grow flex flex-col md:flex-row gap-4 overflow-hidden">
                        <div className="flex-1 flex flex-col gap-4">
                            <div className="flex-grow bg-slate-900/50 rounded-3xl border border-slate-800 p-6 flex flex-col">
                                <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
                                    <DocumentCheckIcon className="w-6 h-6 text-cyan-400" /> Writing Workspace
                                </h2>
                                <textarea
                                    className="flex-grow bg-transparent border-none outline-none resize-none text-lg text-slate-300 placeholder:text-slate-600 leading-relaxed"
                                    placeholder="Paste your essay or text here to be evaluated under Cambridge criteria..."
                                    value={writingInput} onChange={e => setWritingInput(e.target.value)}
                                />
                                <div className="mt-4 flex items-center gap-4 pt-4 border-t border-slate-800">
                                    <label className="flex items-center gap-2 text-sm text-slate-400 cursor-pointer hover:text-cyan-400 transition-colors">
                                        <UploadIcon className="w-5 h-5" />
                                        <span>{uploadedFile ? 'File Attached' : 'Upload Image/Doc'}</span>
                                        <input type="file" className="hidden" accept="image/*,.pdf" onChange={handleWritingFile} />
                                    </label>
                                    <div className="flex-grow" />
                                    <LevelSelector level={writingLevel} setLevel={setWritingLevel} disabled={isCorrecting} />
                                    <button
                                        onClick={runWritingCorrection}
                                        disabled={isCorrecting || (!writingInput.trim() && !uploadedFile)}
                                        className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 px-6 py-2 rounded-xl font-bold flex items-center gap-2 transition-all"
                                    >
                                        {isCorrecting ? <LoadingSpinner className="w-5 h-5" /> : <SparklesIcon className="w-5 h-5" />}
                                        Analyze
                                    </button>
                                </div>
                            </div>
                        </div>
                        <div className="flex-1 bg-slate-900/50 rounded-3xl border border-slate-800 p-6 overflow-y-auto">
                            {!writingResult && !isCorrecting && (
                                <div className="h-full flex items-center justify-center text-slate-600 text-center italic">
                                    Analysis results will appear here.
                                </div>
                            )}
                            {isCorrecting && (
                                <div className="h-full flex flex-col items-center justify-center gap-4">
                                    <LoadingSpinner className="w-12 h-12 text-cyan-500" />
                                    <p className="text-slate-400 animate-pulse">Evaluating based on Cambridge Assessment criteria...</p>
                                </div>
                            )}
                            {writingResult && (
                                <div className="space-y-6 animate-fade-in">
                                    <div className="flex justify-between items-center bg-slate-800 p-4 rounded-2xl border border-slate-700">
                                        <div className="text-sm font-bold uppercase tracking-wider text-slate-400">Overall Score</div>
                                        <div className="text-3xl font-black text-cyan-400">{writingResult.score}/5</div>
                                    </div>
                                    <div>
                                        <h3 className="font-bold text-slate-200 mb-2">Examiner's Summary</h3>
                                        <p className="text-slate-400 text-sm leading-relaxed">{writingResult.summary}</p>
                                    </div>
                                    <div>
                                        <h3 className="font-bold text-slate-200 mb-2">Key Improvements</h3>
                                        <div className="space-y-3">
                                            {writingResult.corrections.map((c: any, i: number) => (
                                                <div key={i} className="bg-slate-800/50 p-3 rounded-xl border-l-4 border-yellow-500">
                                                    <p className="text-xs text-rose-400 line-through mb-1">{c.original}</p>
                                                    <p className="text-sm text-emerald-400 font-medium mb-1">{c.improved}</p>
                                                    <p className="text-[11px] text-slate-500 italic">{c.explanation}</p>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                    <div>
                                        <h3 className="font-bold text-slate-200 mb-2">Final Polished Version</h3>
                                        <div className="bg-slate-950 p-4 rounded-xl text-slate-300 text-sm leading-relaxed border border-slate-800">
                                            {writingResult.improvedVersion}
                                        </div>
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
                                <p className="text-slate-500">Generate a professional dialogue with questions and multi-speaker audio to practice your ear.</p>
                                <div className="flex items-center justify-center gap-4 py-4">
                                    <LevelSelector level={listeningLevel} setLevel={setListeningLevel} disabled={isGenerating} />
                                    <button onClick={generateListeningExercise} className="bg-purple-600 px-6 py-2 rounded-xl font-bold">Generate</button>
                                </div>
                            </div>
                        )}
                        {isGenerating && (
                            <div className="m-auto flex flex-col items-center gap-4">
                                <LoadingSpinner className="w-16 h-16 text-purple-500" />
                                <p className="text-xl font-medium animate-pulse">Creating your exercise...</p>
                            </div>
                        )}
                        {exercise && (
                            <div className="space-y-8 max-w-2xl mx-auto w-full">
                                <div className="bg-slate-800 p-6 rounded-3xl border border-slate-700 flex items-center justify-between">
                                    <div>
                                        <h3 className="text-lg font-bold">Dialogue Audio</h3>
                                        <p className="text-sm text-slate-400">Listen carefully and answer below.</p>
                                    </div>
                                    <button
                                        onClick={() => {
                                            if (isListeningAudioPlaying) {
                                                // Handle stop if logic added
                                                setIsListeningAudioPlaying(false);
                                            } else if (listeningAudioBuffer) {
                                                const ctx = getOutputAudioContext();
                                                const s = ctx.createBufferSource();
                                                s.buffer = listeningAudioBuffer;
                                                s.connect(ctx.destination);
                                                s.onended = () => setIsListeningAudioPlaying(false);
                                                s.start();
                                                setIsListeningAudioPlaying(true);
                                            }
                                        }}
                                        disabled={!listeningAudioBuffer}
                                        className="bg-cyan-600 p-4 rounded-full shadow-lg shadow-cyan-900/20 hover:scale-105 transition-transform"
                                    >
                                        {isListeningAudioPlaying ? <StopIcon className="w-8 h-8" /> : <PlayIcon className="w-8 h-8" />}
                                    </button>
                                </div>
                                <div className="space-y-4">
                                    <h4 className="text-cyan-400 font-bold uppercase tracking-widest text-xs">Questions</h4>
                                    <div className="bg-slate-800/50 p-6 rounded-3xl border border-slate-800 text-slate-200 leading-loose whitespace-pre-wrap">
                                        {exercise.questions}
                                    </div>
                                </div>
                                <details className="group">
                                    <summary className="cursor-pointer text-sm font-bold text-slate-500 hover:text-cyan-400 transition-colors list-none flex items-center gap-2">
                                        <span className="group-open:rotate-90 transition-transform">▶</span> Show Transcript
                                    </summary>
                                    <div className="mt-4 bg-slate-950 p-6 rounded-2xl border border-slate-800 text-slate-400 font-mono text-sm leading-relaxed">
                                        {exercise.transcript}
                                    </div>
                                </details>
                                <button onClick={() => setExercise(null)} className="text-slate-600 text-sm hover:underline w-full text-center">Clear and generate new exercise</button>
                            </div>
                        )}
                    </div>
                )}
            </main>
        </div>
    );
}