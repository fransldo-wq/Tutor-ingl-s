import React, { useState, useRef, useCallback, useEffect } from 'react';
import { GoogleGenAI, LiveSession, LiveServerMessage, Modality, Blob } from '@google/genai';
import { Speaker, TranscriptEntry, SessionStatus, AppMode, Correction } from './types';
import { encode, decode, decodeAudioData } from './utils/audio';
import { MicIcon, StopIcon, LoadingSpinner, SparklesIcon, PlayIcon } from './components/Icons';

// --- Helper Functions & Constants ---
const INPUT_SAMPLE_RATE = 16000;
const OUTPUT_SAMPLE_RATE = 24000;
const SCRIPT_PROCESSOR_BUFFER_SIZE = 4096;

const TUTOR_SYSTEM_INSTRUCTION = `You are a world-class, native-speaking English language tutor. Your primary goal is to facilitate immersive language practice. You must be encouraging, patient, and highly expressive.

KEY RULES:

- Always Speak in English. All tutor output must be in English.
- Be Verbose: In Conversation Mode, your responses should be significantly longer (minimum 3-5 sentences) than a typical conversational partner's to maximize user listening practice.
- Strictly Use Defined XML-like Tags for all output elements, which will enable the user's application to render them (e.g., highlighting corrections in yellow).

I. CONVERSATION TUTOR MODE

The user will select a Topic (e.g., 'Travel Planning', 'Current Events', 'Cooking') and a Level (B1, B2, C1).

Tutor Response Format (Conversation)
Your response must contain exactly two sections: <TUTOR_SPEECH> and optionally <CORRECTION>.

<TUTOR_SPEECH>: This is your main conversational reply to the user's last statement.
- Your reply must align with the selected topic and difficulty level.
- Maintain the flow of the dialogue.

<CORRECTION> (Mandatory if user made errors): If the user's last statement contained a grammatical, lexical, or phrasing error (if detected from transcribed input, assume typical errors), you must include this tag.
- Structure: <CORRECTION>Original expression: "[User's phrase]". Correct expression: "[The grammatically correct, natural phrase]". Explanation: [Brief explanation of the error (e.g., tense, preposition, or more natural phrasing)].</CORRECTION>
- This section should immediately follow <TUTOR_SPEECH>.
- Goal: To provide immediate, constructive feedback that the user's app will display prominently (e.g., in yellow).

Example Start (First Turn):
Upon receiving the topic "Travel Planning" and level "B2", your first response should be:
<TUTOR_SPEECH>Hello! That's a great topic. At the B2 level, we can discuss travel in detail. So, tell me, if you could go anywhere in the world right now, where would you choose and why?</TUTOR_SPEECH>

II. LISTENING COMPREHENSION MODE

The user will select the Level (B1, B2, C1) and the "Listening" mode.

Listening Mode Process
1. Generate a Scenario: On a random, engaging topic, create a realistic dialogue (approx. 200-300 words) appropriate for the selected level. The dialogue should feature exactly two distinct speakers to simulate a real conversation (e.g., Alice and Bob).
   - Transcript Format: Use the format [Speaker Name]: [Dialogue Line].
2. Generate Comprehension Questions: Following the transcript, generate 3-4 comprehension questions covering main ideas, details, and inferences, in multiple-choice format (A, B, C). Do not provide the answers.

Listening Response Format
Your response must contain exactly two sections: <TRANSCRIPT> and <QUESTIONS>.

<TRANSCRIPT>: Contains the full dialogue.
- Example Format: [Alice]: That new café on Elm Street is fantastic! [Bob]: Oh really? I thought it was just okay. (Continue dialogue...)

<QUESTIONS>: Contains the multiple-choice questions.
- Example Format:
1. What is Bob’s initial opinion of the café?
   A. He believes it is fantastic.
   B. He thinks it is just acceptable.
   C. He hasn't been there yet.
2. What specific item does Chris praise?
   A. The coffee quality.
   B. The location.
   C. The pastries.

Continue this pattern for all 3-4 questions.

III. PRESENTATION
Ensure all generated content is clean, well-structured, and highly readable, simulating an engaging and professional interface.

BEGIN SESSION: The user will now provide the initial input (Topic/Level for Conversation, or "Listening" for Listening Comprehension Mode).`;


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

function parseCorrection(correctionText: string): Correction | null {
    const originalMatch = correctionText.match(/Original expression: "(.*?)"/);
    const correctedMatch = correctionText.match(/Correct expression: "(.*?)"/);
    const explanationMatch = correctionText.match(/Explanation: (.*)/);

    if (originalMatch && correctedMatch && explanationMatch) {
        return {
            original: originalMatch[1],
            corrected: correctedMatch[1],
            explanation: explanationMatch[1],
        };
    }
    return null;
}

// --- UI Components ---
const LevelSelector: React.FC<{
    level: string;
    setLevel: (level: string) => void;
    disabled: boolean;
}> = ({ level, setLevel, disabled }) => (
    <div className="flex items-center gap-2">
        <span className="text-slate-400 font-medium">Level:</span>
        <select
            value={level}
            onChange={(e) => setLevel(e.target.value)}
            disabled={disabled}
            className="bg-slate-700 text-slate-100 rounded-lg px-3 py-1 border border-slate-600 focus:ring-2 focus:ring-cyan-500 focus:outline-none"
        >
            <option value="B1">B1</option>
            <option value="B2">B2</option>
            <option value="C1">C1</option>
        </select>
    </div>
);

interface ConversationControlsProps {
    status: SessionStatus;
    topic: string;
    level: string;
    onTopicChange: (topic: string) => void;
    onLevelChange: (level: string) => void;
    onStart: () => void;
    onStop: () => void;
}

const ConversationControls: React.FC<ConversationControlsProps> = ({ status, topic, level, onTopicChange, onLevelChange, onStart, onStop }) => {
    const isInactive = status === SessionStatus.INACTIVE || status === SessionStatus.ERROR;
    const isConnecting = status === SessionStatus.CONNECTING;

    return (
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl p-4 shadow-2xl border border-slate-700">
            <div className="flex flex-col md:flex-row items-center gap-4">
                <input
                    type="text"
                    value={topic}
                    onChange={(e) => onTopicChange(e.target.value)}
                    placeholder="Enter conversation topic..."
                    disabled={!isInactive}
                    className="w-full flex-grow bg-slate-700 text-slate-100 placeholder-slate-400 rounded-lg px-4 py-3 border border-slate-600 focus:ring-2 focus:ring-cyan-500 focus:outline-none transition-all duration-300"
                />
                <LevelSelector level={level} setLevel={onLevelChange} disabled={!isInactive} />
                {isInactive ? (
                    <button
                        onClick={onStart}
                        disabled={!topic.trim()}
                        className="w-full md:w-auto flex items-center justify-center gap-2 px-6 py-3 bg-cyan-600 text-white font-semibold rounded-lg shadow-md hover:bg-cyan-700 disabled:bg-slate-600 disabled:cursor-not-allowed transition-all duration-300"
                    >
                        <MicIcon className="w-5 h-5" />
                        <span>Start</span>
                    </button>
                ) : (
                    <button
                        onClick={onStop}
                        disabled={isConnecting}
                        className="w-full md:w-auto flex items-center justify-center gap-2 px-6 py-3 bg-rose-600 text-white font-semibold rounded-lg shadow-md hover:bg-rose-700 disabled:bg-slate-600 transition-all duration-300"
                    >
                        {isConnecting ? (
                            <>
                                <LoadingSpinner className="w-5 h-5" />
                                <span>Connecting...</span>
                            </>
                        ) : (
                            <>
                                <StopIcon className="w-5 h-5" />
                                <span>Stop</span>
                            </>
                        )}
                    </button>
                )}
            </div>
        </div>
    );
};

interface TranscriptViewProps {
    transcript: TranscriptEntry[];
}

const TranscriptView: React.FC<TranscriptViewProps> = ({ transcript }) => {
    const endOfMessagesRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [transcript]);

    return (
        <div className="flex-grow space-y-6 overflow-y-auto p-4 md:p-6">
            {transcript.length === 0 && (
                <div className="text-center text-slate-400 py-10">
                    <h2 className="text-2xl font-bold text-slate-200 mb-2">Welcome!</h2>
                    <p>Enter a topic, select a level, and click "Start" to begin your practice session.</p>
                </div>
            )}
            {transcript.map((entry, index) => (
                <div key={index} className={`flex items-start gap-4 ${entry.speaker === Speaker.USER ? 'flex-row-reverse' : 'flex-row'}`}>
                    <div className={`max-w-xl p-4 rounded-2xl shadow-md ${entry.speaker === Speaker.USER ? 'bg-blue-600 text-white rounded-br-none' : 'bg-slate-700 text-slate-200 rounded-bl-none'}`}>
                        <p className="font-bold mb-1">{entry.speaker}</p>
                        <p className="whitespace-pre-wrap">{entry.text}</p>
                        {entry.speaker === Speaker.TUTOR && entry.correction && (
                            <div className="mt-3 p-3 bg-yellow-500/10 border-l-4 border-yellow-400 rounded-r-lg">
                                <p className="text-sm text-yellow-200 font-semibold">Correction Feedback</p>
                                <p className="text-sm text-rose-300 mt-2"><span className="font-bold">Original:</span> "{entry.correction.original}"</p>
                                <p className="text-sm text-green-300 mt-1"><span className="font-bold">Suggested:</span> "{entry.correction.corrected}"</p>
                                <p className="text-sm text-slate-300 mt-2"><span className="font-bold">Tip:</span> {entry.correction.explanation}</p>
                            </div>
                        )}
                    </div>
                </div>
            ))}
            <div ref={endOfMessagesRef} />
        </div>
    );
};

interface ModeSelectorProps {
    mode: AppMode;
    setMode: (mode: AppMode) => void;
}

const ModeSelector: React.FC<ModeSelectorProps> = ({ mode, setMode }) => (
    <div className="flex justify-center p-2 bg-slate-800/80 rounded-xl">
        <button
            onClick={() => setMode(AppMode.CONVERSATION)}
            className={`px-6 py-2 rounded-lg font-semibold transition-all duration-300 ${mode === AppMode.CONVERSATION ? 'bg-cyan-600 text-white shadow-md' : 'bg-transparent text-slate-300 hover:bg-slate-700'}`}
        >
            Conversation Tutor
        </button>
        <button
            onClick={() => setMode(AppMode.LISTENING)}
            className={`px-6 py-2 rounded-lg font-semibold transition-all duration-300 ${mode === AppMode.LISTENING ? 'bg-cyan-600 text-white shadow-md' : 'bg-transparent text-slate-300 hover:bg-slate-700'}`}
        >
            Listening Comprehension
        </button>
    </div>
);


// --- Main App Component ---

export default function App() {
    // Shared state
    const [mode, setMode] = useState<AppMode>(AppMode.CONVERSATION);

    // Conversation mode state
    const [topic, setTopic] = useState<string>('');
    const [level, setLevel] = useState<string>('B1');
    const [status, setStatus] = useState<SessionStatus>(SessionStatus.INACTIVE);
    const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);

    // Listening mode state
    const [listeningLevel, setListeningLevel] = useState<string>('B1');
    const [isGenerating, setIsGenerating] = useState<boolean>(false);
    const [exercise, setExercise] = useState<{ transcript: string, questions: string } | null>(null);
    const [listeningAudioBuffer, setListeningAudioBuffer] = useState<AudioBuffer | null>(null);
    const [isListeningAudioPlaying, setIsListeningAudioPlaying] = useState<boolean>(false);
    const [showListeningTranscript, setShowListeningTranscript] = useState<boolean>(false);

    // Mutable refs for managing session and audio resources
    const sessionPromiseRef = useRef<Promise<LiveSession> | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const audioProcessorRef = useRef<ScriptProcessorNode | null>(null);
    const audioContextsRef = useRef<{ input?: AudioContext; output?: AudioContext } | null>(null);
    const audioPlaybackQueueRef = useRef<{ nextStartTime: number, sources: Set<AudioBufferSourceNode> }>({ nextStartTime: 0, sources: new Set() });
    const listeningAudioSourceRef = useRef<AudioBufferSourceNode | null>(null);

    const currentInputTranscription = useRef<string>('');
    const currentOutputTranscription = useRef<string>('');

    // --- Core Logic ---
    const cleanupAudio = useCallback(() => {
        // Conversation mode cleanup
        streamRef.current?.getTracks().forEach(track => track.stop());
        streamRef.current = null;
        audioProcessorRef.current?.disconnect();
        audioProcessorRef.current = null;
        audioPlaybackQueueRef.current.sources.forEach(source => {
            try { source.stop(); } catch (e) { /* ignore */ }
        });
        audioPlaybackQueueRef.current.sources.clear();
        audioPlaybackQueueRef.current.nextStartTime = 0;

        // Listening mode cleanup
        if (listeningAudioSourceRef.current) {
            try { listeningAudioSourceRef.current.stop(); } catch (e) { /* ignore */ }
            listeningAudioSourceRef.current = null;
        }
        setIsListeningAudioPlaying(false);

        // Close all audio contexts
        if (audioContextsRef.current) {
            audioContextsRef.current.input?.close().catch(console.error);
            audioContextsRef.current.output?.close().catch(console.error);
            audioContextsRef.current = null;
        }
    }, []);

    const stopSession = useCallback(async () => {
        if (sessionPromiseRef.current) {
            try {
                const session = await sessionPromiseRef.current;
                session.close();
            } catch (error) {
                console.error('Error closing session:', error);
            }
            sessionPromiseRef.current = null;
        }
        cleanupAudio();
        setStatus(SessionStatus.INACTIVE);
    }, [cleanupAudio]);


    const startSession = useCallback(async () => {
        if (!topic.trim()) {
            alert('Please enter a topic.');
            return;
        }
        const apiKey = process.env.API_KEY;
        if (!apiKey) {
            alert('API Key is not configured.');
            setStatus(SessionStatus.ERROR);
            return;
        }
        setTranscript([]);
        setStatus(SessionStatus.CONNECTING);
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            streamRef.current = stream;
            const ai = new GoogleGenAI({ apiKey });
            const inputAudioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: INPUT_SAMPLE_RATE });
            const outputAudioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: OUTPUT_SAMPLE_RATE });
            audioContextsRef.current = { input: inputAudioContext, output: outputAudioContext };
            audioPlaybackQueueRef.current = { nextStartTime: 0, sources: new Set() };
            const dynamicSystemInstruction = `${TUTOR_SYSTEM_INSTRUCTION}\n\nThe user has selected Conversation Mode. Topic: "${topic}", Level: "${level}". Please start the conversation now.`;
            sessionPromiseRef.current = ai.live.connect({
                model: 'gemini-2.5-flash-native-audio-preview-09-2025',
                config: {
                    responseModalities: [Modality.AUDIO],
                    inputAudioTranscription: {},
                    outputAudioTranscription: {},
                    speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } } },
                    systemInstruction: dynamicSystemInstruction,
                },
                callbacks: {
                    onopen: () => {
                        setStatus(SessionStatus.ACTIVE);
                        const source = inputAudioContext.createMediaStreamSource(stream);
                        const scriptProcessor = inputAudioContext.createScriptProcessor(SCRIPT_PROCESSOR_BUFFER_SIZE, 1, 1);
                        scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
                            const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
                            sessionPromiseRef.current?.then((session) => {
                                session.sendRealtimeInput({ media: createPcmBlob(inputData) });
                            });
                        };
                        source.connect(scriptProcessor);
                        scriptProcessor.connect(inputAudioContext.destination);
                        audioProcessorRef.current = scriptProcessor;
                    },
                    onmessage: async (message: LiveServerMessage) => {
                        if (message.serverContent?.inputTranscription) { currentInputTranscription.current += message.serverContent.inputTranscription.text; }
                        if (message.serverContent?.outputTranscription) { currentOutputTranscription.current += message.serverContent.outputTranscription.text; }
                        if (message.serverContent?.turnComplete) {
                            const finalInput = currentInputTranscription.current.trim();
                            const fullOutput = currentOutputTranscription.current.trim();
                            const speechMatch = fullOutput.match(/<TUTOR_SPEECH>(.*?)<\/TUTOR_SPEECH>/s);
                            const correctionMatch = fullOutput.match(/<CORRECTION>(.*?)<\/CORRECTION>/s);
                            const finalText = speechMatch ? speechMatch[1].trim() : fullOutput;
                            const correction = correctionMatch ? parseCorrection(correctionMatch[1].trim()) : null;
                            setTranscript(prev => {
                                let newTranscript = [...prev];
                                if (finalInput) newTranscript.push({ speaker: Speaker.USER, text: finalInput });
                                if (finalText) newTranscript.push({ speaker: Speaker.TUTOR, text: finalText, correction: correction ?? undefined });
                                return newTranscript;
                            });
                            currentInputTranscription.current = '';
                            currentOutputTranscription.current = '';
                        }
                        const base64Audio = message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
                        if (base64Audio && outputAudioContext) {
                            const { nextStartTime, sources } = audioPlaybackQueueRef.current;
                            const effectiveStartTime = Math.max(nextStartTime, outputAudioContext.currentTime);
                            const audioBuffer = await decodeAudioData(decode(base64Audio), outputAudioContext, OUTPUT_SAMPLE_RATE, 1);
                            const sourceNode = outputAudioContext.createBufferSource();
                            sourceNode.buffer = audioBuffer;
                            sourceNode.connect(outputAudioContext.destination);
                            sourceNode.addEventListener('ended', () => { sources.delete(sourceNode); });
                            sourceNode.start(effectiveStartTime);
                            audioPlaybackQueueRef.current.nextStartTime = effectiveStartTime + audioBuffer.duration;
                            sources.add(sourceNode);
                        }
                        if (message.serverContent?.interrupted) {
                            audioPlaybackQueueRef.current.sources.forEach(source => source.stop());
                            audioPlaybackQueueRef.current.sources.clear();
                            audioPlaybackQueueRef.current.nextStartTime = 0;
                        }
                    },
                    onerror: (e: ErrorEvent) => {
                        console.error('Session error:', e);
                        alert('A connection error occurred.');
                        setStatus(SessionStatus.ERROR);
                        stopSession();
                    },
                    onclose: () => {
                        console.log('Session closed.');
                        cleanupAudio();
                        setStatus(SessionStatus.INACTIVE);
                    },
                },
            });
        } catch (error) {
            console.error('Failed to start session:', error);
            let errorMessage = 'Failed to start session.';
            if (error instanceof Error && (error.name === 'NotAllowedError' || error.message.includes('Permission denied'))) {
                errorMessage = 'Microphone permission was denied.';
            }
            alert(errorMessage);
            cleanupAudio();
            setStatus(SessionStatus.INACTIVE);
        }
    }, [topic, level, cleanupAudio, stopSession]);

    const generateListeningExercise = useCallback(async () => {
        const apiKey = process.env.API_KEY;
        if (!apiKey) {
            alert('API Key is not configured.');
            return;
        }

        setIsGenerating(true);
        setExercise(null);
        setListeningAudioBuffer(null);
        setShowListeningTranscript(false);
        if (listeningAudioSourceRef.current) {
            try { listeningAudioSourceRef.current.stop(); } catch (e) { /* ignore */ }
        }

        try {
            const ai = new GoogleGenAI({ apiKey });
            // Step 1: Generate the text for the transcript and questions
            const textPrompt = `Generate a listening comprehension exercise on a random topic for level ${listeningLevel}.`;
            const textResponse = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: textPrompt,
                config: { systemInstruction: TUTOR_SYSTEM_INSTRUCTION },
            });
            const responseText = textResponse.text;
            const transcriptMatch = responseText.match(/<TRANSCRIPT>([\s\S]*?)<\/TRANSCRIPT>/);
            const questionsMatch = responseText.match(/<QUESTIONS>([\s\S]*?)<\/QUESTIONS>/);

            if (!transcriptMatch || !questionsMatch) {
                throw new Error("Failed to parse the exercise from the AI's response.");
            }
            const transcriptText = transcriptMatch[1].trim();
            const questionsText = questionsMatch[1].trim();
            setExercise({ transcript: transcriptText, questions: questionsText });

            // Step 2: Generate the audio from the transcript text
            const speakerRegex = /\[([^\]]+)\]:/g;
            const speakers = [...new Set(transcriptText.match(speakerRegex)?.map(s => s.replace(/[\[\]:]/g, '').trim()) ?? [])];
            if (speakers.length !== 2) {
                throw new Error(`Expected 2 speakers, but found ${speakers.length}. Please try generating again.`);
            }
            
            const audioResponse = await ai.models.generateContent({
                model: "gemini-2.5-flash-preview-tts",
                contents: [{ parts: [{ text: transcriptText }] }],
                config: {
                    responseModalities: [Modality.AUDIO],
                    speechConfig: {
                        multiSpeakerVoiceConfig: {
                            speakerVoiceConfigs: [
                                { speaker: speakers[0], voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } } },
                                { speaker: speakers[1], voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Puck' } } }
                            ]
                        }
                    }
                }
            });

            const base64Audio = audioResponse.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
            if (!base64Audio) {
                throw new Error("Failed to generate audio data.");
            }

            // Step 3: Decode audio data and store it in the buffer
            if (!audioContextsRef.current?.output || audioContextsRef.current.output.state === 'closed') {
                audioContextsRef.current = { ...audioContextsRef.current, output: new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: OUTPUT_SAMPLE_RATE }) };
            }
            const outputAudioContext = audioContextsRef.current.output;
            const audioBuffer = await decodeAudioData(decode(base64Audio), outputAudioContext, OUTPUT_SAMPLE_RATE, 1);
            setListeningAudioBuffer(audioBuffer);

        } catch (error) {
            console.error("Failed to generate exercise:", error);
            alert(`Sorry, there was an error generating the exercise: ${error instanceof Error ? error.message : String(error)}`);
            setExercise(null);
        } finally {
            setIsGenerating(false);
        }
    }, [listeningLevel]);

    const handleToggleListeningAudio = useCallback(() => {
        if (isListeningAudioPlaying) {
            if (listeningAudioSourceRef.current) {
                try { listeningAudioSourceRef.current.stop(); } catch (e) { /* ignore */ }
                // onended callback will handle state changes
            }
        } else {
            if (!listeningAudioBuffer) return;
            if (!audioContextsRef.current?.output || audioContextsRef.current.output.state === 'closed') {
                audioContextsRef.current = { ...audioContextsRef.current, output: new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: OUTPUT_SAMPLE_RATE }) };
            }
            const outputAudioContext = audioContextsRef.current.output;
            const sourceNode = outputAudioContext.createBufferSource();
            sourceNode.buffer = listeningAudioBuffer;
            sourceNode.connect(outputAudioContext.destination);
            sourceNode.onended = () => {
                setIsListeningAudioPlaying(false);
                listeningAudioSourceRef.current = null;
            };
            sourceNode.start();
            listeningAudioSourceRef.current = sourceNode;
            setIsListeningAudioPlaying(true);
        }
    }, [isListeningAudioPlaying, listeningAudioBuffer]);

    useEffect(() => {
        return () => { stopSession(); };
    }, [stopSession]);

    useEffect(() => {
        // Full cleanup when mode changes
        stopSession();
        setTranscript([]);
        setExercise(null);
        setListeningAudioBuffer(null);
        setShowListeningTranscript(false);
    }, [mode, stopSession]);


    return (
        <div className="h-screen w-screen bg-slate-900 text-slate-100 flex flex-col font-sans overflow-hidden">
            <header className="text-center p-4 border-b border-slate-700/50">
                <h1 className="text-2xl md:text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">
                    English Language Practice
                </h1>
            </header>

            <main className="flex-grow flex flex-col p-4 gap-4 overflow-hidden">
                {mode === AppMode.CONVERSATION && (
                    <div className="flex-grow bg-slate-800/50 rounded-2xl flex flex-col overflow-hidden border border-slate-700">
                        <TranscriptView transcript={transcript} />
                    </div>
                )}
                {mode === AppMode.LISTENING && (
                    <div className="flex-grow bg-slate-800/50 rounded-2xl flex flex-col overflow-y-auto p-4 md:p-6 border border-slate-700">
                        {isGenerating && (
                            <div className="m-auto text-center text-slate-400">
                                <LoadingSpinner className="w-12 h-12 mx-auto mb-4" />
                                <p className="text-lg">Generating your exercise...</p>
                                <p className="text-sm text-slate-500">This may take a moment.</p>
                            </div>
                        )}
                        {!isGenerating && !exercise && (
                            <div className="m-auto text-center text-slate-400">
                                <h2 className="text-2xl font-bold text-slate-200 mb-2">Listening Practice</h2>
                                <p>Select a level and click "Generate Exercise" to begin.</p>
                            </div>
                        )}
                        {exercise && (
                            <div className="space-y-8 animate-fade-in">
                                <div>
                                    <h3 className="text-xl font-bold text-cyan-400 mb-4">Listen to the Conversation</h3>
                                    <div className="flex items-center gap-4 p-4 bg-slate-900/50 rounded-lg">
                                        <button
                                            onClick={handleToggleListeningAudio}
                                            disabled={!listeningAudioBuffer}
                                            className="flex items-center justify-center gap-3 px-5 py-3 w-40 bg-cyan-600 text-white font-semibold rounded-lg shadow-md hover:bg-cyan-700 disabled:bg-slate-600 disabled:cursor-not-allowed transition-all duration-300"
                                        >
                                            {isListeningAudioPlaying ? <StopIcon className="w-5 h-5" /> : <PlayIcon className="w-5 h-5" />}
                                            <span>{isListeningAudioPlaying ? 'Stop' : 'Play Audio'}</span>
                                        </button>
                                        {!listeningAudioBuffer && <div className="flex items-center gap-2 text-slate-400"><LoadingSpinner className="w-5 h-5" /> <span>Processing audio...</span></div>}
                                    </div>
                                </div>
                                <div>
                                    <h3 className="text-xl font-bold text-cyan-400 mb-2">Comprehension Questions</h3>
                                    <div className="p-4 bg-slate-900/50 rounded-lg whitespace-pre-wrap text-slate-200 leading-relaxed">{exercise.questions}</div>
                                </div>
                                <div>
                                    <button onClick={() => setShowListeningTranscript(s => !s)} className="text-cyan-400 hover:text-cyan-300 font-semibold">
                                        {showListeningTranscript ? 'Hide' : 'Show'} Transcript
                                    </button>
                                    {showListeningTranscript && (
                                        <div className="mt-2 p-4 bg-slate-900/50 rounded-lg whitespace-pre-wrap text-slate-300 font-mono text-sm leading-relaxed animate-fade-in">
                                            {exercise.transcript}
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                )}

                <div className="flex-shrink-0 flex flex-col gap-4">
                    <ModeSelector mode={mode} setMode={setMode} />
                    {mode === AppMode.CONVERSATION && (
                        <ConversationControls
                            status={status}
                            topic={topic}
                            level={level}
                            onTopicChange={setTopic}
                            onLevelChange={setLevel}
                            onStart={startSession}
                            onStop={stopSession}
                        />
                    )}
                    {mode === AppMode.LISTENING && (
                        <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl p-4 shadow-2xl border border-slate-700 flex flex-col md:flex-row items-center gap-4">
                            <LevelSelector level={listeningLevel} setLevel={setListeningLevel} disabled={isGenerating} />
                            <div className="flex-grow" />
                            <button
                                onClick={generateListeningExercise}
                                disabled={isGenerating}
                                className="w-full md:w-auto flex items-center justify-center gap-2 px-6 py-3 bg-purple-600 text-white font-semibold rounded-lg shadow-md hover:bg-purple-700 disabled:bg-slate-600 disabled:cursor-not-allowed transition-all duration-300"
                            >
                                {isGenerating ? <LoadingSpinner className="w-5 h-5" /> : <SparklesIcon className="w-5 h-5" />}
                                <span>{isGenerating ? 'Generating...' : 'Generate Exercise'}</span>
                            </button>
                        </div>
                    )}
                </div>
            </main>
        </div>
    );
}