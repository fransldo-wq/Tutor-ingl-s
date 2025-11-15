

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { GoogleGenAI, LiveSession, LiveServerMessage, Modality, Blob, Type } from '@google/genai';
import { Speaker, TranscriptEntry, SessionStatus, AppMode } from './types';
import { encode, decode, decodeAudioData } from './utils/audio';
import { MicIcon, StopIcon, LoadingSpinner, SparklesIcon, PlayIcon } from './components/Icons';

// --- Helper Functions & Constants ---
const INPUT_SAMPLE_RATE = 16000;
const OUTPUT_SAMPLE_RATE = 24000;
const SCRIPT_PROCESSOR_BUFFER_SIZE = 4096;

const LISTENING_TOPICS = [
    // Technology & Science
    'a recent breakthrough in artificial intelligence',
    'the ethical implications of gene editing',
    'exploring the possibility of life on Mars',
    'the impact of social media on society',
    'a debate on renewable energy sources',
    'the future of transportation, like self-driving cars',
    'how quantum computing works',

    // Culture & Arts
    'a review of a critically acclaimed film or TV series',
    'the history of a specific music genre, like Jazz or Hip Hop',
    'a discussion about a famous painting or artist',
    'the experience of visiting a world-renowned museum',
    'a conversation about the importance of literature',
    'the process of writing a novel',

    // Travel & Lifestyle
    'planning a backpacking trip through Southeast Asia',
    'the challenges of living as an expatriate',
    'a story about an unexpected travel adventure',
    'comparing city life versus country life',
    'a discussion on minimalism and simple living',
    'sharing a unique recipe and its cultural origin',

    // Personal & Professional Development
    'a dialogue about effective public speaking techniques',
    'the importance of emotional intelligence in the workplace',
    'a debate on different learning styles',
    'setting and achieving long-term career goals',
    'a conversation about managing stress and preventing burnout',
    'the benefits of learning a new language',

    // Society & History
    'a discussion about a pivotal moment in modern history',
    'the challenges of urbanization in developing countries',
    'a debate on the pros and cons of globalization',
    'an interview with a community leader about local issues',
    'a short story inspired by a historical event',
    'exploring different cultural traditions around the world',
];

const CONVERSATION_TUTOR_SYSTEM_INSTRUCTION = `You are a friendly and helpful English language tutor. The user is a {LEVEL} level English learner who wants to have a conversation about "{TOPIC}".

Your primary goal is to keep the conversation flowing naturally. After the user speaks, you MUST follow this process in your response:

1.  **Correction:** Analyze the user's previous statement for grammatical errors, awkward phrasing, or unnatural sentences. If you find any errors, your response MUST begin *exactly* with the phrase "Correction:" followed by the corrected version of the user's sentence.
2.  **Reply Separator:** After the correction sentence, you MUST include the separator token "||".
3.  **Reply:** After the separator, provide your natural, conversational reply.

If there are no errors, just provide the conversational reply without the "Correction:" prefix or the "||" separator.

Example 1 (with correction):
User: "I goed to the store yesterday."
Your response text: "Correction: I went to the store yesterday.||Oh, what did you buy?"

Example 2 (no correction):
User: "I went to the store yesterday."
Your response text: "Oh, what did you buy?"

Keep your replies concise to encourage the user to speak more.`;


const LISTENING_SYSTEM_INSTRUCTION = `You are an AI assistant for English language learners. Your task is to generate a listening comprehension exercise.

Process:
1.  On a random, engaging topic, create a realistic dialogue (approx. 200-300 words) appropriate for the selected level.
2.  The dialogue MUST feature exactly two distinct speakers (e.g., Alice and Bob).
3.  The dialogue text MUST be formatted with speaker tags, where each line of dialogue is prefixed with the speaker's name in brackets. Example: "[Alice]: Hi Bob, how are you? [Bob]: I'm doing great, thanks!".
4.  After the dialogue, generate 3-4 multiple-choice comprehension questions (A, B, C) based on the dialogue. Do not provide the answers.
`;


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
                                <span>Starting...</span>
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
    isTutorReplying: boolean;
    status: SessionStatus;
    liveUserTranscript: string;
}

const TranscriptView: React.FC<TranscriptViewProps> = ({ transcript, isTutorReplying, status, liveUserTranscript }) => {
    const endOfMessagesRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [transcript, isTutorReplying, liveUserTranscript]);

    return (
        <div className="flex-grow space-y-6 overflow-y-auto p-4 md:p-6">
            {transcript.length === 0 && !isTutorReplying && status !== SessionStatus.ACTIVE && status !== SessionStatus.CONNECTING && !liveUserTranscript && (
                <div className="text-center text-slate-400 py-10">
                    <h2 className="text-2xl font-bold text-slate-200 mb-2">Welcome!</h2>
                    <p>Enter a topic, select a level, and click "Start" to begin your practice session.</p>
                </div>
            )}
             {status === SessionStatus.ACTIVE && transcript.length === 0 && !isTutorReplying && !liveUserTranscript && (
                 <div className="text-center text-slate-400 py-10 animate-pulse">
                    <MicIcon className="w-12 h-12 mx-auto text-cyan-400 mb-4" />
                    <h2 className="text-2xl font-bold text-slate-200 mb-2">Listening...</h2>
                    <p>Please start by saying something about the topic.</p>
                </div>
            )}
            {transcript.map((entry, index) => (
                <div key={index} className={`flex items-start gap-4 ${entry.speaker === Speaker.USER ? 'flex-row-reverse' : 'flex-row'}`}>
                    <div className={`max-w-xl p-4 rounded-2xl shadow-md ${entry.speaker === Speaker.USER ? 'bg-blue-600 text-white rounded-br-none' : 'bg-slate-700 text-slate-200 rounded-bl-none'}`}>
                        <p className="font-bold mb-1">{entry.speaker}</p>
                        {entry.speaker === Speaker.TUTOR && entry.correction && (
                            <div className="mb-3 p-3 bg-slate-600/50 rounded-lg border border-slate-500/80">
                                <p className="text-sm font-semibold text-yellow-400 mb-1">Correction</p>
                                <p className="whitespace-pre-wrap italic text-slate-300">"{entry.correction}"</p>
                            </div>
                        )}
                        <p className="whitespace-pre-wrap">{entry.text}</p>
                    </div>
                </div>
            ))}
            {liveUserTranscript && (
                <div className="flex items-start gap-4 flex-row-reverse animate-fade-in">
                    <div className="max-w-xl p-4 rounded-2xl shadow-md bg-blue-600/80 text-white rounded-br-none">
                        <p className="font-bold mb-1">{Speaker.USER}</p>
                        <p className="whitespace-pre-wrap italic">{liveUserTranscript}...</p>
                    </div>
                </div>
            )}
            {isTutorReplying && (
                <div className="flex items-start gap-4 flex-row">
                    <div className="max-w-xl p-4 rounded-2xl shadow-md bg-slate-700 text-slate-200 rounded-bl-none">
                        <p className="font-bold mb-1">{Speaker.TUTOR}</p>
                        <div className="flex items-center gap-2 text-slate-400">
                            <LoadingSpinner className="w-4 h-4" />
                            <span>Thinking...</span>
                        </div>
                    </div>
                </div>
            )}
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
    const [isTutorReplying, setIsTutorReplying] = useState<boolean>(false);
    const [liveUserTranscript, setLiveUserTranscript] = useState<string>('');

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
    const audioContextsRef = useRef<{ input?: AudioContext; output?: AudioContext }>({});
    const audioPlaybackQueueRef = useRef<{ nextStartTime: number, sources: Set<AudioBufferSourceNode> }>({ nextStartTime: 0, sources: new Set() });
    const listeningAudioSourceRef = useRef<AudioBufferSourceNode | null>(null);

    const currentInputTranscription = useRef<string>('');
    const currentOutputTranscription = useRef<string>('');

     // --- Robust AudioContext Management ---
    const getOutputAudioContext = useCallback(() => {
        let context = audioContextsRef.current.output;
        if (!context || context.state === 'closed') {
            context = new (window.AudioContext || (window as any).webkitAudioContext)({
                sampleRate: OUTPUT_SAMPLE_RATE,
            });
            audioContextsRef.current.output = context;
        }
        return context;
    }, []);

    // --- Core Logic ---
    const cleanupSessionResources = useCallback(() => {
        // Conversation-specific cleanup
        streamRef.current?.getTracks().forEach(track => track.stop());
        streamRef.current = null;
        audioProcessorRef.current?.disconnect();
        audioProcessorRef.current = null;
        audioPlaybackQueueRef.current.sources.forEach(source => {
            try { source.stop(); } catch (e) { /* ignore */ }
        });
        audioPlaybackQueueRef.current.sources.clear();
        audioPlaybackQueueRef.current.nextStartTime = 0;
        
        if (audioContextsRef.current.input) {
            audioContextsRef.current.input.close().catch(console.error);
            delete audioContextsRef.current.input;
        }
        
        // Listening-specific cleanup
        if (listeningAudioSourceRef.current) {
            try { listeningAudioSourceRef.current.stop(); } catch (e) { /* ignore */ }
            listeningAudioSourceRef.current = null;
        }
        setIsListeningAudioPlaying(false);
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
        cleanupSessionResources();
        setIsTutorReplying(false);
        setLiveUserTranscript('');
        setStatus(SessionStatus.INACTIVE);
    }, [cleanupSessionResources]);


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
        setLiveUserTranscript('');
        setStatus(SessionStatus.CONNECTING);
        
        try {
            // Step 1: Get microphone access first.
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            streamRef.current = stream;

            // Step 2: Initialize dependencies
            const ai = new GoogleGenAI({ apiKey });
            const outputAudioContext = getOutputAudioContext();
            const inputAudioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: INPUT_SAMPLE_RATE });
            audioContextsRef.current.input = inputAudioContext;

            audioPlaybackQueueRef.current = { nextStartTime: 0, sources: new Set() };

            const systemInstruction = CONVERSATION_TUTOR_SYSTEM_INSTRUCTION
                .replace('{TOPIC}', topic)
                .replace('{LEVEL}', level);

            // Step 3: Connect to the AI session.
            sessionPromiseRef.current = ai.live.connect({
                model: 'gemini-2.5-flash-native-audio-preview-09-2025',
                config: {
                    responseModalities: [Modality.AUDIO],
                    inputAudioTranscription: {},
                    outputAudioTranscription: {},
                    speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } } },
                    systemInstruction: systemInstruction,
                },
                callbacks: {
                    onopen: () => {
                        // Step 4: Once connected, start streaming microphone audio.
                        setStatus(SessionStatus.ACTIVE);
                        setIsTutorReplying(false);

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
                        if (message.serverContent?.inputTranscription) {
                            currentInputTranscription.current += message.serverContent.inputTranscription.text;
                            setLiveUserTranscript(currentInputTranscription.current);
                        }
                        if (message.serverContent?.outputTranscription) {
                            if (!isTutorReplying) setIsTutorReplying(true);
                            currentOutputTranscription.current += message.serverContent.outputTranscription.text;
                        }
                        if (message.serverContent?.turnComplete) {
                            const finalInput = currentInputTranscription.current.trim();
                            const finalOutput = currentOutputTranscription.current.trim();
                            
                            currentInputTranscription.current = '';
                            currentOutputTranscription.current = '';
                            setLiveUserTranscript('');
                            setIsTutorReplying(false);

                            setTranscript(prev => {
                                let newTranscript = [...prev];
                                if (finalInput) {
                                    newTranscript.push({ speaker: Speaker.USER, text: finalInput });
                                }
                                if (finalOutput) {
                                    const correctionPrefix = "Correction:";
                                    const separator = "||";
                                    let correctionText: string | undefined = undefined;
                                    let replyText: string;

                                    if (finalOutput.startsWith(correctionPrefix) && finalOutput.includes(separator)) {
                                        const parts = finalOutput.split(separator);
                                        correctionText = parts[0].substring(correctionPrefix.length).trim();
                                        replyText = parts.slice(1).join(separator).trim();
                                    } else {
                                        replyText = finalOutput;
                                    }
                                    
                                    if (replyText || correctionText) {
                                        newTranscript.push({ speaker: Speaker.TUTOR, text: replyText, correction: correctionText });
                                    }
                                }
                                return newTranscript;
                            });
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
                        cleanupSessionResources();
                    },
                },
            });
        } catch (error) {
            console.error('Failed to start session:', error);
            let errorMessage = 'Failed to start session.';
            if (error instanceof Error) {
                if (error.name === 'NotAllowedError' || error.message.includes('Permission denied')) {
                    errorMessage = 'Microphone permission was denied. Please allow microphone access to use this feature.';
                } else {
                    errorMessage = `Failed to start session: ${error.message}`;
                }
            }
            alert(errorMessage);
            cleanupSessionResources();
            setIsTutorReplying(false);
            setStatus(SessionStatus.INACTIVE);
        }
    }, [topic, level, cleanupSessionResources, stopSession, getOutputAudioContext]);

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
            const randomTopic = LISTENING_TOPICS[Math.floor(Math.random() * LISTENING_TOPICS.length)];
            const textPrompt = `Generate a listening comprehension exercise about ${randomTopic} for level ${listeningLevel}.`;

            const textResponse = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: textPrompt,
                config: {
                    systemInstruction: LISTENING_SYSTEM_INSTRUCTION,
                    responseMimeType: "application/json",
                    responseSchema: {
                        type: Type.OBJECT,
                        properties: {
                            transcript: {
                                type: Type.STRING,
                                description: "The full dialogue script between two speakers. CRITICAL: The response MUST follow the format '[SpeakerName]: Dialogue text.'. For example: '[Alice]: Did you see the news today? [Bob]: No, what happened?'",
                            },
                            questions: {
                                type: Type.STRING,
                                description: "The multiple-choice comprehension questions based on the dialogue.",
                            },
                        },
                        required: ['transcript', 'questions'],
                    },
                },
            });

            const responseJson = JSON.parse(textResponse.text);

            if (!responseJson.transcript || !responseJson.questions) {
                throw new Error("Failed to parse the exercise from the AI's response. The JSON structure is incorrect.");
            }
            const transcriptText = responseJson.transcript.trim();
            const questionsText = responseJson.questions.trim();
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
            const outputAudioContext = getOutputAudioContext();
            const audioBuffer = await decodeAudioData(decode(base64Audio), outputAudioContext, OUTPUT_SAMPLE_RATE, 1);
            setListeningAudioBuffer(audioBuffer);

        } catch (error) {
            console.error("Failed to generate exercise:", error);
            alert(`Sorry, there was an error generating the exercise: ${error instanceof Error ? error.message : String(error)}`);
            setExercise(null);
        } finally {
            setIsGenerating(false);
        }
    }, [listeningLevel, getOutputAudioContext]);

    const handleToggleListeningAudio = useCallback(() => {
        if (isListeningAudioPlaying) {
            if (listeningAudioSourceRef.current) {
                try { listeningAudioSourceRef.current.stop(); } catch (e) { /* ignore */ }
                // onended callback will handle state changes
            }
        } else {
            if (!listeningAudioBuffer) return;
            
            const outputAudioContext = getOutputAudioContext();
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
    }, [isListeningAudioPlaying, listeningAudioBuffer, getOutputAudioContext]);

    useEffect(() => {
        // This effect runs when the component unmounts
        return () => {
            stopSession();
            // Final cleanup of the persistent output context
            if (audioContextsRef.current.output) {
                audioContextsRef.current.output.close().catch(console.error);
            }
        };
    }, [stopSession]);

    useEffect(() => {
        // Full cleanup when mode changes
        stopSession();
        setTranscript([]);
        setLiveUserTranscript('');
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
                        <TranscriptView transcript={transcript} isTutorReplying={isTutorReplying} status={status} liveUserTranscript={liveUserTranscript} />
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
