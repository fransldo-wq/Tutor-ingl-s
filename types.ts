export enum Speaker {
  USER = 'You',
  TUTOR = 'Tutor',
}

export interface Correction {
  original: string;
  corrected: string;
  explanation: string;
}

export interface TranscriptEntry {
  speaker: Speaker;
  text: string;
  correction?: Correction;
}

export enum SessionStatus {
    INACTIVE = 'INACTIVE',
    CONNECTING = 'CONNECTING',
    ACTIVE = 'ACTIVE',
    ERROR = 'ERROR',
}

export enum AppMode {
    CONVERSATION = 'CONVERSATION',
    LISTENING = 'LISTENING',
}
