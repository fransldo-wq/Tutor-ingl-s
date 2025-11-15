export enum Speaker {
  USER = 'You',
  TUTOR = 'Tutor',
}

export interface TranscriptEntry {
  speaker: Speaker;
  text: string;
  correction?: string;
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
