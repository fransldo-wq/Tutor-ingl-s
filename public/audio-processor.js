class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        // Le decimos que junte 4096 muestras de audio antes de molestar a la app principal
        this.bufferSize = 4096;
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
    }

    process(inputs) {
        const input = inputs[0];
        
        if (input && input.length > 0) {
            const channelData = input[0];
            
            // Vamos metiendo el sonido en nuestro "paquete" (buffer)
            for (let i = 0; i < channelData.length; i++) {
                this.buffer[this.bufferIndex] = channelData[i];
                this.bufferIndex++;
                
                // Cuando el paquete está lleno, lo enviamos a la IA y vaciamos el paquete
                if (this.bufferIndex >= this.bufferSize) {
                    // Hacemos una copia para no borrarlo mientras viaja
                    this.port.postMessage(new Float32Array(this.buffer));
                    this.bufferIndex = 0; 
                }
            }
        }
        
        return true; // Mantenemos el micrófono encendido
    }
}

registerProcessor('audio-processor', AudioProcessor);
