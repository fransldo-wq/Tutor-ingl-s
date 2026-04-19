class AudioProcessor extends AudioWorkletProcessor {
    process(inputs, outputs, parameters) {
        // Tomamos la entrada del micrófono (el primer canal)
        const input = inputs[0];
        
        if (input && input.length > 0) {
            const channelData = input[0];
            // Enviamos los datos de audio crudos a nuestra aplicación principal
            this.port.postMessage(channelData);
        }
        
        // Devolvemos true para mantener el procesador vivo y escuchando
        return true;
    }
}

// Registramos este procesador con un nombre para poder llamarlo desde React
registerProcessor('audio-processor', AudioProcessor);
