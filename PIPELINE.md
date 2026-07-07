STT --> Tokenizer --> Vectorizer/positional embeddings --> Seq2Seq model --> memory layer innovation --> vector of macro commands --> mapped to mario controls in emulator

### Done:
- STT: stt/FasterWhisper.py
- Tokenizer and Vectorizer: tokenvector/TokenVectorizer.py
- Mario emulator: emulator/mario.py

### To do:
- Seq2Seq model
  - Encoder module
  - Decoder module
  - Attention mechanism
- Memory layer innovation
- Mapping output vector of macro commands to controls in emulator
- Convert everything to OOP


### Latency fixes:
- Updating to faster-whisper with CUDA optimization technology
- Lower beam_size - only one word considered per time
- Debate: use tiny vs base Whisper model
