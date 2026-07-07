Our NLP model, a Seq2Seq architecture known as MacSeq, parses the intent from the text transcribed live by OpenAI Whisper,
allowing for our model to understand the user and map their speech to certain sequence of actions/commands,
such as controlling Mario in a Mario Bros emulator. Our main innovation is in creating a separate execution memory layer on top of the model,
allowing for the resolution of an issue that comes with OpenAI’s Whisper in the edge cases where commands are corrected once a recording window has ended.

### Pipeline:
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
