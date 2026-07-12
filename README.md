# Macroroni

## Description
Our model utilizes MacBERT to extract user intent from live OpenAI Whisper transcriptions, translating speech into executable macro commands for a variety of fields, like controlling a Mario emulator. This novel pipeline bridges the gaps between GUI and VUI, featuring a novel memory layer that resolves live transcription boundaries and allows for real-time command corrections.

#### Library Link: [https://pypi.org/project/macroroni/](https://pypi.org/project/macroroni/)

## Contributors
- Nihaal Garud
- Sarthak Aggarwal
- Joshua Lopez
- Harshad Goswami

## Sources Used
| Link                                                                                                                                                                                                                             | Purpose                                                                                        | Person  |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------- |
| [https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string](https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string)                                                 | Removing punctuation from string                                                               | Nihaal  |
| [https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/](https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/)                                           | Saving model weights                                                                           | Nihaal  |
| [https://github.com/openai/whisper](https://github.com/openai/whisper)                                                                                                                                                           | Whisper usage                                                                                  | Nihaal  |
| [https://www.geeksforgeeks.org/machine-learning/understanding-tf-idf-term-frequency-inverse-document-frequency/](https://www.geeksforgeeks.org/machine-learning/understanding-tf-idf-term-frequency-inverse-document-frequency/) | Using TF-IDF vectorizers                                                                       | Nihaal  |
|https://www.kaggle.com/code/neilanshchauhan/multi-label-text-classification-using-distilbert                           | Kaggle Notebook for understanding how to code a distilbert for multi-label text classification | Sarthak |
|https://github.com/SimGus/Chatette?tab=readme-ov-file    | Used for creating dataset for NLP Bert model.                                                  | Sarthak |
|https://github.com/Kautenja/nes-py                   | Used to make emulator for Super Mario Bros                                                     | Joshua  |
| https://pypi.org/project/pynput/                                                                                                                                                                                                 | Used for initial mapping of the game controls                                                  | Joshua  |
| https://www.geeksforgeeks.org/computer-vision/essential-opencv-functions-to-get-started-into-computer-vision/                                                                                                                    | Used to render and resize the display for the emulator                                         | Joshua  |
| https://agombert.github.io/AdvancedNLPClasses/chapter3/Session_3_1_Word2Vec_Training/                                                                                                                                            | Vectorizer code largely based off this article                                                 | Nihaal  |
| https://gymnasium.farama.org/introduction/migration_guide/                                                                                                                                                                       | Used for more gym environment code in emulator                                                 | Joshua  |
| https://github.com/SYSTRAN/faster-whisper                                                                                                                                                                                        | Documentation on implementing faster whisper syntax                                            | Nihaal  |
| https://stackoverflow.com/questions/54174160/how-to-get-numpy-arrays-output-of-wav-file-format                                                                                                                                   | Ways to save audio to much faster numpy array                                                  | Nihaal  |
| https://huggingface.co/docs/transformers/en/tasks/token_classification                                                                                                                                                           | Token by Token BERT                                                                            | Sarthak |
