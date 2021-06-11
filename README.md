# Fish-emotion-engine
 Low tech live emotion analysis

### Instructions

Generate LSTM and BERT models (see NLP_LSTM_CNN and BERT directories), 
then copy contents of NLP_LSTM_CNN/models to integration/LSTM_model and 
contents of BERT/Alexa classifier/output_trained_model to integration/BERT_model.
Copy contents of NLP_LSTM_CNN/nlpp to integration/nlpp.

Install packages with pip install -r requirements.txt

To make this run with the Unity interface, run python main.py, wait until it prints 
"Waiting for client", then start Unity.
### Citations

Lexicon derived from NRC Word-Emotion Association Lexicon (EmoLex)

NRC Emotion Lexicon
https://saifmohammad.com

