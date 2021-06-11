import csv
import datetime
import os
import socket
from datetime import datetime
from threading import Thread
from queue import Queue
import speech_recognition as sr
import torch
import pickle
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from pandas import Series
from pathlib import Path
from tensorflow.keras.models import load_model
from nlpp.utils import preprocess
from tensorflow.keras.preprocessing.sequence import pad_sequences

def BERT_validate_emo(model, tokenizer, utterances, max_length=0):
    # this function and its model are made by Alexa

    # Tokenize the utterances
    input_ids = []
    for utt in utterances:
        encoded_sent = tokenizer.encode(
            utt,                        # utterances to encode
            add_special_tokens = True,  # Add '[CLS]' and '[SEP]'
        )
        input_ids.append(encoded_sent)

    # pad the encoded input_ids
    if max_length == 0:
        max_length = len(encoded_sent)

    input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long",
                              value=0, truncating="post", padding="post")

    # Create attention masks
    attention_masks = []
    # For each utterance...
    for sent in input_ids:

        # Create the attention mask.
        #   - If token ID is 0, then it is padding => set mask to 0
        #   - If token ID is > 0, then it is a real token => set mask to 1
        att_mask = [float(token_id > 0) for token_id in sent]

        # Store attention mask for utterance
        attention_masks.append(att_mask)

        # Convert all inputs into torch tensors
    validation_inputs = torch.tensor(input_ids)
    validation_masks = torch.tensor(attention_masks)

    model.eval()

    with torch.no_grad():
        outputs = model(validation_inputs,
                        token_type_ids=None,
                        attention_mask=validation_masks)
    logits = outputs[0]
    logits_flat = np.argmax(logits, axis=1).flatten()
    emotion = logits_flat[0].item()

    return emotion

def BERT_emotion_int_to_str(emo_int):
    if emo_int == 0:
        return 'sadness'
    elif emo_int == 1:
        return 'neutral'
    elif emo_int == 2:
        return 'anger'
    elif emo_int == 3:
        return 'fear'
    elif emo_int == 4:
        return 'joy'
    else:
        return 'unknown'

def LSTM_get_tokenizer_and_encoder(tokenizer_path, encoder_path):
    with tokenizer_path.open('rb') as file:
        tokenizer = pickle.load(file)

    with encoder_path.open('rb') as file:
        encoder = pickle.load(file)

    return tokenizer, encoder

print("Initializing BERT")
# We will run the BERT model on the CPU

device = torch.device("cpu")
# Load previously saved tokenizer
BERT_tokenizer_path = 'BERT_model'

BERT_tokenizer = BertTokenizer.from_pretrained(BERT_tokenizer_path, do_lower_case=True)
# Load previously saved model
BERT_model_path = 'BERT_model'
BERT_model = BertForSequenceClassification.from_pretrained(BERT_model_path)

print("Finished initializing BERT")

print("Initializing LSTM_model")
LSTM_tokenizer_path = Path('LSTM_model/tokenizer.pickle').resolve()
LSTM_encoder_path = Path('LSTM_model/encoder.pickle').resolve()
tokenizer, encoder = LSTM_get_tokenizer_and_encoder(LSTM_tokenizer_path, LSTM_encoder_path)

LSTM_model = load_model('LSTM_model/model.h5', compile=False)
print("Finished initializing LSTM_model\nWaiting for client...")

HOST, PORT = "localhost", 10000
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)

r = sr.Recognizer()
audio_queue = Queue()

def recognize_worker(conn, lexicon, negations):
    # this runs in a background thread
    while True:
        audio = audio_queue.get()
        if audio is None: break

        # received audio data, now we'll recognize it using Google Speech Recognition
        try:
            sentence = r.recognize_google(audio)
            print(sentence)

            # set default emotion to neutral (this is what'll get returned if no emotion is found in user input)
            emotion = "neutral"
            # split user input into a list of words
            split_sentence = sentence.split(' ')
            i = 0
            while i < len(split_sentence):
                # check if current word is in negation
                if split_sentence[i] in negations:
                    # check to prevent IndexError
                    if i+1 == len(split_sentence):
                        break
                    # check if the negation is one or two words, then move the iterator 2 or 3 places respectively
                    if '%s %s' % (split_sentence[i], split_sentence[i+1]) in negations:
                        i += 3
                    else:
                        i += 2
                # check to prevent IndexError
                if i < len(split_sentence):
                    if split_sentence[i] in lexicon:
                        # if so, set emotion to whatever emotion corresponds to the word and exit the loop
                        emotion = lexicon[split_sentence[i]]
                        break
                i += 1
            print("Rule based system result: %s" % emotion)
            conn.sendall(bytes(emotion, encoding="utf-8"))

            home = str(Path.home())
            path = os.path.join(home, 'fish\\emotions.csv')
            with open(path, 'a+', encoding='UTF8', newline='') as f:
                # create the csv writer
                writer = csv.writer(f)

                # BERT
                BERT_emotion = BERT_validate_emo(model=BERT_model, tokenizer=BERT_tokenizer, utterances=[sentence], max_length=100)
                print("BERT's result: %s" % BERT_emotion_int_to_str(BERT_emotion))
                writer.writerow([datetime.now(), 'BERT', BERT_emotion_int_to_str(BERT_emotion)])

                # LSTM
                sequence = Series(emotion)
                sequence = preprocess(sequence)
                list_tokenized = tokenizer.texts_to_sequences(sequence)
                sequence = pad_sequences(list_tokenized, maxlen=100)
                predictions = LSTM_model.predict(sequence)
                pred = predictions.argmax(axis=1)
                print("LSTM_model's result: %s" % encoder.classes_[pred[0]])
                writer.writerow([datetime.now(), 'LSTM_model', encoder.classes_[pred[0]]])

        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

        audio_queue.task_done()  # mark the audio processing job as completed in the queue

def main():
    # load lexicon into a dictionary
    with open('emotion-lexicon.csv') as csv_lexicon:
        reader = csv.reader(csv_lexicon)
        lexicon = {rows[0]: rows[1] for rows in reader}

    # load negation list into a list
    with open('negation-list.csv') as csv_negations:
        reader = csv.reader(csv_negations)
        negations = {rows[0] for rows in reader}
    while True:
        conn, addr = s.accept()
        print('Connected to %s\nEnter a sentence...' % str(addr))
        recognize_thread = Thread(target=recognize_worker, args=(conn,lexicon,negations))
        recognize_thread.daemon = True
        recognize_thread.start()

        with sr.Microphone() as source:
            try:
                while True:  # repeatedly listen for phrases and put the resulting audio on the audio processing job queue
                    audio_queue.put(r.listen(source))
            except KeyboardInterrupt:  # allow Ctrl + C to shut down the program
                pass

        audio_queue.join()
        audio_queue.put(None)
        recognize_thread.join()


if __name__ == '__main__':
    main()
