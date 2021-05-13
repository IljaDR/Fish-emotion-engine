import csv
import socket
import sys
import json

HOST, PORT = "localhost", 10000
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)


def main():
    print("Enter a sentence...")

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
        print(addr)
        while True:
            # get user input
            sentence = input()

            # set default emotion to neutral (this is what'll get returned if no emotion is found in user input)
            emotion = "neutral"

            # split user input into a list of words
            sentence = sentence.split(' ')

            i = 0
            while i < len(sentence):
                # check if current word is in negation
                if sentence[i] in negations:
                    # check to prevent IndexError
                    if i+1 == len(sentence):
                        break
                    # check if the negation is one or two words, then move the iterator 2 or 3 places respectively
                    if '%s %s' % (sentence[i], sentence[i+1]) in negations:
                        i += 3
                    else:
                        i += 2
                # check to prevent IndexError
                if i < len(sentence):
                    if sentence[i] in lexicon:
                        # if so, set emotion to whatever emotion corresponds to the word and exit the loop
                        emotion = lexicon[sentence[i]]
                        break
                i += 1
            conn.sendall(bytes(emotion,encoding="utf-8"))


if __name__ == '__main__':
    main()
