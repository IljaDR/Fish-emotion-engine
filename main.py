import csv


def main():
    with open('emotion-lexicon.csv') as csv_lexicon:
        reader = csv.reader(csv_lexicon)
        lexicon = {rows[0]: rows[1] for rows in reader}
    while True:
        sentence = input()
        sentiment = "neutral"
        sentence = sentence.split(' ')
        for word in sentence:
            if word in lexicon:
                sentiment = lexicon[word]
        print(sentiment)


if __name__ == '__main__':
    main()
