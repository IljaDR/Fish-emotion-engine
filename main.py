import csv


def main():
    print("Enter a sentence...")
    # load lexicon into a dictionary
    with open('emotion-lexicon.csv') as csv_lexicon:
        reader = csv.reader(csv_lexicon)
        lexicon = {rows[0]: rows[1] for rows in reader}
    while True:
        # get user input
        sentence = input()

        # set default sentiment to neutral (this is what'll get returned if no sentiment is found in user input)
        sentiment = "neutral"

        # split user input into a list of words
        sentence = sentence.split(' ')

        # for each word
        for word in sentence:
            # check if word exists in lexicon
            if word in lexicon:
                # if so, set sentiment to whatever sentiment corresponds to the word and exit the loop
                sentiment = lexicon[word]
                break
        print(sentiment)


if __name__ == '__main__':
    main()
