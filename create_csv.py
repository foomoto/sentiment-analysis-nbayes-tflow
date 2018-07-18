open('train_data.csv', 'w').close()
with open("train_data.csv", "a") as csv:
    csv.write("sentiment, sentence\n")
    with open("train_data.txt") as f:
        for line in f:
            line = line.replace('\n', '')
            sentiment = '"' + line.split(" ", 1)[0] + '"'
            text = '"' + line.split(" ", 1)[1] + '"'
            csv.write(sentiment + ',' + text + '\n')

