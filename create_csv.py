open('train_data.csv', 'w').close()
with open("train_data.csv", "a") as csv:
    csv.write("sentiment, sentence\n")
    with open("train_data.txt") as f:
        for line in f:
            csv_line = line.replace(' ', ',', 1)
            csv.write(csv_line)

