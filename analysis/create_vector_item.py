import numpy as np


def read_dictionaries():
    dict_path = "imdb/words.csv"
    count = 0

    d = {}
    for line in open(dict_path):
        word = line.strip()
        if word in d:
            print("fuck")

        d[word] = count
        count += 1

    return d


def converter_to_vector():
    d = read_dictionaries()
    data_analyzed_path = "imdb/plot-summaries-analyzed"
    data_vector_path = "imdb/movie-vectors"

    f = open(data_vector_path, "w")
    for line in open(data_analyzed_path):
        args = line.split(",")
        words = args[3].split()

        # create vector
        vector = {}
        for word in words:
            k = d[word]
            if k not in vector.keys():
                vector[k] = 0
            vector[k] += 1

        content = ""
        for key in vector.keys():
            content += "%d:%d " % (key, vector[key])

        f.write("%s,%s\n" % (args[0], content.strip()))


if __name__ == "__main__":
    converter_to_vector()
