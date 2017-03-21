from scipy.sparse import dok_matrix
from analysis.create_vector_item import read_dictionaries


def read_vectors():
    d = read_dictionaries()
    num_words = len(d)
    path = "imdb/movie-vectors"

    print("Number of words %d " % num_words)

    num_movies = 0
    for line in open(path):
        args = line.split(",")
        movie_id = float(args[0])

        if movie_id > num_movies:
            num_movies = movie_id

    print("Number of movies %d" % num_movies)

    mat = dok_matrix((num_movies, num_words), dtype=float)
    for line in open(path):
        args = line.split(",")
        movie_id = float(args[0])

        for item in args[1].split():
            temp = item.split(":")
            word_id = int(temp[0])
            value = float(temp[1])

            mat[movie_id - 1, word_id] = value

    return mat


if __name__ == "__main__":
    d = read_dictionaries()

    a = {}

    for k in d.keys():
        a[d[k]] = k

    for x in read_vectors()[0].keys():
        print(a[x[1]])
