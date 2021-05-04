def loading_from_scratch():
    # loading files as list(movies) of lists(reviews)
    import csv
    movies = []
    with open(r"movies_data.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            movies.append((row[0], row[1]))
    return tuple(movies)


def size_check():
    count_0 = 0
    count_1 = 1
    movies = loading_from_scratch()
    for review in movies:
        if review[1] == "0":
            count_0 += 1
        elif review[1] == "1":
            count_1 += 1
    return (f"There is {count_0} reviews with label 0 and {count_1} reviews with label 1")

