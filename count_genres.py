import json

genre_id_to_name = {
    "Action": 0,
    "Adventure": 0,
    "Animation": 0,
    "Comedy": 0,
    "Crime": 0,
    "Documentary": 0,
    "Drama": 0,
    "Family": 0,
    "Fantasy": 0,
    "History": 0,
    "Horror": 0,
    "Music": 0,
    "Mystery": 0,
    "Romance": 0,
    "Science Fiction": 0,
    "TV Movie": 0,
    "Thriller": 0,
    "War": 0,
    "Western": 0,
}


def count_genres(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    for movie in data:
        if 'genre' in movie:
            genre = movie['genre']
            if genre in genre_id_to_name:
                genre_id_to_name[genre] += 1

    for genre, count in genre_id_to_name.items():
        print(f"{genre}: {count}")

    print("-----------------")

    # Find the ten genres with the most movies
    popular_genres = sorted(
        genre_id_to_name, key=genre_id_to_name.get, reverse=True)[:10]

    for genre in popular_genres:
        print(f"{genre}: {genre_id_to_name[genre]}")

    return popular_genres


def filter_movies(filename, output_filename, genres):
    with open(filename, 'r') as f:
        movies = json.load(f)

    filtered_movies = []
    for movie in movies:
        if 'genre' in movie and movie['genre'] in genres:
            filtered_movies.append(movie)

    with open(output_filename, 'w') as f:
        json.dump(filtered_movies, f, indent=4)


popular = count_genres('unique_movies.json')

# filter_movies('movies.json', 'popular_movies.json', popular)
