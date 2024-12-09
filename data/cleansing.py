import json


def read_data(filename):
    """
    Read data from a JSON file and return it as a list of dictionaries.

    Parameters:
    - filename (str): The name of the file to read.

    Returns:
    - data (list): A list of dictionaries containing the data.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def count_genres(data):
    """

    Count and prints the number of movies in each genre and print the results.

    Parameters:
    - data (list): A list of dictionaries containing the data.

    Returns:
    - genres_count (dict): A dictionary containing the count of movies in each genre.

    """

    genres_count = {}

    for movie in data:
        if 'genre' in movie:
            genre = movie['genre']
            if genre is None:
                continue
            elif genre in genres_count:
                genres_count[genre] += 1
            else:
                genres_count[genre] = 1

    for genre, count in genres_count.items():
        print(f"{genre:<20} {count}")

    return genres_count


def get_most_popular_genres(genres_count, count):
    """

    Find the X genres with the most movies and print the results.

    Parameters:
    - genres_count (dict): A dictionary containing the count of movies in each genre.
    - count (int): The number of genres to return.

    Returns:
    - popular_genres (list): A list of the ten genres with the most movies.

    """

    # Find the ten genres with the most movies
    popular_genres = sorted(
        genres_count, key=genres_count.get, reverse=True)[:count]

    for genre in popular_genres:
        print(f"{genre:<20} {genres_count[genre]}")

    return popular_genres


def filter_movies_by_genre(filename, genres):
    """

    Filter the movies by genre and return the filtered movies.

    Parameters:
    - filename (str): The name of the file to read.
    - genres (list): A list of genres to keep.

    Returns:
    - filtered_movies (list): A list of dictionaries containing the filtered data.

    """
    with open(filename, 'r') as f:
        movies = json.load(f)

    filtered_movies = []
    for movie in movies:
        if 'genre' in movie and movie['genre'] in genres:
            filtered_movies.append(movie)

    return filtered_movies


def remove_duplicates(data):
    """
    Remove duplicates from a list of dictionaries and return the deduplicated data.

    Parameters:
    - data (list): A list of dictionaries containing the data.

    Returns:
    - deduplicated_data (list): A list of dictionaries containing the deduplicated data.

    """
    count = 0
    no_desc = 0

    unique_descriptions = set()
    res = []

    for item in data:
        desc = item['overview']

        if desc not in unique_descriptions:
            res.append(item)
            unique_descriptions.add(desc)
        elif desc != "No overview found.":
            count += 1
        else:
            no_desc += 1

    print(f"\nNumber of duplicates removed: {count}")
    print(f"Number of 'No overview found' removed: {no_desc}\n")

    return res


def balance_genres(movies, amount, genres):
    """

    Balance the number of movies per genre and return the balanced movies.

    Parameters:
    - movies (list): A list of dictionaries containing the data.
    - amount (int): The number of movies to keep per genre.
    - genres (list): A list of genres to balance.

    Returns:
    - balanced_movies (list): A list of dictionaries containing the balanced data.

    """

    balanced_movies = []
    for genre in genres:
        genre_movies = [movie for movie in movies if movie['genre'] == genre]
        balanced_movies.extend(genre_movies[:amount])

    return balanced_movies


def stratify_genres(movies, total_movies, genres_count):
    """

    Stratify the movies to balance the proportions of genres and return the stratified movies.

    Parameters:
    - movies (list): A list of dictionaries containing the data.
    - total_movies (int): The total number of movies to keep.
    - genres_count (dict): A dictionary containing the count of movies in each genre.

    Returns:
    - stratified_movies (list): A list of dictionaries containing the stratified data.

    """

    # Calculate the total current movies and scaling factor
    current_total = sum(genres_count.values())
    scaling_factor = total_movies / current_total

    stratified_movies = []
    for genre, count in genres_count.items():
        # Calculate the number of movies to keep for each genre
        target_count = max(1, int(count * scaling_factor))
        genre_movies = [movie for movie in movies if movie['genre'] == genre]
        stratified_movies.extend(genre_movies[:target_count])

    return stratified_movies


def calculate_overview_length(data):
    count = []

    for movie in data:
        count.append(len(movie['overview'].split(" ")))

    print(f"\nAverage amount of words in overview: {sum(count)/len(count)}")


def write_data(data, filename):
    """

    Write data to a JSON file.

    Parameters:
    - data (list): A list of dictionaries containing the data.
    - filename (str): The name of the file to write.

    """

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def main():
    DATA_FILE = 'movies.json'
    AMOUNT_OF_GENRES = 10
    MOVIES_PER_GENRE = 1000

    print("\nCleansing data...")

    data = read_data(DATA_FILE)
    print(f"Number of movies in data: {len(data)}")

    print("\nNumber of movies per genre:")
    genres_count = count_genres(data)

    most_popular_genres = get_most_popular_genres(
        genres_count, AMOUNT_OF_GENRES)

    # Keep only the movies that belong to one of the most popular genres
    filtered_movies = filter_movies_by_genre(DATA_FILE, most_popular_genres)

    # Remove duplicates
    filtered_movies = remove_duplicates(filtered_movies)

    # Balance the number of movies per genre
    balanced_movies = balance_genres(
        filtered_movies, MOVIES_PER_GENRE, most_popular_genres)

    # Count genres in filtered dataset
    filtered_genres_count = count_genres(filtered_movies)

    # Stratify the movies to balance proportions
    stratified_movies = stratify_genres(
        filtered_movies, MOVIES_PER_GENRE*AMOUNT_OF_GENRES, filtered_genres_count)

    print("\nNumber of movies in filtered: ", len(filtered_movies))
    count_genres(filtered_movies)

    print(
        f"\nNumber of movies in balanced: {len(balanced_movies)}")
    count_genres(balanced_movies)

    print(
        f"\nNumber of movies in stratified: {len(stratified_movies)}")
    count_genres(stratified_movies)

    write_data(stratified_movies, 'movies_stratified.json')
    write_data(balanced_movies, 'movies_balanced.json')
    write_data(most_popular_genres, 'popular_genres.json')

    calculate_overview_length(filtered_movies)


main()
