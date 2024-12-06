import pandas as pd
import json  # For parsing JSON strings in the genres column


def main():
    """
    Load the movies_metadata.csv file, extract the overview and genres columns,
    parse the genres column to extract the first genre's name, and save the
    processed dataset to a JSON file.

    The processed dataset will contain only the overview and genre columns.

    """

    # Load data
    data = pd.read_csv('movies_metadata.csv')

    print(data.head())

    # Only keep overview and genres columns
    data = data[['overview', 'genres']]

    # Drop rows with missing values
    data = data.dropna()

    # Parse the genres column and extract the first genre's name
    def extract_first_genre(genres):
        try:
            genres_list = json.loads(
                genres.replace("'", '"'))  # Convert to JSON
            if genres_list:
                # Get the name of the first genre
                return genres_list[0]['name']
        except Exception as e:
            return None  # Return None if parsing fails

    data['genre'] = data['genres'].apply(extract_first_genre)

    # Drop the genres column
    data = data.drop(columns=['genres'])

    # Save the processed dataset to a JSON file with indentation
    data.to_json('processed_movies_metadata.json', orient='records', indent=4)

    print(f"Processed dataset saved to 'processed_movies_metadata.json'.")


main()
