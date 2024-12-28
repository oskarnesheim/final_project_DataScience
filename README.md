# Group 27 - Foundations of Data Science, La Sapienza University of Rome 2024

This is the repository for the final project of the course Foundations of Data Science at La Sapienza University of Rome in 2024.

## Group members
- Oskar Nesheim
- August Nyheim
- Magnus Ouren

## Project description

The project is a about single-label classification of movies into genres based on their plot summaries. 

## Dataset

The dataset for this project was obtained from Kaggle API. The original data contains about 45,000 movies. To create a single-label dataset, we retained only the primary genre for movies with multiple genres. The dataset lists genres by relevance, allowing us to capture the most representative genre for each movie.

Dataset is available at: [https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download)

## Environment

The project is developed in Python. The required packages are listed in the `requirements.txt` file. To install the required packages, run the following command:

### Optional: Create a virtual environment
For creating a virtual environment, run:

```bash
python -m venv venv
```

For activating the virtual environment, run:

```bash
source venv/bin/activate
```

### Install required packages


```bash
pip install -r requirements.txt
```

For installing torh, run:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```