# data.py

import random
import gzip
import json
import requests
import torch

genre_url_dict = {
    'poetry': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz',
    'children': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz',
    'comics_graphic': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz',
    'fantasy_paranormal': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz',
    'history_biography': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz',
    'mystery_thriller_crime': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz',
    'romance': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz',
    'young_adult': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz'
}


def load_reviews(url, head=10000, sample_size=2000):
    reviews = []
    response = requests.get(url, stream=True)

    with gzip.open(response.raw, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            reviews.append(d['review_text'])
            if head and i >= head:
                break

    return random.sample(reviews, min(sample_size, len(reviews)))


def get_data():

    genre_reviews = {}

    for genre, url in genre_url_dict.items():
        print(f"Loading reviews for {genre}")
        genre_reviews[genre] = load_reviews(url)

    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for genre, reviews in genre_reviews.items():

        reviews = random.sample(reviews, 1000)

        train_texts += reviews[:800]
        train_labels += [genre] * 800

        test_texts += reviews[800:]
        test_labels += [genre] * 200

    return train_texts, train_labels, test_texts, test_labels


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)