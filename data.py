import pickle
import torch
from torchtext.data import get_tokenizer
from datasets import load_dataset

BOS = "<BOS>"
EOS = "<EOS>"
SPECIAL_TOKENS = [BOS, EOS]

def _convert_row_to_tokens(row, tokenizer):
    text = ' '.join([row['context'], row['question'], row['answers']['text'][0]])
    body_tokens = tokenizer(text)
    assert not any([token in SPECIAL_TOKENS for token in body_tokens])
    return body_tokens

def _add_special_tokens(tokens):
    return [BOS] + tokens + [EOS]

def _get_vocabulary(tokens_per_row):
    token_to_id = {}
    token_to_count = {}

    for row in tokens_per_row:
        for token in row:
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)

            if token not in token_to_count:
                token_to_count[token] = 0
            token_to_count[token] += 1
            
    return token_to_id, token_to_count

def _create_infinite_data_iterator(tokens_per_row):
    i = 0
    while True:
        i = i % len(tokens_per_row)
        yield tokens_per_row[i]
        i += 1

def _create_starting_batch(data_iterator, batch_size):
    batch = []
    for _ in range(batch_size):
        batch.append(next(data_iterator))
    return batch

def _pad_batch_with_eos(batch):
    max_length = max([len(row) for row in batch])
    for row in batch:
        row += ["<EOS>"] * (max_length - len(row))
    return batch

def _convert_batch_to_ids(batch, token_to_id):
    return [[token_to_id[token] for token in row] for row in batch]

def _convert_batch_of_ids_to_x_and_y(ids_batch):
    x = []
    y = []
    for row in ids_batch:
        x.append(row[:-1])
        y.append(row[1:])

    x_tens = torch.tensor(x)
    y_tens = torch.tensor(y)  

    return x_tens, y_tens

def get_or_create_dataset(data_cache_path, create, limit=None, char_limit=None):
    if create:
        tokenizer = get_tokenizer("basic_english")
        raw_rows = load_dataset('squad', split='train')
        token_rows = [
            _convert_row_to_tokens(row, tokenizer) for row in raw_rows
        ]
        with open(data_cache_path, 'wb') as f:
            pickle.dump(token_rows, f)
    else:
        with open(data_cache_path, 'rb') as f:
            token_rows = pickle.load(f)

    # For debugging purposes, we can limit the number of characters in each row
    if char_limit:
        token_rows = [
            row[:char_limit] for row in token_rows
        ]

    # Add special tokens
    token_rows = [
        _add_special_tokens(row) for row in token_rows
    ]

    # For debugging purposes, we can limit the number of rows we train on
    if limit:
        token_rows = token_rows[:limit]

    dataset = _create_infinite_data_iterator(token_rows)
    token_to_id, _ = _get_vocabulary(token_rows)
    print(len(token_to_id))
    exit()
    return dataset, token_to_id

def get_dummy_dataset():
    tokenizer = get_tokenizer("basic_english")
    raw_rows = ["hi bye", "dog cat", "man woman", "on off"]
    token_rows = [ 
        ["<BOS>"] + tokenizer(row) + ["<EOS>"]
        for row in raw_rows
    ]
    dataset = _create_infinite_data_iterator(token_rows)
    token_to_id, _ = _get_vocabulary(token_rows)
    return dataset, token_to_id


def get_batch(dataset, token_to_id, batch_size):
    unpadded_batch = _create_starting_batch(dataset, batch_size)
    padded_batch = _pad_batch_with_eos(unpadded_batch)
    padded_batch_of_ids = _convert_batch_to_ids(padded_batch, token_to_id)
    return _convert_batch_of_ids_to_x_and_y(padded_batch_of_ids)