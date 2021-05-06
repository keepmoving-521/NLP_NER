import pandas as pd
import numpy as np

CORPUS_PATH = './data/train.txt'

KEYS_MODEL_SAVE_PATH = './data/bi_lstm_ner.h5'
WORD_DICTIONARY_PATH = './data/word_dictionary.pk'
INVERSE_WORD_DICTIONARY_PATH = './data/inverse_word_dictionary.pk'
LABEL_DICTIONARY_PATH = './data/label_dictionary.pk'
OUTPUT_DICTIONARY_PATH = './data/output_dictionary.pk'

CONSTANTS = [
    KEYS_MODEL_SAVE_PATH,
    WORD_DICTIONARY_PATH,
    INVERSE_WORD_DICTIONARY_PATH,
    LABEL_DICTIONARY_PATH,
    OUTPUT_DICTIONARY_PATH
]


# load data from corpus to from pandas DataFrame
def load_data():
    with open(CORPUS_PATH, 'r') as f:
        text_data = [text.strip() for text in f.readlines()]
    text_data = [text_data[k].split('\t') for k in range(0, len(text_data))]
    index = range(0, len(text_data), 3)

    # Transforming data to matrix format for neural network
    input_data = list()
    for i in range(1, len(index) - 1):
        rows = text_data[index[i - 1]: index[i]]
        sentence_no = np.array([i] * len(rows[0]), dtype=str)
        rows.append(sentence_no)
        rows = np.array(rows).T
        input_data.append(rows)

    input_data = pd.DataFrame(np.concatenate([item for item in input_data]), columns=['word', 'pos', 'tag', 'sent_no'])

    return input_data


if __name__ == '__main__':
    data = load_data()
    print(data)
