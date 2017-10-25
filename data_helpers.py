import numpy as np
from tokenize_text import clean_str

# Data Preparation
# ==================================================
def load_config_dataset(cfg, dataset_name):
    if dataset_name == "mrpolarity":
        datasets = get_datasets_mrpolarity(cfg["datasets"][dataset_name]["positive_data_file"]["path"],
                                           cfg["datasets"][dataset_name]["negative_data_file"]["path"])
    elif dataset_name == "20newsgroup":
        datasets = get_datasets_20newsgroup(subset="train",
                                            categories=cfg["datasets"][dataset_name]["categories"],
                                            shuffle=cfg["datasets"][dataset_name]["shuffle"],
                                            random_state=cfg["datasets"][dataset_name]["random_state"])
    elif dataset_name == "localdata":
        datasets = get_datasets_localdata(container_path=cfg["datasets"][dataset_name]["container_path"],
                                          categories=cfg["datasets"][dataset_name]["categories"],
                                          shuffle=cfg["datasets"][dataset_name]["shuffle"],
                                          random_state=cfg["datasets"][dataset_name]["random_state"])
    else:
        raise Exception('Unknown dataset {}'.format(dataset_name))
    return load_data_labels(datasets)


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [clean_str(s.strip()) for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [clean_str(s.strip()) for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def load_from_tsv(tsv_file):
    """
    Loads samples from tsv file where first column is the sentence and second column is the integer label
    """
    # Load data from files
    all_examples = list(open(tsv_file, "r", encoding='utf-8').readlines())
    split_lines = [l.split('\t') for l in all_examples]
    x_text = [clean_str(s[0].strip()) for s in split_lines]
    label_integers = [int(s[1].strip()) for s in split_lines]
    label_values = list(set(label_integers))
    if len(label_values) > 2 or min(label_values) != 0 or max(label_values) != 1:
        raise Exception('Labels are not in correct format {0} {1}'.format(label_values[0], label_values[1]))
    y = np.array([[0, 1] if l == 1 else [1, 0] for l in label_integers])
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def get_datasets_20newsgroup(subset='train', categories=None, shuffle=True, random_state=42):
    """
    Retrieve data from 20 newsgroups
    :param subset: train, test or all
    :param categories: List of newsgroup name
    :param shuffle: shuffle the list or not
    :param random_state: seed integer to shuffle the dataset
    :return: data and labels of the newsgroup
    """
    from sklearn.datasets import fetch_20newsgroups
    datasets = fetch_20newsgroups(subset=subset, categories=categories, shuffle=shuffle, random_state=random_state)
    return datasets


def get_datasets_mrpolarity(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]

    datasets = dict()
    datasets['data'] = positive_examples + negative_examples
    target = [0 for x in positive_examples] + [1 for x in negative_examples]
    datasets['target'] = target
    datasets['target_names'] = ['positive_examples', 'negative_examples']
    return datasets


def get_datasets_localdata(container_path=None, categories=None, load_content=True,
                           encoding='utf-8', shuffle=True, random_state=42):
    """
    Load text files with categories as subfolder names.
    Individual samples are assumed to be files stored a two levels folder structure.
    :param container_path: The path of the container
    :param categories: List of classes to choose, all classes are chosen by default (if empty or omitted)
    :param shuffle: shuffle the list or not
    :param random_state: seed integer to shuffle the dataset
    :return: data and labels of the dataset
    """
    from sklearn.datasets import load_files
    datasets = load_files(container_path=container_path, categories=categories,
                          load_content=load_content, shuffle=shuffle, encoding=encoding,
                          random_state=random_state)
    return datasets


def load_data_labels(datasets):
    """
    Load data and labels
    :param datasets:
    :return:
    """
    # Split by words
    x_text = datasets['data']
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    labels = []
    for i in range(len(x_text)):
        label = [0 for j in datasets['target_names']]
        label[datasets['target'][i]] = 1
        labels.append(label)
    y = np.array(labels)
    return [x_text, y]


def load_embedding_vectors_word2vec(vocabulary, filename, binary):
    # load embedding_vectors from the word2vec
    encoding = 'utf-8'
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        # initial matrix with random uniform
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        if binary:
            binary_len = np.dtype('float32').itemsize * vector_size
            for line_no in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':
                        word.append(ch)
                word = str(b''.join(word), encoding=encoding, errors='strict')
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.seek(binary_len, 1)
        else:
            for line_no in range(vocab_size):
                line = f.readline()
                if line == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                parts = str(line.rstrip(), encoding=encoding, errors='strict').split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                word, vector = parts[0], list(map('float32', parts[1:]))
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = vector
        f.close()
        return embedding_vectors


def load_embedding_vectors_glove(vocabulary, filename, vector_size):
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filename, encoding='utf-8')
    for line in f:
        values = line.rstrip().split(' ')
        if len(values) <= 2:
            continue  # header
        word = values[0]
        if word == '':
            vector_start = 2
        else:
            vector_start = 1
        vector = np.asarray([float(i) for i in values[vector_start:]])
        if vector.shape[0] != vector_size:
            raise Exception('Embedding size mismatch. Pre trained {0}, input option {1}'
                            .format(vector.shape[0], vector_size))
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors
