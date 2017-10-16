import numpy as np


def generate_parameter():
    # alphabetical
    beta1 = 1 - 10 ** (-1 - np.random.rand())  # 0.9 to 0.99
    beta2 = 1 - 10 ** (-3 - 2 * np.random.rand())  # 0.999 to 0.99999
    embedding_dimensions = 2 ** np.random.randint(low=5, high=8)
    valid_filter_sizes = ['1,3', '2,3', '3,4', '2,4', '1,2,3',
                          '2,3,4', '2,3,4', '2,3,4',
                          '3,4,5', '1,2,3,4', '2,3,4,5']
    filter_sizes = np.random.choice(valid_filter_sizes)
    min_learning_rate = 10 ** (-4 * np.random.rand() - 2)
    num_filters = 2 ** np.random.randint(low=5, high=8)
    output_string = '\t'.join(
        [str(i) for i in [beta1, beta2, embedding_dimensions, filter_sizes, min_learning_rate, num_filters]])
    print(output_string)
    return output_string
