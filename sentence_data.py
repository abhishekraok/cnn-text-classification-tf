import numpy as np
import sys


class SentenceData:
    """
    Standard sentence classification data format. TSV with first column as text and second column as label (0,1)
    """

    def __init__(self, sentences, labels):
        self.sentences = sentences  # size [samples]
        self.labels = labels  # size [samples, num classes]

    @staticmethod
    def load_from_tsv(tsv_file):
        """
        Loads samples from tsv file where first column is the sentence and second column is the integer label
        """
        # Load data from files
        all_examples = list(open(tsv_file, "r", encoding='utf-8').readlines())
        split_lines = [l.split('\t') for l in all_examples]
        sentences = [s[0].strip() for s in split_lines]
        label_integers = [int(s[1].strip()) for s in split_lines]
        label_values = list(set(label_integers))
        if len(label_values) > 2 or min(label_values) != 0 or max(label_values) != 1:
            raise Exception('Labels are not in correct format {0} {1}'.format(label_values[0], label_values[1]))
        labels = np.array([[0, 1] if l == 1 else [1, 0] for l in label_integers])
        return SentenceData(sentences, labels)

    def summary(self):
        sample_size = len(self.sentences)
        positive_count = sum(self.labels[:, 1] == 1)
        positive_percentage = (100 * positive_count) / sample_size
        return sample_size, positive_count, positive_percentage

    @staticmethod
    def format_summary(summary):
        header = 'sample size\t positive count\t positive percentage\n'
        ss, pc, pp = summary
        values = '{0}\t{1}\t{2}\n'.format(ss, pc, pp)
        return header + values


if __name__ == '__main__':
    sys_input_filename = sys.argv[1]
    sys_output_filename = sys.argv[2]
    sentence_data = SentenceData.load_from_tsv(sys_input_filename)
    sd_summary = sentence_data.summary()
    print('SystemLog: {0}')
    with open(sys_output_filename, 'w') as f:
        f.write(SentenceData.format_summary(sd_summary))
