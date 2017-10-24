import os


def process_large(input_directory, output_folder, function):
    file_names = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
    batch_size = 10 ** 6  # Batching to prevent trashing of HD
    # https://stackoverflow.com/questions/16669428/process-very-large-20gb-text-file-line-by-line
    batch = []
    for file_name in file_names:
        output_tsv_filename = os.path.join(output_folder, file_name)
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        with open(os.path.join(input_directory, file_name), encoding='utf-8') as fi, \
                open(output_tsv_filename, 'w', encoding='utf-8') as fo:
            for line in fi:
                processed_line = function(line)
                batch.append(processed_line)
                if len(batch) == batch_size:
                    fo.writelines(batch)
                    batch = []
            fo.writelines(batch)
