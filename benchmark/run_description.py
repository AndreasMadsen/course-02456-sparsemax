import _benchmark

import os.path as path

import datasets
from table import Table

thisdir = path.dirname(path.realpath(__file__))
tabledir = path.join(thisdir, '..', 'latex', 'report', 'tables')


def descriptions(datasets):
    col_names = ['\\#Features', '\\#Labels', 'Train size', 'Test size']
    row_names = []
    content = []

    for DatasetInitializer in datasets:
        dataset = DatasetInitializer()
        row_names.append(dataset.name)

        content.append([
            str(dataset.input_size),
            str(dataset.output_size),
            str(dataset.train.inputs.shape[0]),
            str(dataset.test.inputs.shape[0])
        ])

    return (content, col_names, row_names)


def main():
    content, col_names, row_names = descriptions(datasets.all_datasets)

    table = Table(content, col_names, row_names)
    print(str(table))
    table.save(path.join(tabledir, 'descriptions.tex'))

if __name__ == "__main__":
    main()
