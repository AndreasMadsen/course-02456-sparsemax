
import itertools

import numpy as np
import scipy.stats


class Table:
    def __init__(self, content, col_names, row_names):
        self.content = content
        self.col_names = col_names
        self.row_names = row_names

    def __str__(self):
        output = ''

        # print header
        format_header = 'c' * len(self.col_names)
        output += '\\begin{tabular}{r|%s}\n' % format_header
        output += '& %s \\\\\n' % (' & '.join(self.col_names))
        output += '\\hline\n'

        # print body
        for row_name, row_content in zip(self.row_names, self.content):
            output += "%s & %s \\\\\n" % (row_name, ' & '.join(row_content))

        # print footer
        output += '\\end{tabular}\n'
        return output

    def save(self, filename):
        with open(filename, "w") as fd:
            fd.write(str(self))


class SummaryTable(Table):
    def __init__(self, content, col_names, row_names,
                 format="$%.2f \\pm %.2f$"):
        formatted = self.content(content, format=format)
        super().__init__(formatted, col_names, row_names)

    @staticmethod
    def content(content, format="$%.2f \\pm %.2f$"):
        # calculate statistics
        mean = np.mean(content, axis=2)
        sem = scipy.stats.sem(content, axis=2)
        ci = scipy.stats.t.interval(
            0.95, content.shape[2] - 1, scale=sem
        )[1]

        return [
            [
                format % (mean_val, ci_val)
                for mean_val, ci_val in zip(mean_row, ci_row)
            ] for mean_row, ci_row in zip(mean, ci)
        ]


class PairTable(Table):
    def __init__(self, content_left, content_right,
                 col_names, pair_names, row_names):
        self.content_left = content_left
        self.content_right = content_right
        self.col_names = col_names
        self.pair_names = pair_names
        self.row_names = row_names

    def __str__(self):
        output = ''

        # print latex header
        format_header = '|cc' * len(self.col_names)
        output += '\\begin{tabular}{r%s}\n' % format_header

        # print table header
        col_header = ' & '.join(map(
            lambda s: '\multicolumn{2}{c|}{%s}' % s, self.col_names[:-1]
        )) + (' & \multicolumn{2}{c}{%s}' % self.col_names[-1])
        pair_header = ' & '.join(self.pair_names * len(self.col_names))
        output += '& %s \\\\\n' % col_header
        output += '& %s \\\\\n' % pair_header
        output += '\\hline\n'

        # print body
        for row_name, row_content_left, row_content_right in \
                zip(self.row_names, self.content_left, self.content_right):
            row_content = itertools.chain.from_iterable(
                zip(row_content_left, row_content_right)
            )
            output += "%s & %s \\\\\n" % (row_name, ' & '.join(row_content))

        # print footer
        output += '\\end{tabular}\n'
        return output
