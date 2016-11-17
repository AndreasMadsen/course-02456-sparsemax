
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
        format_header = 'c' * (len(self.col_names) + 1)
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
    def __init__(self, content, col_names, row_names):
        # calculate statistics
        mean = np.mean(content, axis=2)
        sem = scipy.stats.sem(content, axis=2)
        ci = scipy.stats.t.interval(
            0.95, content.shape[2] - 1, scale=sem
        )[1]

        formatted = [
            [
                "$%.2f \\pm %.2f$" % (mean_val, ci_val)
                for mean_val, ci_val in zip(mean_row, ci_row)
            ] for mean_row, ci_row in zip(mean, ci)
        ]

        super().__init__(formatted, col_names, row_names)
