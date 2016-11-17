
import numpy as np
import scipy.stats


class Table:
    def __init__(self, results, col_names, row_names):
        # calculate statistics
        self.mean = np.mean(results, axis=2)
        sem = scipy.stats.sem(results, axis=2)
        self.ci = scipy.stats.t.interval(
            0.95, results.shape[2] - 1, scale=sem
        )[1]

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
        for row_name, row_mean, row_ci in \
                zip(self.row_names, self.mean, self.ci):
            row_format = ' & '.join([
                "$%.2f \\pm %.2f$" % stats for stats in zip(row_mean, row_ci)
            ])
            output += "%s & %s \\\\\n" % (row_name, row_format)

        # print footer
        output += '\\end{tabular}\n'
        return output

    def save(self, filename):
        with open(filename, "w") as fd:
            fd.write(str(self))
