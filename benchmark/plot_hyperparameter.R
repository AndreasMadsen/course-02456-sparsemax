#!/usr/bin/env Rscript
library(ggplot2);

args = commandArgs(TRUE);
dataframe = read.csv(file('stdin'))

p = ggplot(aes(x=regualizer, y=mean, colour=regressors, ymin=lower, ymax=upper), data=dataframe) +
  geom_line() +
  geom_errorbar(width=0.4) +
  scale_x_log10() +
  facet_grid(datasets ~ .)

ggsave(args[1], p)
print(args[1])
