#!/usr/bin/env Rscript
library(ggplot2);

args = commandArgs(TRUE);
dataframe = read.csv(file('stdin'))

p = ggplot(aes(x=regualizer, y=mean, colour=regressors, ymin=lower, ymax=upper), data=dataframe) +
  geom_line() +
  geom_errorbar(width=0.4) +
  scale_x_log10() +
  facet_grid(datasets ~ .) +
  ylab("JS divergence") +
  theme(
    legend.title=element_blank(),
    text=element_text(size=10),
    legend.position="bottom"
  )

# width should be
#   \usepackage{layouts}
#   \printinunitsof{cm}\prntlen{\columnwidth}
ggsave(args[1], p, width=8.59877, height=13, units="cm")
