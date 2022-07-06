library(psych)


common <- read.csv("common_set.csv", stringsAsFactors = FALSE, encoding = "UTF-8")
df <- common[,c(40,42,41,43,44)]
names(df) <- c("Count", "Ten words", "Separate words", "Norm mean", "Tf-idf mean")

pairs.panels(df, smooth = TRUE, scale = FALSE, density = TRUE, ellipses = TRUE, 
             method = "spearman", pch = 4,  lm = FALSE, cor = TRUE, jiggle = FALSE, hist.col = 4,
             stars = TRUE, ci = FALSE)
