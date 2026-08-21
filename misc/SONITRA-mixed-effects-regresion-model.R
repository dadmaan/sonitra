benchmark <- read.csv("regression_table_with_metadata.csv")
benchmark$song <- as.factor(benchmark$song)
benchmark$condition <- as.factor(benchmark$condition)
benchmark$condition <- relevel(benchmark$condition, ref = "baseline")
benchmark$composer <- as.factor(benchmark$meta.canonical_composer)
benchmark$duration <- as.numeric(benchmark$meta.duration)
benchmark$year <- as.numeric(benchmark$meta.year)

library(rlang)
library(glmmTMB)

model <- glmmTMB(
  note.onset_f1 ~ condition + duration + year + (1 | song)+ (1 | composer), 
  data = benchmark,
  family = beta_family(link = "logit"))

print(summary(model))  
x <- ranef(model)$cond$composer
print(x[order(x[,1]), , drop = FALSE])

