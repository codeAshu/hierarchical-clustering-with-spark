#!/bin/env R

library(knitr)
opts_knit$set(upload.fun = image_uri)

knit2html("benchmark.Rmd", options=c("toc", "mathjax"))

