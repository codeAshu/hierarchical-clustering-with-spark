#!/bin/env R

library(knitr)
opts_knit$set(upload.fun = image_uri)

knit2html("benchmark2.Rmd", options=c("toc", "mathjax"))

