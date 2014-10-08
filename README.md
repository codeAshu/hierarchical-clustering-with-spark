# README

## What's this?

This is a hierarchical clustering algorithm implementation on Apache Spark.

## Prerequirement

You have to build your Apache Spark cluster on EC2 with `spark-ec2` which is in spark project.

## Execute the Benchmark Script

`bin/experiment.sh` is the benchmark script.

```
## how to build
cd ~
git clone {this_repository}
cd ${this_project}
./sbt/sbt clean package

## execute the benchmark script
bash ./bin/experiment.sh
```
