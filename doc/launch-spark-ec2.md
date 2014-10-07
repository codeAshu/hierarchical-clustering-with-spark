# How to launch a Apache Spark cluster on Amazon EC2

## Prerequirement

You should download Apache Spark source code from Github.
https://github.com/apache/spark

## Command

You should change directory into the Spark source code directory.

```
_REGION='ap-northeast-1'
_ZONE='ap-northeast-1b'
_VERSION='1.1.0'
_MASTER_INSTANCE_TYPE='r3.large'
_SLAVE_INSTANCE_TYPE='r3.8xlarge'
_SLAVES=2
_PRICE=0.5
_CLUSTER_NAME="spark-cluster-v${_VERSION}-${_SLAVE_INSTANCE_TYPE}x${_SLAVES}"
./ec2/spark-ec2 -k "yu_ishikawa@recruit" -i ~/.ec2/yu_ishikawarecruit.pem -s $_SLAVES --master-instance-type="$_MASTER_INSTANCE_TYPE" --instance-type="$_SLAVE_INSTANCE_TYPE" --region="$_REGION" --zone="$_ZONE" --spot-price=$_PRICE --spark-version="${_VERSION}" --hadoop-major-version=2  launch "$_CLUSTER_NAME"
```
