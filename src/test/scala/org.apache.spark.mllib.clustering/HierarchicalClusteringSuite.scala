/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.clustering

import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.random.UniformGenerator
import org.apache.spark.rdd.RDD
import org.apache.spark.util.LocalSparkContext
import org.scalatest.{BeforeAndAfterEach, FunSuite}

class HierarichicalClusteringConfSuite extends FunSuite {

  test("constract a new instance without parameters") {
    val conf = new HierarchicalClusteringConf()
    assert(conf.getNumClusters === 100)
    assert(conf.getSubIterations === 20)
    assert(conf.getEpsilon === 10E-6)
  }

  test("can replace numClusters") {
    val conf = new HierarchicalClusteringConf()
    assert(conf.getNumClusters === 100)
    conf.setNumClusters(50)
    assert(conf.getNumClusters === 50)
  }

  test("can replace subIterations") {
    val conf = new HierarchicalClusteringConf()
    assert(conf.getSubIterations === 20)
    conf.setSubIterations(50)
    assert(conf.getSubIterations === 50)
  }
}

class HierarchicalClusteringModelSuite
    extends FunSuite with LocalSparkContext with BeforeAndAfterEach {

  var data: RDD[Vector] = _
  var app: HierarchicalClustering = _
  var model: HierarchicalClusteringModel = _

  override def beforeEach() {
    val seed = (0 to 99).map { i =>
      val label = Math.floor(i / 10)
      val vector = Vectors.dense(label, label, label)
      (label, vector)
    }
    data = sc.parallelize(seed.map(_._2))

    val conf = new HierarchicalClusteringConf().setNumClusters(10).setRandomSeed(1)
    app = new HierarchicalClustering(conf)
    model = app.train(data)
  }

  test("should get the array of ClusterTree") {
    val clusters = model.getClusters()
    assert(clusters.isInstanceOf[Array[ClusterTree]])
    assert(clusters.size === 10)
  }

  test("the number of predicted clusters should be same") {
    val predictedData = model.predict(data)
    // the number of contained vectors in each cluster is 10
    predictedData.map { case (i, vector) => (i, 1)}.reduceByKey(_ + _)
        .collect().foreach { case (idx, n) => assert(n === 10) }
  }
}

class HierarchicalClusteringSuite extends FunSuite with BeforeAndAfterEach with LocalSparkContext {

  var vectors: Seq[Vector] = _
  var data: RDD[Vector] = _

  override def beforeEach() {
    vectors = Seq(
      Vectors.dense(0.0, 0.0, 0.0),
      Vectors.dense(1.0, 2.0, 3.0),
      Vectors.dense(4.0, 5.0, 6.0),
      Vectors.dense(7.0, 8.0, 9.0),
      Vectors.dense(10.0, 11.0, 12.0),
      Vectors.dense(13.0, 14.0, 15.0),
      Vectors.dense(16.0, 17.0, 18.0)
    )
    data = sc.parallelize(vectors, 2)
  }

  test("train method called by the companion object") {
    val numClusters = 2
    val model = HierarchicalClustering.train(data, numClusters)
    assert(model.getClass.getSimpleName.toString === "HierarchicalClusteringModel")
  }

  test("train") {
    val conf = new HierarchicalClusteringConf().setNumClusters(3)
    val app = new HierarchicalClustering(conf)
    val model = app.train(data)
    assert(model.clusterTree.toSeq().filter(_.isLeaf()).size === 3)
    model.clusterTree.toSeq().foreach { tree: ClusterTree => assert(tree.getVariance() != None)}
  }

  test("train with a random dataset") {
    val data = sc.parallelize((1 to 99)
        .map(i => Vectors.dense(i + Math.random, i + Math.random)), 2)
    val conf = new HierarchicalClusteringConf().setNumClusters(10)
    val app = new HierarchicalClustering(conf)
    val model = app.train(data)
    assert(model.clusterTree.treeSize() === 10)
    model.clusterTree.toSeq().foreach { tree: ClusterTree => assert(tree.getVariance() != None)}
  }

  test("should stop if the variance inclreases fot the clustering") {
    // generate data which has more than 5 clusters
    val data = sc.parallelize((1 to 99)
        .map(i => Vectors.dense((i % 5).toDouble, (i % 5).toDouble)), 2)
    // but the number of given clusters is 10
    val conf = new HierarchicalClusteringConf().setNumClusters(10)
    val app = new HierarchicalClustering(conf)
    val model = app.train(data)
    assert(model.clusterTree.treeSize() === 5)
    model.clusterTree.toSeq().foreach { tree: ClusterTree => assert(tree.getVariance() != None)}
  }

  test("should stop if there is no splittable cluster") {
    val data = sc.parallelize((1 to 100).map(i => Vectors.dense(0.0, 0.0)), 1)
    val conf = new HierarchicalClusteringConf().setNumClusters(5)
    val app = new HierarchicalClustering(conf)
    val model = app.train(data)
    assert(model.clusterTree.treeSize() === 1)
  }

  test("parameter validation") {
    val data = sc.parallelize((1 to 100).map(i => Vectors.dense(0.0, 0.0)), 1)
    intercept[IllegalArgumentException] {
      val conf = new HierarchicalClusteringConf().setNumClusters(101)
      val app = new HierarchicalClustering(conf)
      app.train(data)
    }
  }

  test("takeInitCenters") {
    val data = Array(
      Vectors.dense(1.0, 4.0, 7.0),
      Vectors.dense(2.0, 5.0, 8.0), // this is the mean vector in the data
      Vectors.dense(3.0, 6.0, 9.0)
    )
    val tree = ClusterTree.fromRDD(sc.parallelize(data, 2))
    val conf = new HierarchicalClusteringConf().setRandomRange(0.1)
    val initVectors = new HierarchicalClustering(conf).takeInitCenters(tree.center)
    val relativeError1 = (data(1).toBreeze - initVectors(0)).:/(data(1).toBreeze)
    val relativeError2 = (initVectors(1) - data(1).toBreeze).:/(data(1).toBreeze)
    assert(initVectors.size === 2)
    assert(relativeError1.forall(_ <= 0.1))
    assert(relativeError2.forall(_ <= 0.1))
  }
}


class ClusterTreeSuite extends FunSuite with LocalSparkContext with SampleData {

  override var data: RDD[Vector] = _
  override var subData1: RDD[Vector] = _
  override var subData21: RDD[Vector] = _
  override var subData22: RDD[Vector] = _
  override var subData2: RDD[Vector] = _

  override def beforeAll() {
    super.beforeAll()
    data = sc.parallelize(vectors, 3)
    subData1 = sc.parallelize(subVectors1)
    subData2 = sc.parallelize(subVectors2)
    subData21 = sc.parallelize(subVectors21)
    subData22 = sc.parallelize(subVectors22)
  }

  test("insert a new Cluster with a Cluster") {
    val root = ClusterTree.fromRDD(data)
    val child = ClusterTree.fromRDD(subData1)
    assert(root.getChildren().size === 0)
    root.insert(child)
    assert(root.getChildren().size === 1)
    assert(root.getChildren.apply(0) === child)
  }

  test("insert Cluster List") {
    val root = ClusterTree.fromRDD(data)
    val children = List(
      ClusterTree.fromRDD(subData1),
      ClusterTree.fromRDD(subData2)
    )
    assert(root.getChildren().size === 0)
    root.insert(children)
    assert(root.getChildren().size === 2)
    assert(root.getChildren() === children)
  }

  test("treeSize and depth") {
    val root = ClusterTree.fromRDD(data)
    val child1 = ClusterTree.fromRDD(subData1)
    val child2 = ClusterTree.fromRDD(subData2)
    val child21 = ClusterTree.fromRDD(subData21)
    val child22 = ClusterTree.fromRDD(subData22)

    root.insert(child1)
    root.insert(child2)
    child2.insert(child21)
    child2.insert(child22)
    assert(root.treeSize() === 3)
    assert(child1.treeSize() === 1)
    assert(child2.treeSize() === 2)
    assert(child21.treeSize() === 1)
    assert(child22.treeSize() === 1)

    assert(root.depth() === 0)
    assert(child1.depth() === 1)
    assert(child2.depth() === 1)
    assert(child21.depth() === 2)
    assert(child22.depth() === 2)
  }
}

class ClusteringStatsUpdaterSuite extends FunSuite with LocalSparkContext {

  test("the variance of a data should be greater than that of another one") {
    // the variance of subData2 is greater than that of subData1
    def rand(): Double = new UniformGenerator().nextValue()
    def rand2(): Double = 10 * new UniformGenerator().nextValue()
    val subData1 = (1 to 99).map(i => Vectors.dense(rand, rand))
    val subData2 = (1 to 99).map(i => Vectors.dense(rand2, rand2))
    val data = subData1 ++ subData2

    val root = ClusterTree.fromRDD(sc.parallelize(data, 2))
    val child1 = ClusterTree.fromRDD(sc.parallelize(subData1, 2))
    val child2 = ClusterTree.fromRDD(sc.parallelize(subData2, 2))
    root.insert(child1)
    root.insert(child2)

    val updater = new ClusterTreeStatsUpdater
    root.toSeq().foreach(tree => updater(tree))
    assert(child1.getVariance().get < child2.getVariance().get)
  }

  test("the sum of variance should be 0.0 with all same records") {
    val data = sc.parallelize((1 to 9).map(i => Vectors.dense(1.0, 2.0)), 4)
    val clusterTree = ClusterTree.fromRDD(data)
    val updater = new ClusterTreeStatsUpdater()
    updater(clusterTree)
    assert(clusterTree.getDataSize().get === 9)
    assert(clusterTree.getVariance().get === 0.0)
  }

  test("the sum of variance should be 0.0 with one record") {
    val data = sc.parallelize(Seq(Vectors.dense(1.0, 2.0, 3.0)), 1)
    val clusterTree = ClusterTree.fromRDD(data)
    val updater = new ClusterTreeStatsUpdater()
    updater(clusterTree)
    assert(clusterTree.getVariance().get === 0.0)
  }

  test("the variance should be 7.5") {
    val seedData = Seq(
      Vectors.dense(1.0), Vectors.dense(2.0), Vectors.dense(3.0),
      Vectors.dense(4.0), Vectors.dense(5.0), Vectors.dense(6.0),
      Vectors.dense(7.0), Vectors.dense(8.0), Vectors.dense(9.0)
    )
    val clusterTree = ClusterTree.fromRDD(sc.parallelize(seedData, 3))
    val updater = new ClusterTreeStatsUpdater()
    updater(clusterTree)
    assert(clusterTree.getDataSize().get === 9)
    assert(clusterTree.getVariance().get === 7.5)
  }

  test("the variance should be 8332500") {
    val data = sc.parallelize((1 to 9999).map(Vectors.dense(_)), 4)
    val clusterTree = ClusterTree.fromRDD(data)
    val updater = new ClusterTreeStatsUpdater()
    updater(clusterTree)
    assert(clusterTree.getDataSize().get === 9999)
    assert(clusterTree.getVariance().get === 8332500)
  }
}

sealed trait SampleData {
  val vectors = (0 to 99).map(i => Vectors.dense(Math.random(), Math.random(), Math.random())).toSeq
  val subVectors1 = (0 to 69).map(vectors(_))
  val subVectors2 = (70 to 99).map(vectors(_))
  val subVectors21 = (70 to 89).map(vectors(_))
  val subVectors22 = (90 to 99).map(vectors(_))

  var data: RDD[Vector]
  var subData1: RDD[Vector]
  var subData2: RDD[Vector]
  var subData21: RDD[Vector]
  var subData22: RDD[Vector]
}
