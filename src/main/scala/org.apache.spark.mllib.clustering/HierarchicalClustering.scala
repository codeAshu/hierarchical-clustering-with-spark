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

import breeze.linalg.{DenseVector => BDV, Vector => BV, norm => breezeNorm}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

/**
 * the configuration for a hierarchical clusterint algorithm
 *
 * @param strategy the strategy to select the next splited node
 * @param numClusters the number of clusters you want
 * @param subIterations the number of iterations at digging
 * @param epsilon the threshold to stop the sub-iterations
 * @param randomSeed uses in sampling data for initializing centers in each sub iterations
 */
class HierarchicalClusteringConf(
  private var strategy: ClusterTreeStats,
  private var numClusters: Int,
  private var subIterations: Int,
  private var epsilon: Double,
  private var randomSeed: Int) extends Serializable {

  def this() = this(new ClusterVarianceStats, 100, 20, 10E-6, 1)

  def setStrategy(strategy: ClusterTreeStats): this.type = {
    this.strategy = strategy
    this
  }

  def getStrategy(): ClusterTreeStats = this.strategy

  def setNumClusters(numClusters: Int): this.type = {
    this.numClusters = numClusters
    this
  }

  def getNumClusters(): Int = this.numClusters

  def setSubIterations(iterations: Int): this.type = {
    this.subIterations = iterations
    this
  }

  def getSubIterations(): Int = this.subIterations

  def setEpsilon(epsilon: Double): this.type = {
    this.epsilon = epsilon
    this
  }

  def getEpsilon(): Double = this.epsilon

  def setRandomSeed(seed: Int): this.type = {
    this.randomSeed = seed
    this
  }

  def getRandomSeed(): Int = this.randomSeed
}


/**
 * This is a divisive hierarchical clustering algorithm based on bi-sect k-means algorithm.
 *
 * @param conf the configuration class for the hierarchical clustering
 */
class HierarchicalClustering(val conf: HierarchicalClusteringConf) extends Serializable {

  /**
   * Constracts with the default configuration
   */
  def this() = this(new HierarchicalClusteringConf())

  /**
   * Trains a hierarchical clustering model with the given configuration
   *
   * @param data training points
   * @return a model for hierarchical clustering
   */
  def train(data: RDD[Vector]): HierarchicalClusteringModel = {
    validateData(data)

    val startTime = System.currentTimeMillis() // to measure the execution time
    val clusterTree = ClusterTree.fromRDD(data) // make the root node
    val model = new HierarchicalClusteringModel(clusterTree)

    var node = model.clusterTree
    var subNodes = split(node) // get sub nodes
    updateAllStats(model.clusterTree) // update the statistics to split the clusters

    // If the followed conditions are satisfied, and then stop the training.
    //   1. The number of the splitted clusters is greater than that of given clusters
    //   2. There is no splittable cluster
    while (model.clusterTree.treeSize() < this.conf.getNumClusters && subNodes.size >= 2) {
      // add the sub nodes in to the tree
      node.insert(subNodes.toList)

      node = nextNode(model.clusterTree)
      subNodes = split(node)
      updateAllStats(model.clusterTree)
    }

    model.isTrained = true
    model.trainTime = (System.currentTimeMillis() - startTime).toInt
    model
  }

  /**
   * validate the given data to train
   */
  private[clustering] def validateData(data: RDD[Vector]) {
    conf match {
      case conf if conf.getNumClusters() > data.count() =>
        throw new IllegalArgumentException("# clusters must be less than # input data records")
      case _ =>
    }
  }

  /**
   * Selects the next node to split
   */
  private[clustering] def nextNode(clusterTree: ClusterTree): ClusterTree = {
    // select the max value stats of clusters which are leafs of a tree
    clusterTree.toSeq().filter(_.isSplitable()).maxBy(_.getStats())
  }

  /**
   * Updates the statistics of each cluster
   */
  private[clustering] def updateAllStats(tree: ClusterTree) {
    val strategy = this.conf.getStrategy()
    val trees = tree.toSeq().filter(_.isSplitable()).filter(_.getStats() == None)
    strategy(trees).foreach { case (tree, stats) => tree.setStats(Some(stats))}
  }

  /**
   * Takes the initial centers for bi-sect k-means
   */
  private[clustering] def takeInitCenters(data: RDD[Vector]): Array[Vector] = {
    // NOTE: In order to avoid to select the same point stochastically, select the many points
    var sampledVectors = data.takeSample(false, 1000, conf.getRandomSeed)

    // finds the pair of the most far points
    // If the both of the pair are same, the all may be same
    var maxDistance = Double.MinValue
    var initVectors = sampledVectors.take(2) // because of bi-sect ==> 2
    for (v1 <- sampledVectors; v2 <- sampledVectors if v1 != v2) {
      val tmpDistance = breezeNorm(v1.toBreeze - v2.toBreeze, 2)
      if(tmpDistance > maxDistance) {
        maxDistance = tmpDistance
        initVectors = Array(v1, v2)
      }
    }
    initVectors.sortWith { case (v1, v2) => breezeNorm(v1.toBreeze, 2) < breezeNorm(v2.toBreeze, 2)}
  }

  /**
   * Splits the given cluster (tree) with bi-sect k-means
   */
  private def split(clusterTree: ClusterTree): Array[ClusterTree] = {
    val data = clusterTree.data
    var centers = takeInitCenters(data)
    var finder: ClosestCenterFinder = new EuclideanClosestCenterFinder(centers)

    def relativeError(o: Double, n: Double): Double = Math.abs((o - n) / o)
    def totalStats(array: Array[Vector]): Double = array.map(v => breezeNorm(v.toBreeze, 2.0)).sum

    // If the following conditions are satisfied, the iteration is stopped
    //   1. the relative error is less than that of configuration
    //   2. the number of executed iteration is greater than that of configuration
    var numIter = 0
    var error = Double.MaxValue
    while (error > conf.getEpsilon() && numIter < conf.getSubIterations()) {

      // finds the closest center of each point
      val closest = data.mapPartitions { iter =>
        val map = scala.collection.mutable.Map.empty[Int, (BV[Double], Int)]
        iter.foreach { point =>
          val idx = finder(point)
          val (sumBV, n) = map.get(idx).getOrElse((BV.zeros[Double](point.size), 0))
          map(idx) = (sumBV + point.toBreeze, n + 1)
        }
        map.toIterator
      }
      // calculates the statistics for splitting of each cluster
      val pointStats = scala.collection.mutable.Map.empty[Int, (BV[Double], Int)]
      closest.collect().foreach { case (key, (point, count)) =>
        val (sumBV, n) = pointStats.get(key).getOrElse((BV.zeros[Double](point.size), 0))
        pointStats(key) = (sumBV + point, n + count)
      }
      // creates the new centers
      val newCenters = pointStats.map { case ((idx: Int, (center: BV[Double], counts: Int))) =>
        Vectors.fromBreeze(center :/ counts.toDouble)
      }.toArray

      error = relativeError(totalStats(centers), totalStats(newCenters))
      centers = newCenters
      numIter += 1
      finder = new EuclideanClosestCenterFinder(centers)
    }

    val closest = data.map(point => (finder(point), point))
    val nodes = centers.zipWithIndex.map { case (center, i) =>
      val subData = closest.filter(_._1 == i).map(_._2)
      new ClusterTree(subData, center)
    }
    nodes
  }
}

/**
 * top-level methods for calling the hierarchical clustering algorithm
 */
object HierarchicalClustering {

  /**
   * Trains a hierarichical clustering model with the given data and the number of clusters
   *
   * NOTE: If there is no splittable cluster, however the number of clusters is
   *       less than the given that, the clustering is stoppped
   *
   * @param data trained data
   * @param numClusters the maximum number of clusters you want
   * @return a hierarchical clustering model
   *
   * TODO: The other parameters for the hierarichical clustering will be applied
   */
  def train(data: RDD[Vector], numClusters: Int): HierarchicalClusteringModel = {
    val conf = new HierarchicalClusteringConf()
        .setNumClusters(numClusters)
    val app = new HierarchicalClustering(conf)
    app.train(data)
  }
}

/**
 * the model for Hierarchical clustering
 *
 * @param clusterTree a cluster as a tree node
 * @param trainTime the time for training as milli-seconds
 * @param isTrained if the model has been trained, the flag is true
 */
class HierarchicalClusteringModel private (
  val clusterTree: ClusterTree,
  var trainTime: Int,
  var isTrained: Boolean) extends Serializable {

  def this(clusterTree: ClusterTree) = this(clusterTree, 0, false)
}

/**
 * A cluster as a tree node which can have its sub nodes
 *
 * @param data the data in the cluster
 * @param center the center of the cluster
 * @param stats the statistics for splitting of the cluster
 * @param dataSize the data size of its data
 * @param children the sub node(s) of the cluster
 * @param parent the parent node of the cluster
 */
class ClusterTree(
  val data: RDD[Vector],
  val center: Vector,
  private var stats: Option[Double],
  private var dataSize: Option[Double],
  private var children: List[ClusterTree],
  private var parent: Option[ClusterTree]) extends Serializable {

  def this(data: RDD[Vector], center: Vector) =
    this(data, center, None, None, List.empty[ClusterTree], None)

  /**
   * Converts the tree into Seq class
   * the sub nodes are recursively expanded
   *
   * @return Seq class which the cluster tree is expanded
   */
  def toSeq(): Seq[ClusterTree] = {
    this.children.size match {
      case 0 => Seq(this)
      case _ => Seq(this) ++ this.children.map(child => child.toSeq()).flatten
    }
  }

  /**
   * Gets the depth of the cluster in the tree
   *
   * @return the depth
   */
  def depth(): Int = {
    this.parent match {
      case None => 0
      case _ => 1 + this.parent.get.depth()
    }
  }

  /**
   * Inserts sub nodes as its children
   *
   * @param children inserted sub nodes
   */
  def insert(children: List[ClusterTree]): Unit = {
    this.children = this.children ++ children
    children.foreach(child => child.setParent(Some(this)))
  }

  /**
   * Inserts a sub node as its child
   *
   * @param child inserted sub node
   */
  def insert(child: ClusterTree): Unit = insert(List(child))

  /**
   * Gets the number of the clusters in the tree. The clusters are only leaves
   *
   * @return the number of the clusters in the tree
   */
  def treeSize(): Int = this.toSeq().filter(_.isLeaf()).size

  /**
   * Gets the data size of the cluster
   *
   * @return the data size
   */
  def getDataSize(): Long = {
    // In order to decrease the training time
    if(this.dataSize == None) {
      this.dataSize = Some(this.data.count())
    }
    this.dataSize.get.toLong
  }

  def setParent(parent: Option[ClusterTree]) = this.parent = parent

  def getParent(): Option[ClusterTree] = this.parent

  def getChildren(): List[ClusterTree] = this.children

  def setStats(stats: Option[Double]) = this.stats = stats

  def getStats(): Option[Double] = this.stats

  def isLeaf(): Boolean = (this.children.size == 0)

  /**
   * The flag that the cluster is splitable
   *
   * @return true is splitable
   */
  def isSplitable(): Boolean = (this.isLeaf() && this.getDataSize() >= 2)
}

/**
 * Companion object for ClusterTree class
 */
object ClusterTree {

  /**
   * Converts `RDD[Vector]` into a ClusterTree instance
   *
   * @param data the data in a cluster
   * @return a ClusterTree instance
   */
  def fromRDD(data: RDD[Vector]): ClusterTree = {
    // calculates its center
    val pointStat = data.mapPartitions { iter =>
      val stat = iter.map(v => (v.toBreeze, 1)).reduce((a, b) => (a._1 + b._1, a._2 + b._2))
      Iterator(stat)
    }.collect().reduce((a, b) => (a._1 + b._1, a._2 + b._2))
    val center = Vectors.fromBreeze(pointStat._1.:/(pointStat._2.toDouble))

    new ClusterTree(data, center)
  }
}

/**
 * This trait class is use for the condition to split the cluster in training
 */
trait ClusterTreeStats
    extends Function1[Seq[ClusterTree], Map[ClusterTree, Double]] with Serializable{
  def apply(clusterTrees: Seq[ClusterTree]): Map[ClusterTree, Double]
}

/**
 * Calculates the data size of a cluster
 */
class DataSizeStats extends ClusterTreeStats {

  override def apply(clusterTrees: Seq[ClusterTree]): Map[ClusterTree, Double] = {
    clusterTrees.map(tree => (tree, tree.data.count().toDouble)).toMap
  }
}

/**
 * Calculates the sum of the variances of the cluster
 */
class ClusterVarianceStats private (private var dimension: Option[Int]) extends ClusterTreeStats {

  def this() = this(None)

  override def apply(clusterTrees: Seq[ClusterTree]): Map[ClusterTree, Double] = {
    clusterTrees.map(tree => (tree, calculateVariance(tree.data))).toMap
  }

  /**
   * Calculates the sum of the variances in the cluster
   *
   * @param data the data of the cluster
   * @return the sum of the variances
   */
  def calculateVariance(data: RDD[Vector]): Double = {
    if (this.dimension == None) this.dimension = Some(data.first().size)
    val zeroVector = () => Vectors.zeros(this.dimension.get).toBreeze

    // mapper for each partition
    val eachStats = data.mapPartitions { iter =>
      var n = 0.0
      var sum = zeroVector()
      var sumOfSquares = zeroVector()
      val diff = zeroVector()
      iter.map(point => point.toBreeze).foreach { point =>
        n += 1.0
        sum = sum + point
        sumOfSquares = sumOfSquares + point.map(Math.pow(_, 2.0))
      }
      Iterator((n, sum, sumOfSquares))
    }

    // reducer
    val (n, sum, sumOfSquares) = eachStats.reduce {
      case ((nA, sumA, sumOfSquareA), (nB, sumB, sumOfSquareB)) =>
        val nAB = nA + nB
        val sumAB = sumA + sumB
        val sumbOfSquareAB = sumOfSquareA + sumOfSquareB
        (nAB, sumAB, sumbOfSquareAB)
    }

    // calculate the sum of the variances of each element
    val variance = n match {
      case n if n > 1 => (sumOfSquares.:*(n) - (sum :* sum)) :/ (n * (n - 1.0))
      case _ => zeroVector()
    }

    variance.fold(0.0)(_ + _)
  }
}
