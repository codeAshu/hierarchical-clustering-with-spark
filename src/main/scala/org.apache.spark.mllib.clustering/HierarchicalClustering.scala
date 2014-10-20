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
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.util.random.XORShiftRandom

/**
 * the configuration for a hierarchical clustering algorithm
 *
 * @param numClusters the number of clusters you want
 * @param subIterations the number of iterations at digging
 * @param epsilon the threshold to stop the sub-iterations
 * @param randomSeed uses in sampling data for initializing centers in each sub iterations
 * @param randomRange the range coefficient to generate random points in each clustering step
 */
class HierarchicalClusteringConf(
  private var numClusters: Int,
  private var subIterations: Int,
  private var numRetries: Int,
  private var epsilon: Double,
  private var randomSeed: Int,
  private[mllib] var randomRange: Double) extends Serializable {

  def this() = this(100, 20, 20, 10E-6, 1, 0.1)

  def setNumClusters(numClusters: Int): this.type = {
    this.numClusters = numClusters
    this
  }

  def getNumClusters(): Int = this.numClusters

  def setSubIterations(iterations: Int): this.type = {
    this.subIterations = iterations
    this
  }

  def setNumRetries(numRetries: Int): this.type = {
    this.numRetries = numRetries
    this
  }

  def getNumRetries(): Int = this.numRetries

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

  def setRandomRange(range: Double): this.type = {
    this.randomRange = range
    this
  }
}


/**
 * This is a divisive hierarchical clustering algorithm based on bi-sect k-means algorithm.
 *
 * @param conf the configuration class for the hierarchical clustering
 */
class HierarchicalClustering(val conf: HierarchicalClusteringConf) extends Serializable {

  /**
   * Constructs with the default configuration
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
    val statsUpdater = new ClusterTreeStatsUpdater()

    var node: Option[ClusterTree] = Some(model.clusterTree)
    statsUpdater(node.get)

    // If the followed conditions are satisfied, and then stop the training.
    //   1. There is no splittable cluster
    //   2. The number of the splitted clusters is greater than that of given clusters
    //   3. The total variance of all clusters increases, when a cluster is splitted
    var totalVariance = Double.MaxValue
    var newTotalVariance = model.clusterTree.getVariance().get
    while (node != None
        && model.clusterTree.treeSize() < this.conf.getNumClusters
        && totalVariance >= newTotalVariance) {

      // split some times in order not to be wrong clustering result
      var isMerged = false
      var isSingleCluster = false
      for (retry <- 1 to this.conf.getNumRetries()) {
        if (isMerged == false && isSingleCluster == false) {
          var subNodes = split(node.get).map(subNode => statsUpdater(subNode))
          // it seems that there is no splittable node
          if (subNodes.size == 1) isSingleCluster = false
          // add the sub nodes in to the tree
          // if the sum of variance of sub nodes is greater than that of pre-splitted node
          if (node.get.getVariance().get > subNodes.map(_.getVariance().get).sum) {
            node.get.insert(subNodes.toList)
            isMerged = true
          }
        }
      }
      node.get.isVisited = true

      // update the total variance and select the next splittable node
      totalVariance = newTotalVariance
      newTotalVariance = model.clusterTree.toSeq().filter(_.isLeaf()).map(_.getVariance().get).sum
      node = nextNode(model.clusterTree)
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
  private[clustering] def nextNode(clusterTree: ClusterTree): Option[ClusterTree] = {
    // select the max variance of clusters which are leafs of a tree
    clusterTree.toSeq().filter(tree => tree.isSplittable() && !tree.isVisited) match {
      case list if list.isEmpty => None
      case list => Some(list.maxBy(_.getVariance()))
    }
  }

  /**
   * Takes the initial centers for bi-sect k-means
   */
  private[clustering] def takeInitCenters(centers: Vector): Array[BV[Double]] = {
    val random = new XORShiftRandom()
    Array(
      centers.toBreeze.map(elm => elm - random.nextDouble() * elm * this.conf.randomRange),
      centers.toBreeze.map(elm => elm + random.nextDouble() * elm * this.conf.randomRange)
    )
  }

  /**
   * Splits the given cluster (tree) with bi-sect k-means
   *
   * @param clusterTree the splitted cluster
   * @return an array of ClusterTree. its size is generally 2, but its size can be 1
   */
  private def split(clusterTree: ClusterTree): Array[ClusterTree] = {
    val data = clusterTree.data
    var centers = takeInitCenters(clusterTree.center)
    var finder: ClosestCenterFinder = new EuclideanClosestCenterFinder(centers)

    // If the following conditions are satisfied, the iteration is stopped
    //   1. the relative error is less than that of configuration
    //   2. the number of executed iteration is greater than that of configuration
    //   3. the number of centers is greater then 1. if 1 means that the cluster is not splittable
    var numIter = 0
    var error = Double.MaxValue
    while (error > conf.getEpsilon()
        && numIter < conf.getSubIterations()
        && centers.size > 1) {
      // finds the closest center of each point
      data.sparkContext.broadcast(finder)
      val newCenters = data.mapPartitions { iter =>
        // calculate the accumulation of the all point in a partition and count the rows
        val map = scala.collection.mutable.Map.empty[Int, (BV[Double], Int)]
        iter.foreach { point =>
          val idx = finder(point)
          val (sumBV, n) = map.get(idx).getOrElse((BV.zeros[Double](point.size), 0))
          map(idx) = (sumBV + point, n + 1)
        }
        map.toIterator
      }.reduceByKeyLocally {
        // sum the accumulation and the count in the all partition
        case ((p1, n1), (p2, n2)) => (p1 + p2, n1 + n2)
      }.map { case ((idx: Int, (center: BV[Double], counts: Int))) =>
        center :/ counts.toDouble
      }

      val normSum = centers.map(v => breezeNorm(v, 2.0)).sum
      val newNormSum = newCenters.map(v => breezeNorm(v, 2.0)).sum
      error = Math.abs((normSum - newNormSum) / normSum)
      centers = newCenters.toArray
      numIter += 1
      finder = new EuclideanClosestCenterFinder(centers)
    }

    val vectors = centers.map(center => Vectors.fromBreeze(center))
    val nodes = centers.size match {
      case 1 => Array(new ClusterTree(vectors(0), data))
      case 2 => {
        val closest = data.map(point => (finder(point), point))
        centers.zipWithIndex.map { case (center, i) =>
          val subData = closest.filter(_._1 == i).map(_._2)
          subData.cache
          new ClusterTree(vectors(i), subData)
        }
      }
      case _ => throw new RuntimeException(s"something wrong with # centers:${centers.size}")
    }
    nodes
  }
}

/**
 * top-level methods for calling the hierarchical clustering algorithm
 */
object HierarchicalClustering {

  /**
   * Trains a hierarchical clustering model with the given data and the number of clusters
   *
   * NOTE: If there is no splittable cluster, however the number of clusters is
   * less than the given that, the clustering is stopped
   *
   * @param data trained data
   * @param numClusters the maximum number of clusters you want
   * @return a hierarchical clustering model
   *
   *         TODO: The other parameters for the hierarchical clustering will be applied
   */
  def train(data: RDD[Vector], numClusters: Int): HierarchicalClusteringModel = {
    val conf = new HierarchicalClusteringConf()
        .setNumClusters(numClusters)
    val app = new HierarchicalClustering(conf)
    app.train(data)
  }
}

/**
 * this class is used for the model of the hierarchical clustering
 *
 * @param clusterTree a cluster as a tree node
 * @param trainTime the milliseconds for executing a training
 * @param predictTime the milliseconds for executing a prediction
 * @param isTrained if the model has been trained, the flag is true
 * @param clusters the clusters as the result of the training
 */
class HierarchicalClusteringModel private (
  val clusterTree: ClusterTree,
  var trainTime: Int,
  var predictTime: Int,
  var isTrained: Boolean,
  private var clusters: Option[Array[ClusterTree]]) extends Serializable {

  def this(clusterTree: ClusterTree) = this(clusterTree, 0, 0, false, None)

  /**
   * Gets the centers
   * @return the centers of clusters
   */
  def getClusters(): Array[ClusterTree] = {
    if (clusters == None) {
      val clusters = this.clusterTree.toSeq().filter(_.isLeaf())
          .sortWith((a, b) => a.depth() < b.depth()).toArray
      this.clusters = Some(clusters)
    }
    this.clusters.get
  }

  /**
   * Predicts the closest cluster of each point
   * @param data the data to predict
   * @return predicted data
   */
  def predict(data: RDD[Vector]): RDD[(Int, Vector)] = {
    val startTime = System.currentTimeMillis() // to measure the execution time

    val centers = getClusters().map(_.center.toBreeze)
    val finder = new EuclideanClosestCenterFinder(centers)
    data.sparkContext.broadcast(centers)
    data.sparkContext.broadcast(finder)
    val predicted = data.map { point =>
      val closestIdx = finder(point.toBreeze)
      (closestIdx, point)
    }
    this.predictTime = (System.currentTimeMillis() - startTime).toInt
    predicted
  }
}

/**
 * A cluster as a tree node which can have its sub nodes
 *
 * @param data the data in the cluster
 * @param center the center of the cluster
 * @param variance the statistics for splitting of the cluster
 * @param dataSize the data size of its data
 * @param children the sub node(s) of the cluster
 * @param parent the parent node of the cluster
 */
class ClusterTree(
  val center: Vector,
  private[mllib] val data: RDD[BV[Double]],
  private[mllib] var variance: Option[Double],
  private[mllib] var dataSize: Option[Long],
  private[mllib] var children: List[ClusterTree],
  private[mllib] var parent: Option[ClusterTree],
  private[mllib] var isVisited: Boolean) extends Serializable {

  def this(center: Vector, data: RDD[BV[Double]]) =
    this(center, data, None, None, List.empty[ClusterTree], None, false)

  override def toString(): String = {
    val elements = Array(
      s"hashCode:${this.hashCode()}",
      s"dataSize:${this.dataSize.get}",
      s"variance:${this.variance.get}",
      s"parent:${this.parent.hashCode()}",
      s"children:${this.children.map(_.hashCode())}",
      s"isLeaf:${this.isLeaf()}",
      s"isVisited:${this.isVisited}"
    )
    elements.mkString(", ")
  }

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

  private[clustering] def setVariance(variance: Option[Double]): this.type = {
    this.variance = variance
    this
  }

  def getVariance(): Option[Double] = this.variance

  private[clustering] def setDataSize(dataSize: Option[Long]): this.type = {
    this.dataSize = dataSize
    this
  }

  def getDataSize(): Option[Long] = this.dataSize

  private[clustering] def setParent(parent: Option[ClusterTree]) = this.parent = parent

  def getParent(): Option[ClusterTree] = this.parent

  def getChildren(): List[ClusterTree] = this.children

  def isLeaf(): Boolean = (this.children.size == 0)

  /**
   * The flag that the cluster is splittable
   *
   * @return true is splittable
   */
  def isSplittable(): Boolean = {
    this.isLeaf && this.getDataSize != None && this.getDataSize.get >= 2
  }
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
    val breezeData = data.map(_.toBreeze).cache
    // calculates its center
    val pointStat = breezeData.mapPartitions { iter =>
      iter match {
        case iter if iter.isEmpty => Iterator.empty
        case _ => {
          val stat = iter.map(v => (v, 1)).reduce((a, b) => (a._1 + b._1, a._2 + b._2))
          Iterator(stat)
        }
      }
    }.reduce((a, b) => (a._1 + b._1, a._2 + b._2))
    val center = Vectors.fromBreeze(pointStat._1.:/(pointStat._2.toDouble))
    new ClusterTree(center, breezeData)
  }
}

/**
 * Calculates the sum of the variances of the cluster
 */
private[clustering]
class ClusterTreeStatsUpdater private (private var dimension: Option[Int])
    extends Function1[ClusterTree, ClusterTree] with Serializable {

  def this() = this(None)

  /**
   * Calculates the sum of the variances in the cluster
   *
   * @param clusterTree the cluster tree
   * @return the sum of the variances
   */
  def apply(clusterTree: ClusterTree): ClusterTree = {
    val data = clusterTree.data
    if (this.dimension == None) this.dimension = Some(data.first().size)
    val zeroVector = () => Vectors.zeros(this.dimension.get).toBreeze

    // mapper for each partition
    val eachStats = data.mapPartitions { iter =>
      var n = 0.0
      var sum = zeroVector()
      var sumOfSquares = zeroVector()
      val diff = zeroVector()
      iter.foreach { point =>
        n += 1.0
        sum = sum + point
        sumOfSquares = sumOfSquares + (point :* point)
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
    // set the number of rows
    clusterTree.setDataSize(Some(n.toLong))
    // set the sum of the variances of each element
    val variance = n match {
      case n if n > 1 => (sumOfSquares.:*(n) - (sum :* sum)) :/ (n * (n - 1.0))
      case _ => zeroVector()
    }
    clusterTree.setVariance(Some(variance.toArray.sum))

    clusterTree
  }
}
