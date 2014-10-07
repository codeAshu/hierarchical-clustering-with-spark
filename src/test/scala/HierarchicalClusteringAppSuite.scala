import org.apache.spark.util.LocalSparkContext
import org.scalatest.FunSuite

class HierarchicalClusteringAppSuite extends FunSuite with LocalSparkContext {

  test("main") {
    val master = "local"
    val maxCores = 1
    val rows = 10000
    val dimension = 3
    val numClusters = 10
    val numPartitions = 4
    val args = Array(master, maxCores, rows, dimension, numClusters, numPartitions).map(_.toString)
    HierarchicalClusteringApp.main(args)
  }
}
