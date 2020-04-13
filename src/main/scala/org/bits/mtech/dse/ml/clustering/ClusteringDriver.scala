package org.bits.mtech.dse.ml.clustering

import org.apache.log4j.Level
import org.apache.spark.{SparkConf, SparkContext}
import org.bits.mtech.dse.ml.clustering.KMeans._
import org.bits.mtech.dse.ml.clustering.KNeighbours._

object ClusteringDriver {

  def main(args: Array[String]) = {
    val inpPath = args(0)

    val conf = new SparkConf().setAppName("CustomerSegmentation-KMeans-KNN").setMaster("local[2]")
    val sc = new SparkContext(conf)
    sc.setLogLevel(Level.WARN.toString)

    val inpRdd = sc.textFile(inpPath)

    val header = inpRdd.first()
    val customersInp = inpRdd.filter(_ != header).
      map(_.split(",")).
      map(c => Customer(c(2).toInt, c(3).toInt, c(4).toInt))

    val K = 5 // K is the number of means (center points of clusters) to find
    val convergeDist = .1 // ConvergeDist -- the threshold "distance" between iterations at which we decide we are done

//    pick K points randomly as center points
    var kPoints = customersInp.takeSample(false, K, 35)

//    K Center point initialized
//    kPoints.foreach(println)

    val finalClusters = computeNewCenters(K, Double.PositiveInfinity, convergeDist, kPoints.map((_,0)), customersInp, sc.emptyRDD)
    val finalCentroids = finalClusters.map(c => (c._1,1)).reduceByKey(_+_).collect()

    val labelledCenters = assignLabels(finalCentroids)
    val finalClustersRdd = sc.parallelize(labelledCenters.sortWith(_._3 < _._3))

    // Display the final center points
    println
    println("Final Centroids Below:")
    println("-----------------------------------------------------------------------------------")
    printf("%-50s %-15s %-15s\n", "Centroid", "Cluster", "Count")
    println("-----------------------------------------------------------------------------------")
    finalClustersRdd.foreach(x => printf("%-50s %-15s %-15d\n", x._1.toString, x._3.toString, x._2))
    println("-----------------------------------------------------------------------------------")

    val clusterAndLabel = finalClustersRdd.map(c=> (c._1, c._3))
    val labelledInput = finalClusters.join(clusterAndLabel)
    println
    println("Labels Assigned ...  ")
//    labelledInput.foreach(println)

//    split clustered inp dataset into Training & Query Datasets @ 70:30 ratio
    val splitRatio = Array(0.7,0.2)
    val splittedRdd = labelledInput.randomSplit(splitRatio, 20)

    val trainingData = splittedRdd(0).map(_._2)
    val actualData = splittedRdd(1).map(_._2).collect()
    val queryData = splittedRdd(1).map(_._2._1)

    val cartesianRdd = queryData.cartesian(trainingData)
//    cartesianRdd.foreach(println)

    println
    println("Apply model to predict on query data ... ")

    val predictedData = queryData.collect().map(query => neighbour(query, trainingData.collect(), 3))

    println
    println("cartesian rdd count => " + cartesianRdd.count)
    println("training count => " + trainingData.count())
    println("query data count => " + queryData.count())
    println("predicted data count => " + predictedData.size)

    displayConfusionMatrix(actualData, predictedData)

    sc.stop()
    System.exit(1)

  }

  /**
   * Displays confusion matrix
   * @param actual Actual Data
   * @param predicted Predicted Data
   */
  def displayConfusionMatrix(actual: Array[(Customer, String)], predicted: Array[(Customer, String)]) = {

    def div(x: Int, y: Int) = x.toDouble/y.toDouble

    val act = actual.toList
    val pred = predicted.toList

    val actClusterCounts=act.groupBy(_._2).mapValues(_.size)
    val predClusterCounts= pred.groupBy(_._2).mapValues(_.size)

    val joinedList = act.map(tup1 => pred.filter(tup2 => tup1._1==tup2._1).map(tup2 => (tup1._1, (tup1._2, tup2._2)))).flatten
    val correctlyPredicted =  joinedList.filter(x => (x._2._1 == x._2._2))

    val correctByCluster = correctlyPredicted.map(_._2).groupBy(_._1).mapValues(_.size)

    val precisionByCluster = correctByCluster.map(x => (x._1, div(x._2, predClusterCounts.get(x._1).get))).toList.sortWith(_._1 < _._1)
    val recallByCluster = correctByCluster.map(x => (x._1, div(x._2, actClusterCounts.get(x._1).get)))

    //Display confusion matrix
    println
    println("-------------------------------------------")
    printf("%-15s %-15s %-15s\n", "CLUSTER", "PRECISION", "RECALL")
    println("-------------------------------------------")
    for(idx <- 0 until precisionByCluster.size) {
      val clstr = precisionByCluster(idx)._1
      val prcn = precisionByCluster(idx)._2
      val acc = recallByCluster(clstr)

      printf("%-15s %-15.2f %-15.2f\n", clstr, prcn, acc )
    }
    println("-------------------------------------------")

    println
    printf("Accuracy => %10.2f", (correctlyPredicted.size.toDouble/joinedList.size.toDouble) )

  }

}
