package org.bits.mtech.dse.ml.kmeans

import org.apache.log4j.Level
import org.apache.spark.{SparkConf, SparkContext}
import org.bits.mtech.dse.ml.kmeans.KMeansModel._

object KMeans {

  def main(args: Array[String]) = {
    val inpPath = args(0)
    val outPath = args(1)

    val conf = new SparkConf().setAppName("kmeans-basic").setMaster("local[2]")
    val sc = new SparkContext(conf)
    sc.setLogLevel(Level.WARN.toString)

    val inpRdd = sc.textFile(inpPath)

    val header = inpRdd.first()
    val customersInp = inpRdd.filter(_ != header).
      map(_.split(",")).
//      map(c => Customer(c(0), c(1), c(2).toInt, c(3).toInt, c(4).toInt))
      map(c => Customer(c(3).toInt, c(4).toInt))

    val K = 3 // K is the number of means (center points of clusters) to find
    val convergeDist = .1 // ConvergeDist -- the threshold "distance" between iterations at which we decide we are done

//    pick K points randomly as center points
    var kPoints = customersInp.takeSample(false, K, 35)

//    K Center point initialized
    kPoints.foreach(println)

    val finalCentroids = computeNewCenters(K, Double.PositiveInfinity, convergeDist, kPoints, customersInp)

    // Display the final center points
    println("Final center points :")
    finalCentroids.foreach(println)
    sc.stop()

  }

}
