package org.bits.mtech.dse.ml.clustering

import org.apache.spark.rdd.RDD
import scala.math.pow

object KMeans {

  case class Customer(age:Int, income: Int, spendingScore: Int)

  /**
   * Squared distance
   *
   * @param c1 customer/vector/point 1
   * @param c2 customer/vector/point 1
   * @return returns a double value of squared distance
   */
  def distanceSquared(c1: Customer, c2: Customer) =
    Math.sqrt(
      pow(c1.income - c2.income, 2) +
        pow(c1.spendingScore - c2.spendingScore, 2) +
        pow(c1.age - c2.age, 2)
    )

  /**
   * * summation of two points
   * * @param c1
   * * @param c2
   * * @return
   */
  def addPoints(c1: Customer, c2: Customer) =
    Customer(c1.age + c2.age, c1.income + c2.income, c1.spendingScore + c2.spendingScore)

  /**
   * Given a random point and k-center points, this method returns a closed center point
   *
   * @param point     random point
   * @param kClusters array of K center points, till this iteration
   * @return closest center point for given random point
   */
  def closestCenterPoint(point: Customer, kClusters: Array[Customer]): Customer =
    kClusters.map(center => (distanceSquared(center, point), center)).sortBy(_._1).head._2

  /**
   * Loops until the clusters converge on given condition and returns final cluster centers
   * @param tempDist
   * @param convergeCutOff
   * @param currentCenterPoints
   * @param inp
   * @return
   */
  def computeNewCenters(K: Int, tempDist: Double, convergeCutOff:Double, currentCenterPoints: Array[(Customer,Int)], inp: RDD[Customer], clusteredData:RDD[(Customer, Customer)]): RDD[(Customer, Customer)] = {
    val currentCenters = currentCenterPoints.map(_._1).toArray
//    if (tempDist <= convergeCutOff) currentCenterPoints
    if (tempDist <= convergeCutOff) clusteredData
    else {
      val closestToKPoints = inp.map(point => (closestCenterPoint(point, currentCenters), (point,1))) //Finds nearest cluster center for given random point
//      closestToKPoints.foreach(println)
      val pointsCalculated = closestToKPoints.reduceByKey((acc, point) => (addPoints(acc._1, point._1), acc._2 + point._2))
//      pointsCalculated.foreach(println)
      val newCentersAndCounts = pointsCalculated.map(center => (Customer(center._2._1.age/center._2._2, center._2._1.income/center._2._2, center._2._1.spendingScore/center._2._2),center._2._2))
      val newCenters = newCentersAndCounts.map(_._1).collect()
      //      newCenters.foreach(println)

      val newDist = (0 until K).map(idx => distanceSquared(currentCenters(idx), newCenters(idx))).sum
      println(newDist)

      val clusteredData = closestToKPoints.map(c => (c._1, c._2._1))
      computeNewCenters(K, newDist, convergeCutOff, newCentersAndCounts.collect(), inp, clusteredData)
  }
  }

  def assignLabels(clusterCenters: Array[(Customer, Int)]) = {
    def assign(customer: Customer, cnt: Int) = customer match {
      case Customer(25,25,77) => (customer, cnt, "CLUSTER_1")
      case Customer(33,57,49) => (customer, cnt, "CLUSTER_2")
      case Customer(41,88,16) => (customer, cnt, "CLUSTER_3")
      case Customer(54,40,36) => (customer, cnt, "CLUSTER_4")
      case Customer(32,86,82) => (customer, cnt, "CLUSTER_5")
    }
    clusterCenters.map(c => assign(c._1, c._2))
  }

}