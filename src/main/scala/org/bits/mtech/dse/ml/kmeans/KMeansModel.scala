package org.bits.mtech.dse.ml.kmeans

import org.apache.spark.rdd.RDD
import scala.math.pow

object KMeansModel {

  case class Customer(income: Int, spendingScore: Int)

  /**
   * Squared distance
   *
   * @param c1 customer/vector/point 1
   * @param c2 customer/vector/point 1
   * @return returns a double value of squared distance
   */
  def distanceSquared(c1: Customer, c2: Customer) =
    Math.sqrt(pow(c1.income - c2.income, 2) + pow(c1.spendingScore - c2.spendingScore, 2))

  /**
   * * summation of two points
   * * @param c1
   * * @param c2
   * * @return
   */
  def addPoints(c1: Customer, c2: Customer) =
    Customer(c1.income + c2.income, c1.spendingScore + c2.spendingScore)

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
  def computeNewCenters(K: Int, tempDist: Double, convergeCutOff:Double, currentCenterPoints: Array[Customer], inp: RDD[Customer]): Array[Customer] = {
    if (tempDist <= convergeCutOff) currentCenterPoints
    else {
      val closestToKPoints = inp.map(point => (closestCenterPoint(point, currentCenterPoints), (point,1))) //Finds nearest cluster center for given random point
//      closestToKPoints.foreach(println)
      val pointsCalculated = closestToKPoints.reduceByKey((acc, point) => (addPoints(acc._1, point._1), acc._2 + point._2))
//      pointsCalculated.foreach(println)
      val newCenters = pointsCalculated.map(center => Customer(center._2._1.income/center._2._2, center._2._1.spendingScore/center._2._2)).collect()
//      newCenters.foreach(println)

      val newDist = (0 until K).map(idx => distanceSquared(currentCenterPoints(idx), newCenters(idx))).sum
      println(newDist)
      computeNewCenters(K, newDist, convergeCutOff, newCenters, inp)
    }
  }

}