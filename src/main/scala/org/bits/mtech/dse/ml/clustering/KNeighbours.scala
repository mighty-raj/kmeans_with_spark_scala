package org.bits.mtech.dse.ml.clustering

import org.bits.mtech.dse.ml.clustering.KMeans.{Customer, distanceSquared}

object KNeighbours {

  /**
   * Evaluates nearest K neighbours and maps query point => cluster
   * @param query new query data point
   * @param train training data
   * @param K K neighbours to be considered
   * @return Returns a tuple, mapping query to nearest neighbour cluster
   */
  def neighbour(query: Customer, train:Array[(Customer, String)], K: Int) = {

    val distanceAndCluster = train.map(c => (distanceSquared(query, c._1), c._2))
    val topK = distanceAndCluster.sortWith(_._1 < _._1).take(K)
    val swapped = topK.map(_.swap)
    val clusterCounts = swapped.groupBy(_._1).mapValues(_.size)
    val finalCluster = clusterCounts.toSeq.sortWith(_._1 > _._1)
    (query, finalCluster.head._1)

  }
}
