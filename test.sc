val l = List(("raja",2), ("brahma", 2), ("gopi", 10), ("raja",3))
//l.groupBy(_._1).mapValues(_.size).foreach(println)

val index = 0 until l.size
for ( i <- index)
  println(l(i))