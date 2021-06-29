import ml.dmlc.xgboost4j.scala.DMatrix
import ml.dmlc.xgboost4j.scala.Booster;
import ml.dmlc.xgboost4j.scala.XGBoost;
import ml.dmlc.xgboost4j.LabeledPoint
import java.io._
import java.nio.file.Paths

object TrainXGB extends App {

  val dir = new File(args(0))
  dir.exists match { 
      case true => null
      case false => dir.mkdirs
  }

  val data = scala.io.Source
    .fromFile(
      "data/iris_binary_training.csv"
    )
    .getLines
  val positiveClassLabel = "Iris-versicolor"
  val negativeClassLabel = "Iris-setosa"
  val headers = data.next
  val nullArray = null.asInstanceOf[Array[Int]]
  val dataIter = data.map { row =>
    val d = row.split(",")
    val len = d.length - 1
    val (features, label) = d.splitAt(len)
    val label_bin = label.apply(0) match {
      case "Iris-setosa"     => 0f
      case "Iris-versicolor" => 1f
      case _                 => throw new Exception("not set for multiclass")
    }
    LabeledPoint(
      label = label_bin,
      len - 1,
      indices = nullArray,
      values = features.map { _.toFloat }.tail
    )
  }

  val dmatrix = new DMatrix(dataIter.toIterator, cacheInfo = null)
  val paramMap = List(
    "eta" -> 0.1,
    "max_depth" -> 5,
    "objective" -> "binary:logistic",
    "verbosity" -> 1
  ).toMap
  // number of iterations
  val round = 100
  val booster =
    XGBoost.train(dmatrix, paramMap, round, earlyStoppingRound = 200)

  val modelPath = Paths.get(dir.toString, "model.bin").toFile
  booster.saveModel(modelPath.toString)

}
