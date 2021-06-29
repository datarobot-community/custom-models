package custom

import com.datarobot.drum._
import collection.JavaConverters._
import ml.dmlc.xgboost4j.scala.DMatrix
import ml.dmlc.xgboost4j.scala.Booster;
import ml.dmlc.xgboost4j.scala.XGBoost;
import ml.dmlc.xgboost4j.LabeledPoint

import org.apache.commons.csv.CSVFormat;

import java.io.{BufferedReader, ByteArrayInputStream, InputStreamReader}

import java.util.HashMap

import util.{Try, Success, Failure}
import java.nio.file.Paths


class XGBoostPredictor(name: String) extends BasePredictor(name) 
{

    var customModelPath: String = null
    var negativeClassLabel: String = null 
    var positiveClassLabel: String = null
    var booster: Booster = null
    val features = Array("SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm")
    val numFeatures = features.length

    override def configure(
      params: java.util.Map[String, AnyRef] = new java.util.HashMap[String, AnyRef]()
    ) = {
        customModelPath = params.get("__custom_model_path__").asInstanceOf[String]
        negativeClassLabel = params.get("negativeClassLabel").asInstanceOf[String]
        positiveClassLabel = params.get("positiveClassLabel").asInstanceOf[String]
        val modelPath = Paths.get(customModelPath, "xgb-model", "model.bin").toFile
        modelPath exists match { 
          case false => throw new Exception(s"${modelPath} does not exist")
          case true => null 
        }
        booster = XGBoost.loadModel(modelPath.toString)
    }
    override def predict(inputBytes: Array[Byte]): String = {
        val reader = new BufferedReader(new InputStreamReader(new ByteArrayInputStream(inputBytes)))
        val csvFormat = CSVFormat.DEFAULT.withFirstRecordAsHeader;
        val parser = csvFormat.parse(reader)
        val sParser = parser.iterator.asScala.map { _.toMap }
        val dataIter = sParser.map{ row => 
          val rs = row.asScala.filter{ case(k,v) => features.contains(k)}.map{ _._2}
          LabeledPoint(0f, numFeatures, null, rs.toArray.map{_.toFloat})
        }.toIterator
        val dmatrix = new DMatrix(dataIter)
        val predictions = booster.predict(dmatrix).map{ p => 
          val p1 = p(0)
          val p0 = 1 - p1
          s"${p0},${p1}"
        }
        predictions.mkString(s"${negativeClassLabel},${positiveClassLabel}\n", "\n", "")
    }
 
}
