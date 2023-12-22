package org.example;

import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class Kmeans {
    public static void main(String[] args) {
        SparkSession ss= SparkSession.builder().appName("Tp spark ml").master("local[*]").getOrCreate();
        Dataset<Row> dataset =ss.read().option("inferSchema",true).option("header",true).csv("Mall_Customers.csv");
        VectorAssembler assembler=new VectorAssembler().setInputCols(new String[]{"Age","Annual Income (k$)","Spending Score (1-100)"}
        ).setOutputCol("features");
        Dataset<Row> assembledDataset = assembler.transform(dataset);
        MinMaxScaler scaler = new MinMaxScaler().setInputCol("features").setOutputCol("normalizeFeatures");
        Dataset<Row> normalizedDS = scaler.fit(assembledDataset).transform(assembledDataset);
        normalizedDS.printSchema();
        KMeans kMeans = new KMeans().setK(5).setSeed(123).setFeaturesCol("normalizeFeatures").setPredictionCol("prediction");
        KMeansModel model = kMeans.fit(normalizedDS);
        Dataset<Row> prediction = model.transform(normalizedDS);
        prediction.show(200);
        ClusteringEvaluator evaluator = new ClusteringEvaluator();
        double score = evaluator.evaluate(prediction);
        System.out.println(score);
    }
}
