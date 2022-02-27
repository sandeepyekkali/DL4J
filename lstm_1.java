package org.deeplearning4j.examples.quickstart.modeling.recurrent;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

import static java.lang.Math.abs;

public class beginnings {

    private static Logger log = LoggerFactory.getLogger(beginnings.class);

    public static void main(String[] args) throws IOException, InterruptedException {

        int size = 1000;
        Random r = new Random(12345);
        Double[] numbersInitial = new Double[size];
        for (int n = 0; n<1000; n++){
            //double d = n;
            numbersInitial[n] = r.nextDouble();
        }

        Double[] numbers = new Double[size-1];

        for (int n = 0; n<size-1; n++){
            double d = n;
            numbers[n] = numbersInitial[n];
        }


        List<Double> numList = Arrays.asList(numbers);
        int timeSteps = 2;


        //split train and test

        int split = (int) (numList.size()*0.7);
        List<Double> trainData = numList.subList(0, split);
        List<Double> testData = numList.subList(split, numList.size());

//          System.out.println(trainData.get(0) + trainData.get(7));
//        NormalizerMinMaxScaler myNormalizer = new NormalizerMinMaxScaler();
//        myNormalizer.fit();

        INDArray trainArray = Nd4j.zeros( trainData.size()-timeSteps,timeSteps,1);
        INDArray trainLabels = Nd4j.zeros(trainData.size()-timeSteps,1,1);

        for (int i = 0; i < trainData.size()-timeSteps; i++) {

            List<Double> trainDataTimeSteps = trainData.subList(i, i+timeSteps);
            //INDArray trainDataVector = Nd4j.zeros(1,timeSteps);
            for(int j=0; j<timeSteps; j++){
                trainArray.putScalar(new int[]{i,j,0}, trainDataTimeSteps.get(j));
            }
            trainLabels.putScalar(new int[]{i,0,0}, trainData.get(i+timeSteps));
        }

        INDArray testArray = Nd4j.zeros(  testData.size()-timeSteps,timeSteps,1);
        INDArray testLabels = Nd4j.zeros( testData.size()-timeSteps,1,1);

        for (int i = 0; i < testData.size()-timeSteps; i++) {

            List<Double> testDataTimeSteps = testData.subList(i, i+timeSteps);
            //INDArray trainDataVector = Nd4j.zeros(1,timeSteps);
            for(int j=0; j<timeSteps; j++){
                testArray.putScalar(new int[]{i,j,0}, testDataTimeSteps.get(j));
            }
            testLabels.putScalar(new int[]{i,0,0}, testData.get(i+timeSteps));
        }

        //Create datasets for train and test
        DataSet trainDataSet = new DataSet(trainArray, trainLabels);
        DataSet testDataSet = new DataSet(testArray, testLabels);

        //System.out.println(trainDataSet);
        //System.out.println(trainLabels);
        //System.out.println(testDataSet);

        //Normalize dataset to minmax
        NormalizerMinMaxScaler minMaxScalerTrain = new NormalizerMinMaxScaler();
        NormalizerMinMaxScaler minMaxScalerTest = new NormalizerMinMaxScaler();

        minMaxScalerTrain.fitLabel(true);
        minMaxScalerTrain.fit(trainDataSet);
        minMaxScalerTrain.transform(trainDataSet);

        //minMaxScalerTest.fitLabel(true);
        //minMaxScalerTest.fit(testDataSet);
        minMaxScalerTrain.transform(testDataSet);


        //System.out.println(trainDataSet);
        //System.out.println(testLabels);

        //Start Neural Network code
        int hiddenLayerNeurons = 50;
        int hiddenLayersNumber = 1;

        //ListBuilder builder = new NeuralNetConfiguration.Builder();

        NeuralNetConfiguration.Builder nnBuilder = new NeuralNetConfiguration.Builder();
        nnBuilder.seed(123);
        nnBuilder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        nnBuilder.biasInit(0);
        //nnBuilder.l2(0.0005);
        nnBuilder.miniBatch(true);
        nnBuilder.updater(new Adam());
        nnBuilder.weightInit(WeightInit.RELU);

        ListBuilder listBuilder = nnBuilder.list();

        for (int i = 0; i < hiddenLayersNumber; i++) {
            LSTM.Builder hiddenLayerBuilder = new LSTM.Builder();
            hiddenLayerBuilder.nIn(i == 0 ? timeSteps : hiddenLayerNeurons);
            hiddenLayerBuilder.nOut(hiddenLayerNeurons);
            hiddenLayerBuilder.activation(Activation.RELU);
            listBuilder.layer(i, hiddenLayerBuilder.build());
        }

        //Output layer
        RnnOutputLayer.Builder outputLayerBuilder = new RnnOutputLayer.Builder();
        outputLayerBuilder.lossFunction(LossFunctions.LossFunction.MSE);
        outputLayerBuilder.activation(Activation.RELU);
        outputLayerBuilder.nIn(hiddenLayerNeurons);
        outputLayerBuilder.nOut(1);
        listBuilder.layer(hiddenLayersNumber, outputLayerBuilder.build());

        //Build network
        MultiLayerConfiguration conf = listBuilder.build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.setListeners(new ScoreIterationListener(10));

        //Run train data
        for (int z = 0; z < 3000; z++) {
            network.fit(trainDataSet);
        }

        //Evaluate test data
        //Evaluation eval = new Evaluation(1);
        INDArray output = network.output(testDataSet.getFeatures());
        double sum = 0, metric;

        minMaxScalerTrain.revert(testDataSet);
        minMaxScalerTrain.revertLabels(output);
        for(int e=0; e<output.size(0); e++){
          sum = sum+ abs(output.getDouble(e,0,0) - testDataSet.getLabels().getDouble(e,0,0));
          System.out.println(testDataSet.getLabels().getDouble(e,0,0) +" - " + output.getDouble(e,0,0)+"|");
        }
        metric = sum/output.size(0);
        System.out.println("Error Metric: " + metric +"\nSize: "+output.size(0));
        //System.out.println("\n"+testDataSet.getLabels().getDouble(3,0,0));
        //System.out.println(testDataSet.getLabels() +"\n" + output);
        //eval.eval(testDataSet.getLabels(), output);
        //log.info(eval.stats());

        //Save file to disk
        int toSave;
        Scanner scan = new Scanner(System.in);
        System.out.println("Do you want to save the model(1)?");
        toSave = scan.nextInt();

        if(toSave == 1){
            //Save trained model
            String locationToSave = "E:/dl4jpractice/persistModels/";
            File fileName = new File(locationToSave + "oilDaily.zip");

            ModelSerializer.writeModel(network, fileName, true);
            System.out.println("Model Saved");
        }
    }
}
