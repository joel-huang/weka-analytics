import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Debug;
import weka.core.Instances;

import java.io.*;

public class KNearestNeighbours {

    public static void main(String[] args) throws IOException {

        // Read the training data
        File trainingData = new File("src/iris_train.arff");
        InputStream trainingInputStream = new FileInputStream(trainingData);
        BufferedReader trainingReader = new BufferedReader(new InputStreamReader(trainingInputStream));

        // Build the kNN model
        Instances data = new Instances(trainingReader);
        data.setClassIndex(data.numAttributes() - 1);
        Classifier ibk = new IBk();
        try {
            ibk.buildClassifier(data);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Load the test data
        File testData = new File("src/iris_test.arff");
        InputStream testInputStream = new FileInputStream(testData);
        BufferedReader testReader = new BufferedReader(new InputStreamReader(testInputStream));

        // Classify
        Instances test = new Instances(testReader);
        test.setClassIndex(test.numAttributes() - 1);

        int correctResults = 0;
        int wrongResults = 0;

        for (int i = 0; i < test.numInstances(); i++) {
            double predictedValue = 0;
            double actualValue = test.instance(i).classValue();
            try {
                predictedValue = ibk.classifyInstance(test.instance(i));
            } catch (Exception e) {
                e.printStackTrace();
            }
            if (predictedValue == actualValue) correctResults++;
            else wrongResults++;
        }

        System.out.println("Correct results: " + correctResults);
        System.out.println("Wrong results: " + wrongResults);

        double correctResultsDouble = correctResults;
        double accuracy = correctResultsDouble / test.numInstances() * 100;
        System.out.println("Accuracy: " + accuracy + "%");
        Evaluation evaluation;

        // Confusion matrix
        Debug.Random random = new Debug.Random(10000);

        try {
            evaluation = new Evaluation(test);
            evaluation.crossValidateModel(ibk, test, 10, random);
            System.out.println("Confusion Matrix: " + evaluation.toMatrixString());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}