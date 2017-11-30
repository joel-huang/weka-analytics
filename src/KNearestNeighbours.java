import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Debug;
import weka.core.Instances;

import java.io.*;
import java.util.Scanner;

public class KNearestNeighbours {

    // Number of folds for the k-fold cross validation
    private static int folds = 2;

    public static void main(String[] args) throws IOException {

        File folder = new File("src/");
        Scanner scanner = new Scanner(System.in);
        int fileCount = 0;
        File[] listOfFiles = folder.listFiles();
        System.out.println("Choose a file to train from:");
        for (File f : listOfFiles) {
            fileCount++;
            System.out.println(fileCount + ". " + f.getName());
        }
        fileCount = 0;
        File selectedTrainingFile = listOfFiles[scanner.nextInt() - 1];
        System.out.println("Choose a file to test:");
        for (File f : listOfFiles) {
            fileCount++;
            System.out.println(fileCount + ". " + f.getName());
        }
        File selectedTestFile = listOfFiles[scanner.nextInt() - 1];

        // Read the training data
        File trainingData = new File(selectedTrainingFile.toString());
        InputStream trainingInputStream = new FileInputStream(trainingData);
        BufferedReader trainingReader = new BufferedReader(new InputStreamReader(trainingInputStream));

        // Build the kNN model
        Instances data = new Instances(trainingReader);

        // setClassIndex is used to define the attribute that will represent the class (for prediction purposes).
        // Given that the index starts at zero, data.numAttributes() - 1 represents the last attribute of the test data set.
        data.setClassIndex(data.numAttributes() - 1);

        // K-nearest neighbours classifier. Can select appropriate value of K based on cross-validation. Can also do distance weighting.
        Classifier ibk = new IBk();
        ibk.setDebug(true);

        try {
            ibk.buildClassifier(data);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Load the test data
        File testData = new File(selectedTestFile.toString());
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
                if (predictedValue == actualValue) {
                    System.out.println("Correctly predicted " + predictedValue + ", actual value: " + actualValue);
                    correctResults++;
                }
                else {
                    System.out.println("Wrongly predicted " + predictedValue + ", actual value: " + actualValue);
                    wrongResults++;
                }
            } catch (Exception e) {
                e.printStackTrace();
            }

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
            evaluation.crossValidateModel(ibk, test, folds, random);
            System.out.println("Confusion Matrix: " + evaluation.toMatrixString());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}