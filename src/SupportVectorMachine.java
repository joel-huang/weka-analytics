import weka.classifiers.Evaluation;
import weka.core.*;
import wlsvm.WLSVM;

import java.io.*;
import java.util.Scanner;

public class SupportVectorMachine {

    // Inputs to the test Instance.
    private static double sepalLength;
    private static double sepalWidth;
    private static double petalLength;
    private static double petalWidth;

    // Store the trained model.

    private static void getInput() {
        Scanner scanner = new Scanner(System.in);
        System.out.println("Enter Sepal Length:");
        sepalLength = scanner.nextDouble();
        System.out.println("Enter Sepal Width:");
        sepalWidth = scanner.nextDouble();
        System.out.println("Enter Petal Length:");
        petalLength = scanner.nextDouble();
        System.out.println("Enter Petal Width:");
        petalWidth = scanner.nextDouble();
        scanner.close();
    }

    private static void trainModel(WLSVM svmClass) throws Exception {
        // Read the training data
        File trainingData = new File("src/iris_train.arff");
        InputStream trainingInputStream = new FileInputStream(trainingData);
        BufferedReader trainingReader = new BufferedReader(new InputStreamReader(trainingInputStream));

        // Model the SVM
        Instances data = new Instances(trainingReader);
        data.setClassIndex(data.numAttributes() - 1);
        svmClass.buildClassifier(data);
    }

    public static void main(String[] args) throws Exception {

        WLSVM svmClass = new WLSVM();
        trainModel(svmClass);

        // Get input as the test
        getInput();

        // Build the Instance template
        Attribute attribute1 = new Attribute("sepallength");
        Attribute attribute2 = new Attribute("sepalwidth");
        Attribute attribute3 = new Attribute("petallength");
        Attribute attribute4 = new Attribute("petalwidth");

        // Declare the class attribute and its values
        FastVector fvClassVal = new FastVector(3);
        fvClassVal.addElement("Iris-setosa");
        fvClassVal.addElement("Iris-versicolor");
        fvClassVal.addElement("Iris-virginica");
        Attribute classAttribute = new Attribute("class", fvClassVal);

        // Declare the feature vector template
        FastVector fvWekaAttributes = new FastVector(5);
        fvWekaAttributes.addElement(attribute1);
        fvWekaAttributes.addElement(attribute2);
        fvWekaAttributes.addElement(attribute3);
        fvWekaAttributes.addElement(attribute4);
        fvWekaAttributes.addElement(classAttribute);

        // Creating testing instances object with name "TestingInstance"
        // using the feature vector template we declared above
        // and with initial capacity of 1
        Instances testingSet = new Instances("TestingInstance", fvWekaAttributes, 1);
        // Setting the column containing class labels:
        testingSet.setClassIndex(testingSet.numAttributes() - 1);
        // Create and fill an instance, and add it to the testingSet
        Instance test = new Instance(testingSet.numAttributes());
        test.setValue((Attribute)fvWekaAttributes.elementAt(0), sepalLength);
        test.setValue((Attribute)fvWekaAttributes.elementAt(1), sepalWidth);
        test.setValue((Attribute)fvWekaAttributes.elementAt(2), petalLength);
        test.setValue((Attribute)fvWekaAttributes.elementAt(3), petalWidth);
        test.setValue((Attribute)fvWekaAttributes.elementAt(4), "Iris-setosa");

        testingSet.add(test);

        double predictedResult = svmClass.classifyInstance(testingSet.instance(0));

        String predictedResultString = "";

        if (predictedResult == 0) predictedResultString = "Iris-setosa";
        else if (predictedResult == 1) predictedResultString = "Iris-versicolor";
        else if (predictedResult == 2) predictedResultString = "Iris-virginica";
        else predictedResultString = "Error";

        System.out.println("Classification result: " + predictedResultString);

    }
}