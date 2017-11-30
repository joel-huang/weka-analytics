import weka.core.*;
import wlsvm.WLSVM;

import java.io.*;
import java.util.Scanner;

public class SupportVectorMachine4 {

    // Inputs to the test Instance.
    private static double attribute1;
    private static double attribute2;
    private static double attribute3;
    private static double attribute4;

    // Store the trained model.

    private static void getInput() {
        Scanner scanner = new Scanner(System.in);
        System.out.println("Enter Attribute 1:");
        attribute1 = scanner.nextDouble();
        System.out.println("Enter Attribute 2:");
        attribute2 = scanner.nextDouble();
        System.out.println("Enter Attribute 3:");
        attribute3 = scanner.nextDouble();
        System.out.println("Enter Attribute 4:");
        attribute4 = scanner.nextDouble();
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
        Attribute attributeA = new Attribute("attribute1");
        Attribute attributeB = new Attribute("attribute2");
        Attribute attributeC = new Attribute("attribute3");
        Attribute attributeD = new Attribute("attribute4");

        // Declare the class attribute and its values
        FastVector fvClassVal = new FastVector(3);
        fvClassVal.addElement("Iris-setosa");
        fvClassVal.addElement("Iris-versicolor");
        fvClassVal.addElement("Iris-virginica");
        Attribute classAttribute = new Attribute("class", fvClassVal);

        // Declare the feature vector template
        FastVector fvWekaAttributes = new FastVector(3);
        fvWekaAttributes.addElement(attributeA);
        fvWekaAttributes.addElement(attributeB);
        fvWekaAttributes.addElement(attributeC);
        fvWekaAttributes.addElement(attributeD);
        fvWekaAttributes.addElement(classAttribute);

        // Creating testing instances object with name "TestingInstance"
        // using the feature vector template we declared above
        // and with initial capacity of 1
        Instances testingSet = new Instances("TestingInstance", fvWekaAttributes, 1);

        // Setting the column containing class labels:
        testingSet.setClassIndex(testingSet.numAttributes() - 1);

        // Create and fill an instance, and add it to the testingSet
        Instance test = new Instance(testingSet.numAttributes());
        test.setValue((Attribute)fvWekaAttributes.elementAt(0), attribute1);
        test.setValue((Attribute)fvWekaAttributes.elementAt(1), attribute2);
        test.setValue((Attribute)fvWekaAttributes.elementAt(2), attribute3);
        test.setValue((Attribute)fvWekaAttributes.elementAt(3), attribute4);
        test.setValue((Attribute)fvWekaAttributes.elementAt(4), "Iris-setosa");
        testingSet.add(test);

        double predictedResult = svmClass.classifyInstance(testingSet.instance(0));

        String predictedResultString;

        if (predictedResult == 0) predictedResultString = "Iris-setosa";
        else if (predictedResult == 1) predictedResultString = "Iris-versicolor";
        else if (predictedResult == 2) predictedResultString = "Iris-virginica";
        else predictedResultString = "Error";

        System.out.println("Classification result: " + predictedResultString);
    }
}