package jae.muzzin.semeval;

import com.robrua.nlp.bert.Bert;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.stream.IntStream;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.weightinit.impl.XavierInitScheme;

/**
 *
 * @author Admin
 */
public class Semeval {

    public static String path = "model.fb";

    public static void main(String[] args) {
        try ( Bert bert = Bert.load("com/robrua/nlp/easy-bert/bert-uncased-L-12-H-768-A-12")) {
            float[] embedding = bert.embedSequence("The EU should extend the model of Pooling & Sharing European military resources beyond air transport.");
            System.out.println(Arrays.toString(embedding));
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
            ex.printStackTrace();
        }
    }

    
    public static INDArray encodeDataset(String path, String writePath) {
        try ( Bert bert = Bert.load("com/robrua/nlp/easy-bert/bert-uncased-L-12-H-768-A-12")) {
            INDArray r = Nd4j.create(data.length, 768 * 2 + 1);
            for (int i = 0; i < data.length; i++) {
                System.out.println("Encoded " + data[i][0]);
                float[] s1 = bert.embedSequence(data[i][3]);
                float[] s2 = bert.embedSequence(data[i][1]);
                float[] a = new float[]{data[i][2].equals("against") ? -1 : 1};
                r.putRow(i, Nd4j.concat(0, Nd4j.create(s1), Nd4j.create(s2), Nd4j.create(a)));
            }
            return r;
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
            ex.printStackTrace();
            throw ex;
        }
    }
    //accepts rows in s2,a,s1 format
    //returns rows in s1,s2,a format
    public static INDArray encodeDataset(String[][] data) {
        try ( Bert bert = Bert.load("com/robrua/nlp/easy-bert/bert-uncased-L-12-H-768-A-12")) {
            INDArray r = Nd4j.create(data.length, 768 * 2 + 1);
            for (int i = 0; i < data.length; i++) {
                System.out.println("Encoded " + data[i][0]);
                float[] s1 = bert.embedSequence(data[i][3]);
                float[] s2 = bert.embedSequence(data[i][1]);
                float[] a = new float[]{data[i][2].equals("against") ? -1 : 1};
                r.putRow(i, Nd4j.concat(0, Nd4j.create(s1), Nd4j.create(s2), Nd4j.create(a)));
            }
            return r;
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
            ex.printStackTrace();
            throw ex;
        }
    }

    //accepts rows in s2,a,s1 format
    //returns rows in s2 format
    public static INDArray encodeDatasetForValueLabels(String[][] data) {
        try ( Bert bert = Bert.load("com/robrua/nlp/easy-bert/bert-uncased-L-12-H-768-A-12")) {
            INDArray r = Nd4j.create(data.length, 768);
            for (int i = 0; i < data.length; i++) {
                System.out.println("Encoded " + data[i][0]);
                float[] s2 = bert.embedSequence(data[i][1]);
                r.putRow(i, Nd4j.concat(0, Nd4j.create(s2)));
            }
            return r;
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
            ex.printStackTrace();
            throw ex;
        }
    }

    //accepts rows in s2,a,s1 format
    //returns rows in s1,a format
    public static INDArray encodeDatasetForValues(String[][] data) {
        try ( Bert bert = Bert.load("com/robrua/nlp/easy-bert/bert-uncased-L-12-H-768-A-12")) {
            INDArray r = Nd4j.create(data.length, 768 * 2 + 1);
            for (int i = 0; i < data.length; i++) {
                System.out.println("Encoded " + data[i][0]);
                float[] s1 = bert.embedSequence(data[i][3]);
                float[] a = new float[]{data[i][2].equals("against") ? -1 : 1};
                r.putRow(i, Nd4j.concat(0, Nd4j.create(s1), Nd4j.create(a)));
            }
            return r;
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
            ex.printStackTrace();
            throw ex;
        }
    }

    public static void trainDiscriminator(int numEpochs, String[] trainCSV, float[][] labels, String[] testCSV, float[][] testLabels) throws IOException {
        SameDiff nn;
        nn = initNetwork(false, 0, path);
        DataSet trainData = new DataSet(
                encodeDataset(Arrays.stream(trainCSV).map(row -> row.split("\\t")).toList().toArray(new String[0][])),
                Nd4j.create(labels));
        DataSet testData = new DataSet(
                encodeDataset(Arrays.stream(testCSV).map(row -> row.split("\\t")).toList().toArray(new String[0][])),
                Nd4j.create(testLabels));
        double learningRate = 1e-3;
        TrainingConfig config = new TrainingConfig.Builder()
                .updater(new Adam(learningRate)) //Adam optimizer with specified learning rate
                .dataSetFeatureMapping("input") //DataSet features array should be associated with variable "input"
                .dataSetLabelMapping("label") //DataSet label array should be associated with variable "label"
                .build();

        nn.setTrainingConfig(config);
        Evaluation evaluation = new Evaluation();
        nn.evaluate(testData.iterateWithMiniBatches(), "output", evaluation);
        double score = evaluation.f1();
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            nn.fit(trainData);
            nn.evaluate(testData.iterateWithMiniBatches(), "output", evaluation);
            if (evaluation.f1() > score) {
                score = evaluation.f1();
                //Print evaluation statistics:
                System.out.println(evaluation.stats());
                nn.save(new File(path), true);
            }
        }
    }

    public static void trainValues(int numEpochs, String[] trainCSV, float[][] trainLabels) throws IOException {
        SameDiff nn;
        for (int i = 0; i < 20; i++) {
            final int fi = i;
            nn = initNetwork(true, 0, path);
            DataSet trainData = new DataSet(
                    encodeDatasetForValues(IntStream.range(0, trainCSV.length)
                            .filter(j -> trainLabels[j][fi] > 0)
                            .mapToObj(j -> trainCSV[j])
                            .map(row -> row.split("\\t")).toList().toArray(new String[0][])),
                    encodeDatasetForValueLabels(IntStream.range(0, trainCSV.length)
                            .filter(j -> trainLabels[j][fi] > 0)
                            .mapToObj(j -> trainCSV[j])
                            .map(row -> row.split("\\t")).toList().toArray(new String[0][])));
            double learningRate = 1e-3;
            TrainingConfig config = new TrainingConfig.Builder()
                    .updater(new Adam(learningRate)) //Adam optimizer with specified learning rate
                    .dataSetFeatureMapping("input") //DataSet features array should be associated with variable "input"
                    .dataSetLabelMapping("label") //DataSet label array should be associated with variable "label"
                    .build();

            nn.setTrainingConfig(config);
            for (int epoch = 0; epoch < numEpochs; epoch++) {
                nn.fit(trainData);
            }
            nn.save(new File(path), true);
        }
    }

    public static SameDiff initNetwork(boolean trainValue, int value, String path) throws IOException {

        SameDiff sd = SameDiff.create();
        if (!trainValue || value != 0) {
            File saveFileForInference = new File(path);
            sd = SameDiff.fromFlatFile(saveFileForInference);
        }

        SDVariable in;
        SDVariable label;
        if (trainValue) {
            in = sd.placeHolder("input", DataType.FLOAT, -1, 768 + 1);
            label = sd.placeHolder("label", DataType.FLOAT, -1, 768);
        } else {
            int nIn = 768 + 768 + 1;
            int nOut = 20;
            in = sd.placeHolder("input", DataType.FLOAT, -1, nIn);
            label = sd.placeHolder("label", DataType.FLOAT, -1, nOut);
        }
        SDVariable[] discrimators = new SDVariable[20];
        SDVariable[][] valueFunctions = new SDVariable[20][2];

        for (int i = 0; i < 20; i++) {
            SDVariable s1 = sd.slice(in, new int[]{0, 768}, 1, 768);
            SDVariable s2 = sd.slice(in, new int[]{0, 0}, 1, 768);
            valueFunctions[i] = valueFunction("value_" + i, s1,
                    (trainValue && i == value) ? label : sd.constant(Nd4j.zeros(1, 768)));
            discrimators[i] = discriminator("d_" + i, sd.concat(
                    0,
                    in,
                    valueFunctions[i][0],
                    sd.math.cosineSimilarity(s2, valueFunctions[i][0], 1)));
            if (trainValue && i == value) {
                sd.setLossVariables(valueFunctions[i][1]);
            }
        }
        SDVariable predictions = sd.concat("output", 0, discrimators);
        if (!trainValue) {
            SDVariable loss = sd.loss.sigmoidCrossEntropy("loss", predictions, label, sd.constant(1));
            sd.setLossVariables(loss);
        }
        return sd;
    }

    public static SDVariable discriminator(String prefix, SDVariable input) {
        //s2, a, s1, v, s2*v
        SameDiff sd = input.getSameDiff();
        int inputSize = 768 + 768 + 1 + 768 + 1;
        if (input.getShape().length < 2 || input.getShape()[1] != inputSize) {
            throw new IllegalArgumentException("bad input size for discriminator");
        }
        SDVariable w0 = sd.var(new XavierInitScheme('c', 768 + 1, 768), DataType.FLOAT, 768 + 1, 768);
        SDVariable b0 = sd.zero(prefix + "_b0", 1, 768);
        SDVariable w1 = sd.var(new XavierInitScheme('c', 768, 768), DataType.FLOAT, 768, 768);
        SDVariable b1 = sd.zero(prefix + "_b1", 1, 768);
        SDVariable w2 = sd.var(new XavierInitScheme('c', 768, 768), DataType.FLOAT, 768, 768);
        SDVariable b2 = sd.zero(prefix + "_b2", 1, 768);
        SDVariable output = sd.nn.sigmoid(prefix + "_output", sd.nn.relu(sd.nn.relu(input.mmul(w0).add(b0), 0).mmul(w1).add(b1), 0).mmul(w2).add(b2));
        return output;
    }
    //

    public static SDVariable[] valueFunction(String name, SDVariable input, SDVariable label) {
        int inputSize = 768 + 1;
        if (input.getShape().length < 2 || input.getShape()[1] != inputSize) {
            throw new IllegalArgumentException("bad input size for discriminator");
        }
        SameDiff sd = input.getSameDiff();
        SDVariable w0 = sd.var(new XavierInitScheme('c', inputSize, 768), DataType.FLOAT, inputSize, 768);
        SDVariable b0 = sd.zero(UUID.randomUUID().toString(), 1, 768);
        SDVariable w1 = sd.var(new XavierInitScheme('c', 768, 768), DataType.FLOAT, 768, 768);
        SDVariable b1 = sd.zero(UUID.randomUUID().toString(), 1, 768);
        SDVariable w2 = sd.var(new XavierInitScheme('c', 768, 768), DataType.FLOAT, 768, 768);
        SDVariable b2 = sd.zero(UUID.randomUUID().toString(), 1, 768);
        SDVariable output = sd.nn.relu(name, sd.nn.relu(input.mmul(w0).add(b0), 0).mmul(w1).add(b1), 0).mmul(w2).add("output", b2);
        SDVariable diff = sd.math.squaredDifference(output, label);
        SDVariable lossMse = diff.mean();
        return new SDVariable[]{output, lossMse};
    }
}
