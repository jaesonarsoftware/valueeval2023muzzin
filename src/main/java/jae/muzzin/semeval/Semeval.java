package jae.muzzin.semeval;

import com.robrua.nlp.bert.Bert;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.UUID;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
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

    public static void trainDiscriminator(int numEpochs) throws IOException {
        SameDiff nn;
        nn = initNetwork(false, 0, path);
        DataSetIterator trainData = null;
        DataSetIterator testData = null;
        double learningRate = 1e-3;
        TrainingConfig config = new TrainingConfig.Builder()
                .updater(new Adam(learningRate)) //Adam optimizer with specified learning rate
                .dataSetFeatureMapping("input") //DataSet features array should be associated with variable "input"
                .dataSetLabelMapping("label") //DataSet label array should be associated with variable "label"
                .build();

        nn.setTrainingConfig(config);
        Evaluation evaluation = new Evaluation();
        nn.evaluate(testData, "output", evaluation);
        double score = evaluation.f1();
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            nn.fit(trainData, 1);
            nn.evaluate(testData, "output", evaluation);
            if (evaluation.f1() > score) {
                score = evaluation.f1();
                //Print evaluation statistics:
                System.out.println(evaluation.stats());
                nn.save(new File(path), true);
            }
        }
    }

    public static void trainValues(int numEpochs) throws IOException {
        SameDiff nn;
        for (int i = 0; i < 20; i++) {
            nn = initNetwork(true, 0, path);
            DataSetIterator trainData = null;
            DataSetIterator testData = null;
            double learningRate = 1e-3;
            TrainingConfig config = new TrainingConfig.Builder()
                    .updater(new Adam(learningRate)) //Adam optimizer with specified learning rate
                    .dataSetFeatureMapping("input") //DataSet features array should be associated with variable "input"
                    .dataSetLabelMapping("label") //DataSet label array should be associated with variable "label"
                    .build();

            nn.setTrainingConfig(config);
            Evaluation evaluation = new Evaluation();
            nn.evaluate(testData, "output", evaluation);
            double score = evaluation.f1();
            for (int epoch = 0; epoch < numEpochs; epoch++) {
                nn.fit(trainData, 1);
                nn.evaluate(testData, "value_" + i, evaluation);
                if (evaluation.f1() > score) {
                    score = evaluation.f1();
                    //Print evaluation statistics:
                    System.out.println(evaluation.stats());
                    nn.save(new File(path), true);
                }
            }
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
