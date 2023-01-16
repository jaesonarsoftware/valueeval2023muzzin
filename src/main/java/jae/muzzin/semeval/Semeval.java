package jae.muzzin.semeval;

import au.com.bytecode.opencsv.CSVReader;
import com.robrua.nlp.bert.Bert;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.UUID;
import java.util.stream.IntStream;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.weightinit.impl.XavierInitScheme;
import de.siegmar.fastcsv.reader.NamedCsvReader;
import de.siegmar.fastcsv.reader.CsvReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.function.IntSupplier;

/**
 *
 * @author Admin
 */
public class Semeval {

    public static final int num_training_examples = 985;
    public static String path = "model.fb";

    public static void main(String[] args) throws IOException {
        //encodeDataset("arguments-training-short.tsv", "arguments-training.nd4j");
        CsvReader reader = CsvReader.builder()
                .fieldSeparator('\t')
                .build(new FileReader("labels-training.tsv"));
        List<float[]> d = new ArrayList<>();
        double[][] labels = reader.stream()
                .skip(1)
                .map(row
                        -> IntStream.range(1, 21)
                        .mapToDouble(i
                                -> Double.parseDouble(row.getField(i))).toArray()
                ).toList().toArray(new double[0][]);
        trainValues(100, Nd4j.read(new FileInputStream("arguments-training.nd4j")), labels);
    }

    public static void encodeDataset(String path, String writePath) throws IOException {

        try ( Bert bert = Bert.load("com/robrua/nlp/easy-bert/bert-uncased-L-12-H-768-A-12")) {
            INDArray r = Nd4j.create(num_training_examples, 768 * 2 + 1);
            IntSupplier jae = new IntSupplier() {
                private int r = 0;

                @Override
                public int getAsInt() {
                    return r++;
                }
            };
            NamedCsvReader.builder()
                    .fieldSeparator('\t')
                    .build(new FileReader(path)).forEach(row -> {
                System.out.println("Encoded " + row.getField("Argument ID"));
                float[] s1 = bert.embedSequence(row.getField("Premise"));
                float[] s2 = bert.embedSequence(row.getField("Conclusion"));
                float[] a = new float[]{row.getField("Stance").equals("against") ? -1 : 1};
                r.putRow(jae.getAsInt(), Nd4j.concat(0, Nd4j.create(s1), Nd4j.create(s2), Nd4j.create(a)));
            });
            Nd4j.write(new FileOutputStream(writePath, true), r);
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
            ex.printStackTrace();
            throw ex;
        }
    }

    public static void trainDiscriminator(int numEpochs, INDArray trainCSV, float[][] labels, INDArray testCSV, float[][] testLabels) throws IOException {
        SameDiff nn;
        nn = initNetwork(false, 0, path);
        DataSet trainData = new DataSet(
                trainCSV,
                Nd4j.create(labels));
        DataSet testData = new DataSet(
                testCSV,
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

    public static void trainValues(int numEpochs, INDArray training, double[][] trainLabels) throws IOException {
        SameDiff nn;
        for (int i = 0; i < 20; i++) {
            final int fi = i;
            nn = initNetwork(true, 0, path);
            int[] argIdsThisValueAffects = IntStream.range(0, trainLabels.length)
                    .filter(j -> trainLabels[j][fi] > 0)
                    .toArray();
            DataSet trainData = new DataSet(
                    training.tensorAlongDimension(0, 0, 2).tensorAlongDimension(1, argIdsThisValueAffects),
                    training.tensorAlongDimension(0, 1).tensorAlongDimension(1, argIdsThisValueAffects)
            );
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
            SDVariable s1plusa = sd.concat(1, sd.slice(in, new int[]{0, 0}, 1, 768), sd.slice(in, new int[]{0, 768 * 2}, 1, 1));
            SDVariable s2 = sd.slice(in, new int[]{0, 768}, 1, 768);
            valueFunctions[i] = valueFunction("value_" + i, s1plusa,
                    (trainValue && i == value) ? s2 : sd.constant(Nd4j.zeros(1, 768)));
            discrimators[i] = discriminator("d_" + i, sd.concat(
                    1,
                    in,
                    valueFunctions[i][0],
                    sd.math.cosineSimilarity(s2, valueFunctions[i][0], 1)));
            if (trainValue && i == value) {
                sd.setLossVariables(valueFunctions[i][1]);
            }
        }
        SDVariable predictions = sd.concat("output", 1, discrimators);
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
        SDVariable w0 = sd.var(new XavierInitScheme('c', inputSize, 768), DataType.FLOAT, inputSize, 768);
        SDVariable b0 = sd.zero(prefix + "_b0", 1, 768);
        SDVariable w1 = sd.var(new XavierInitScheme('c', 768, 512), DataType.FLOAT, 768, 512);
        SDVariable b1 = sd.zero(prefix + "_b1", 1, 512);
        SDVariable w2 = sd.var(new XavierInitScheme('c', 512, 256), DataType.FLOAT, 768, 256);
        SDVariable b2 = sd.zero(prefix + "_b2", 1, 256);
        SDVariable w3 = sd.var(new XavierInitScheme('c', 256, 1), DataType.FLOAT, 256, 1);
        SDVariable b3 = sd.zero(prefix + "_b3", 1, 1);
        SDVariable output = sd.nn.sigmoid(prefix + "_output",
                sd.nn.relu(
                        sd.nn.relu(
                                sd.nn.relu(input.mmul(w0).add(b0), 0)
                                        .mmul(w1).add(b1), 0)
                                .mmul(w2).add(b2), 0).mmul(w3).add(b3));
        return output;
    }
    //

    public static SDVariable[] valueFunction(String name, SDVariable input, SDVariable label) {
        int inputSize = 768 + 1;
        SameDiff sd = input.getSameDiff();
        SDVariable w0 = sd.var(new XavierInitScheme('c', inputSize, 768), DataType.FLOAT, inputSize, 768);
        SDVariable b0 = sd.zero(UUID.randomUUID().toString(), 1, 768);
        SDVariable w1 = sd.var(new XavierInitScheme('c', 768, 768), DataType.FLOAT, 768, 768);
        SDVariable b1 = sd.zero(UUID.randomUUID().toString(), 1, 768);
        SDVariable w2 = sd.var(new XavierInitScheme('c', 768, 768), DataType.FLOAT, 768, 768);
        SDVariable b2 = sd.zero(UUID.randomUUID().toString(), 1, 768);
        SDVariable output = sd.nn.relu(name, sd.nn.relu(input.mmul(w0).add(b0), 0).mmul(w1).add(b1), 0).mmul(w2).add(name+"_b2", b2);
        SDVariable diff = sd.math.squaredDifference(output, label);
        SDVariable lossMse = diff.mean();
        return new SDVariable[]{output, lossMse};
    }
}
