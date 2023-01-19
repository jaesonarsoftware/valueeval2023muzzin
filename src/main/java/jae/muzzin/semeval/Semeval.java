package jae.muzzin.semeval;

import com.robrua.nlp.bert.Bert;
import de.siegmar.fastcsv.reader.CsvReader;
import java.io.File;
import java.io.IOException;
import java.util.UUID;
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
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.IntSupplier;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import org.nd4j.autodiff.listeners.impl.ScoreListener;
import org.nd4j.autodiff.samediff.SDIndex;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Nadam;

public class Semeval {

    public static final int num_training_examples = 5393;
    public static String path = "model.fb";

    public static void main(String[] args) throws IOException {
        //encodeDataset("arguments-training.tsv", "arguments-training.nd4j");
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
        System.out.println("starting training");
        trainValues(3000, Nd4j.read(new FileInputStream("arguments-training.nd4j")), labels);
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
        nn = initNetwork(false, path);
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
        if (new File(path).exists()) {
            nn = SameDiff.load(new File(path), true);
        } else {
            nn = initNetwork(true, path);
        }
        double learningRate = .00005;
        TrainingConfig config = new TrainingConfig.Builder()
                .updater(new Nadam(learningRate)) //Adam optimizer with specified learning rate
                .dataSetFeatureMapping("input") //DataSet features array should be associated with variable "input"
                .dataSetLabelMapping("label") //DataSet label array should be associated with variable "label"
                .build();
        nn.setTrainingConfig(config);
        for (long i = 0; i < 20; i++) {
            System.out.println("Starting " + i);
            final long fi = i;
            long[] argIdsThisValueAffectsAll = LongStream.range(0, trainLabels.length)
                    .filter(j -> trainLabels[(int) j][(int) fi] > 0)
                    .toArray();
            long[] trainingIds = Arrays.copyOfRange(argIdsThisValueAffectsAll, 0, (int) (argIdsThisValueAffectsAll.length * .8));
            long[] testIds = Arrays.copyOfRange(argIdsThisValueAffectsAll, (int) (argIdsThisValueAffectsAll.length * .8), argIdsThisValueAffectsAll.length);
            System.out.println("Found " + trainingIds.length + " matching rows");
            System.out.println("Training " + training.shape()[0]);
            System.out.println("Labels " + trainLabels.length);
            DataSet trainData = new DataSet(
                    training.get(NDArrayIndex.indices(trainingIds), NDArrayIndex.all()),
                    training.get(NDArrayIndex.indices(trainingIds), NDArrayIndex.interval(768 * 2, 768 * 2 + 1))
            );
            DataSet testData = new DataSet(
                    training.get(NDArrayIndex.indices(testIds), NDArrayIndex.all()),
                    training.get(NDArrayIndex.indices(testIds), NDArrayIndex.interval(768 * 2, 768 * 2 + 1))
            );
            //nn.clearOpInputs();
            //nn.clearPlaceholders(true);
            nn.setLossVariables("value_" + i + "_loss");
            System.out.println("fit");
            double bestScore = Double.MAX_VALUE;
            for (int e = 0; e < numEpochs; e++) {
                try {
                    var h = nn.fit(new ViewIterator(trainData, Math.min(100, trainingIds.length - 1)), 1, new ScoreListener(1, true, true));
                    if (e % 500 == 0 && e > 0) {
                        RegressionEvaluation evaluation = new RegressionEvaluation();
                        nn.evaluate(new ViewIterator(trainData, Math.min(300, trainingIds.length - 1)), "value_" + fi, evaluation);
                        System.out.println("Train score:" + evaluation.averageMeanAbsoluteError());
                        evaluation.reset();
                        nn.evaluate(new ViewIterator(testData, Math.min(300, testIds.length - 1)), "value_" + fi, evaluation);
                        //Print evaluation statistics:
                        System.out.println("Test score:" + evaluation.averageMeanAbsoluteError());
                        if (evaluation.averageMeanAbsoluteError() < bestScore && e > 2000) {
                            System.out.println("Best, saved it.");
                            nn.save(new File(path), true);
                            bestScore = evaluation.averageMeanAbsoluteError();
                        }
                    }
                    //nn.getVariable("input").setArray(trainData.getFeatures());
                    //System.out.println(Arrays.toString(nn.getVariable("value_0").eval().shape()));
                    //System.out.println(nn.getVariable("value_0").eval().toStringFull());
                } catch (Exception ex) {
                    System.out.println(ex.getMessage());
                }
            }
        }
    }

    public static SameDiff initNetwork(boolean trainValue, String path) throws IOException {
        SameDiff sd = SameDiff.create();
        if (!trainValue) {
            File saveFileForInference = new File(path);
            sd = SameDiff.fromFlatFile(saveFileForInference);
            return sd;
        }

        SDVariable in;
        SDVariable label;
        int nIn = 768 + 768 + 1;
        in = sd.placeHolder("input", DataType.DOUBLE, -1, nIn);
        if (trainValue) {
            label = sd.placeHolder("label", DataType.DOUBLE, -1, 768);
        } else {
            int nOut = 20;
            label = sd.placeHolder("label", DataType.DOUBLE, -1, nOut);
        }
        SDVariable[] discrimators = new SDVariable[20];

        for (int i = 0; i < 20; i++) {
            SDVariable s1s2 = in.get(SDIndex.all(),
                    SDIndex.interval(0, 768 * 2));
            SDVariable a = in.get(SDIndex.all(), SDIndex.interval(768 * 2, 768 * 2 + 1));
            var valueFunction = valueFunction("value_" + i, s1s2, label);
            discrimators[i] = discriminator("d_" + i, sd.concat(
                    1,
                    in,
                    valueFunction,
                    sd.math.abs(a.sub(valueFunction))));
        }
        SDVariable predictions = sd.concat("output", 1, discrimators);
        if (!trainValue) {
            SDVariable loss = sd.loss.sigmoidCrossEntropy("loss", predictions, label, null);
            sd.setLossVariables(loss);
        }
        return sd;
    }

    public static SDVariable discriminator(String prefix, SDVariable input) {
        //s2, a, s1, v, |a-v|
        SameDiff sd = input.getSameDiff();
        int inputSize = 768 + 768 + 1 + 1 + 1;
        SDVariable w1 = sd.var(new XavierInitScheme('c', inputSize, 256), DataType.DOUBLE, inputSize, 256);
        SDVariable b1 = sd.zero(prefix + "_b1", 1, 256);
        SDVariable w2 = sd.var(new XavierInitScheme('c', 256, 1), DataType.DOUBLE, 256, 1);
        SDVariable b2 = sd.zero(prefix + "_b2", 1, 1);
        SDVariable output = sd.nn.sigmoid(prefix + "_output",
                sd.nn.relu(
                        input.mmul(w1).add(b1), 0)
                        .mmul(w2).add(b2));
        return output;
    }

    public static SDVariable valueFunction(String name, SDVariable input, SDVariable label) {
        int inputSize = 768 * 2;
        SameDiff sd = input.getSameDiff();
        SDVariable w1 = sd.var(new XavierInitScheme('c', inputSize, 128), DataType.DOUBLE, inputSize, 128);
        SDVariable b1 = sd.zero(UUID.randomUUID().toString(), 1, 128);
        SDVariable w2 = sd.var(new XavierInitScheme('c', 128, 64), DataType.DOUBLE, 128, 64);
        SDVariable b2 = sd.zero(UUID.randomUUID().toString(), 1, 64);
        SDVariable w5 = sd.var(new XavierInitScheme('c', 64, 1), DataType.DOUBLE, 64, 1);
        SDVariable b5 = sd.zero(UUID.randomUUID().toString(), 1, 1);
        SDVariable output = sd.nn.tanh(name,
                sd.nn.relu(sd.nn.relu(input.mmul(w1).add(b1), 0).mmul(w2).add(b2), 0).mmul(w5).add(b5)
        );
        //SDVariable output = sd.nn.sigmoid(name, sd.nn.relu(input.mmul(w0), 0));
        SDVariable diff = sd.loss.absoluteDifference(name + "_loss", label, output, null);
        return output;
    }
}
