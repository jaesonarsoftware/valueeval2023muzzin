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
        train(6000, Nd4j.read(new FileInputStream("arguments-training.nd4j")), labels);
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

    public static void train(int numEpochs, INDArray training, double[][] trainLabels) throws IOException {
        SameDiff nn;
        if (new File(path).exists()) {
            nn = SameDiff.load(new File(path), true);
        } else {
            nn = initNetwork(true, path);
        }
        double learningRate = .001;
        TrainingConfig config = new TrainingConfig.Builder()
                .updater(new Nadam(learningRate)) //Adam optimizer with specified learning rate
                .dataSetFeatureMapping("input") //DataSet features array should be associated with variable "input"
                .dataSetLabelMapping("label") //DataSet label array should be associated with variable "label"
                .build();
        nn.setTrainingConfig(config);
        System.out.println("Starting ");
        long[] ids = LongStream.range(0, trainLabels.length)
                .toArray();
        long[] trainingIds = Arrays.copyOfRange(ids, 0, (int) (ids.length * .8));
        long[] testIds = Arrays.copyOfRange(ids, (int) (ids.length * .8), ids.length);
        System.out.println("Found " + trainingIds.length + " matching rows");
        System.out.println("Training " + training.shape()[0]);
        System.out.println("Labels " + trainLabels.length);
        INDArray labelsArr = Nd4j.create(trainLabels);
        DataSet trainData = new DataSet(
                training.get(NDArrayIndex.indices(trainingIds), NDArrayIndex.all()),
                labelsArr.get(NDArrayIndex.indices(trainingIds), NDArrayIndex.all())
        );
        DataSet testData = new DataSet(
                training.get(NDArrayIndex.indices(testIds), NDArrayIndex.all()),
                labelsArr.get(NDArrayIndex.indices(testIds), NDArrayIndex.all())
        );
        //nn.clearOpInputs();
        //nn.clearPlaceholders(true);
        System.out.println("fit");
        double bestScore = Double.MAX_VALUE;
        for (int e = 0; e < numEpochs; e++) {
            try {
                var h = nn.fit(new ViewIterator(trainData, Math.min(100, trainingIds.length - 1)), 1, new ScoreListener(1, true, true));
                if (e % 200 == 0 && e > 0) {
                    Evaluation evaluation = new Evaluation();
                    nn.evaluate(new ViewIterator(trainData, Math.min(300, trainingIds.length - 1)), "output", evaluation);
                    System.out.println("Train score:" + evaluation.f1());
                    System.out.println(evaluation.stats());
                    evaluation.reset();
                    nn.evaluate(new ViewIterator(testData, Math.min(300, testIds.length - 1)), "output", evaluation);
                    //Print evaluation statistics:
                    System.out.println("Train score:" + evaluation.f1());
                    System.out.println(evaluation.stats());
                    if (evaluation.f1() < bestScore) {
                        System.out.println("Best, saved it.");
                        nn.save(new File(path), true);
                        bestScore = evaluation.f1();
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
        SDVariable a = in.get(SDIndex.all(), SDIndex.interval(768 * 2, 768 * 2 + 1));
        for (int i = 0; i < 20; i++) {
            SDVariable w0 = sd.var("w0", new XavierInitScheme('c', 1, 768), 1, 768);
            SDVariable b0 = sd.zero("b0", 1, 768);
            SDVariable aW0plusb0 = sd.nn.tanh(a.mmul(w0).add(b0));
            SDVariable s1 = in.get(
                    SDIndex.all(),
                    SDIndex.interval(0, 768));
            SDVariable s2 = in.get(SDIndex.all(),
                    SDIndex.interval(768, 768 * 2));

            SDVariable w1 = sd.var("w1", new XavierInitScheme('c', 768, 1), 768, 1);
            SDVariable b1 = sd.zero("b0", 1, 1);
            SDVariable reduced = sd.nn.sigmoid(s1.mmul(aW0plusb0).mmul(sd.transpose(s2)).mmul(w1).add(b1));
            discrimators[i] = reduced;
        }
        SDVariable predictions = sd.concat("output", 1, discrimators);
        if (!trainValue) {
            SDVariable loss = sd.loss.sigmoidCrossEntropy("loss", predictions, label, null);
            sd.setLossVariables(loss);
        }
        return sd;
    }

}
