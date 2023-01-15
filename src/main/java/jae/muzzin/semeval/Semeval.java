package jae.muzzin.semeval;

import com.robrua.nlp.bert.Bert;
import java.util.Arrays;
import java.util.UUID;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
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

    public static void main(String[] args) {
        try ( Bert bert = Bert.load("com/robrua/nlp/easy-bert/bert-uncased-L-12-H-768-A-12")) {
            float[] embedding = bert.embedSequence("The EU should extend the model of Pooling & Sharing European military resources beyond air transport.");
            System.out.println(Arrays.toString(embedding));
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
            ex.printStackTrace();
        }
    }

    public static SameDiff initNetwork(boolean trainValue, int value) {
        SameDiff sd = SameDiff.create();

        SDVariable in;

        SDVariable label;
        if (trainValue) {
            in = sd.placeHolder("input", DataType.FLOAT, -1, 768 + 1);
            label = sd.placeHolder("label", DataType.FLOAT, -1, 768);
        } else {
            int nIn = 768 + 768 + 1;
            int nOut = 20;
            in = sd.placeHolder("input", DataType.FLOAT, -1, nIn);                 //Shape: [?, 784] - i.e., minibatch x 784 for MNIST
            label = sd.placeHolder("label", DataType.FLOAT, -1, nOut);             //Shape: [?, 10] - i.e., minibatch x 10 for MNIST
        }
        SDVariable[] discrimators = new SDVariable[20];
        SDVariable[][] valueFunctions = new SDVariable[20][2];
        //Define hidden layer - MLP (fully connected)
        for (int i = 0; i < 20; i++) {
            SDVariable s1 = sd.slice(in, new int[]{0, 768}, 1, 768);
            SDVariable s2 = sd.slice(in, new int[]{0, 0}, 1, 768);
            valueFunctions[i] = valueFunction(s1,
                    (trainValue && i == value) ? label : sd.constant(Nd4j.zeros(1, 768)));
            discrimators[i] = discriminator("d_" + i, sd.concat(0,
                    in,
                    valueFunctions[i][0],
                    sd.math.cosineSimilarity(s2, valueFunctions[i][0], 1)));
            if (trainValue && i==value) {
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

    public static SDVariable[] valueFunction(SDVariable input, SDVariable label) {
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
        SDVariable output = sd.nn.relu(sd.nn.relu(input.mmul(w0).add(b0), 0).mmul(w1).add(b1), 0).mmul(w2).add("output", b2);
        SDVariable diff = sd.math.squaredDifference(output, label);
        SDVariable lossMse = diff.mean();
        return new SDVariable[]{output, lossMse};
    }
}
