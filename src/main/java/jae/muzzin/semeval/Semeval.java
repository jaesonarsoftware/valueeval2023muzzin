

package jae.muzzin.semeval;

import com.robrua.nlp.bert.Bert;
import java.util.Arrays;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.weightinit.impl.XavierInitScheme;
/**
 *
 * @author Admin
 */
public class Semeval {

    public static void main(String[] args) {
        try(Bert bert = Bert.load("com/robrua/nlp/easy-bert/bert-uncased-L-12-H-768-A-12")) {
            float[] embedding = bert.embedSequence("The EU should extend the model of Pooling & Sharing European military resources beyond air transport.");
            System.out.println(Arrays.toString(embedding));
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
            ex.printStackTrace();
        }
    }
    
    public static SameDiff valueFunctionMLP() {
        SameDiff sd = SameDiff.create();
        
        SDVariable in = sd.placeHolder("input", DataType.FLOAT, -1, 768+1);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, 768);
        SDVariable w0 = sd.var("w0", new XavierInitScheme('c', 768+1, 768), DataType.FLOAT, 768+1, 768);
        SDVariable b0 = sd.zero("b0", 1, 768);
        SDVariable w1 = sd.var("w1", new XavierInitScheme('c', 768, 768), DataType.FLOAT, 768, 768);
        SDVariable b1 = sd.zero("b1", 1, 768);
        SDVariable w2 = sd.var("w2", new XavierInitScheme('c', 768, 768), DataType.FLOAT, 768, 768);
        SDVariable b2 = sd.zero("b2", 1, 768);
        SDVariable output = sd.nn.relu(sd.nn.relu(in.mmul(w0).add(b0), 0).mmul(w1).add(b1), 0).mmul(w2).add("output", b2);
        SDVariable diff = sd.math.squaredDifference(output, label);
        SDVariable lossMse = diff.mean();
        sd.setLossVariables(lossMse);
        return sd;
    }
}
