import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class ResultadoModelo {
    public final String modelName;
    public final int[] featureIndexes;
    public final double threshold;
    public final int tp;
    public final int tn;
    public final int fp;
    public final int fn;
    public final double accuracy;
    public final double precision;
    public final double recall;
    public final double specificity;
    public final double f1;
    public final double auc;
    public final int[] yTrue;
    public final double[] probabilities;
    public double featureAcquisitionCost;
    public int delayedTests;

    public ResultadoModelo(String modelName, int[] featureIndexes, double threshold,
                       int tp, int tn, int fp, int fn,
                       double accuracy, double precision, double recall,
                       double specificity, double f1, double auc,
                       int[] yTrue, double[] probabilities) {
        this.modelName = modelName;
        this.featureIndexes = featureIndexes;
        this.threshold = threshold;
        this.tp = tp;
        this.tn = tn;
        this.fp = fp;
        this.fn = fn;
        this.accuracy = accuracy;
        this.precision = precision;
        this.recall = recall;
        this.specificity = specificity;
        this.f1 = f1;
        this.auc = auc;
        this.yTrue = yTrue.clone();
        this.probabilities = probabilities.clone();
    }

    public static ResultadoModelo fromPredictions(String name, int[] featureIndexes, int[] yTrue, double[] probs) {
        return fromValidationAndTest(name, featureIndexes, yTrue, probs, yTrue, probs);
    }

    public static ResultadoModelo fromValidationAndTest(String name, int[] featureIndexes,
                                                    int[] yVal, double[] probsVal,
                                                    int[] yTest, double[] probsTest) {
        double bestT = 0.5;
        double bestF1 = -1.0;
        for (int i = 10; i <= 90; i++) {
            double t = i / 100.0;
            int[] c = confusion(yVal, probsVal, t);
            double p = c[0] + c[2] == 0 ? 0.0 : (double) c[0] / (c[0] + c[2]);
            double r = c[0] + c[3] == 0 ? 0.0 : (double) c[0] / (c[0] + c[3]);
            double f1 = (p + r) == 0 ? 0.0 : (2 * p * r / (p + r));
            if (f1 > bestF1) {
                bestF1 = f1;
                bestT = t;
            }
        }

        int[] c = confusion(yTest, probsTest, bestT);
        int tp = c[0], tn = c[1], fp = c[2], fn = c[3];
        int n = yTest.length;
        double acc = (double) (tp + tn) / n;
        double precision = tp + fp == 0 ? 0.0 : (double) tp / (tp + fp);
        double recall = tp + fn == 0 ? 0.0 : (double) tp / (tp + fn);
        double specificity = tn + fp == 0 ? 0.0 : (double) tn / (tn + fp);
        double f1 = (precision + recall) == 0 ? 0.0 : (2 * precision * recall / (precision + recall));
        double auc = auc(yTest, probsTest);

        return new ResultadoModelo(name, featureIndexes, bestT, tp, tn, fp, fn, acc, precision, recall, specificity, f1, auc,
                yTest, probsTest);
    }

    private static int[] confusion(int[] yTrue, double[] probs, double threshold) {
        int tp = 0, tn = 0, fp = 0, fn = 0;
        for (int i = 0; i < yTrue.length; i++) {
            int pred = probs[i] >= threshold ? 1 : 0;
            if (pred == 1 && yTrue[i] == 1) tp++;
            else if (pred == 0 && yTrue[i] == 0) tn++;
            else if (pred == 1) fp++;
            else fn++;
        }
        return new int[]{tp, tn, fp, fn};
    }

    public static double auc(int[] yTrue, double[] probs) {
        int pos = 0;
        for (int y : yTrue) {
            if (y == 1) pos++;
        }
        int neg = yTrue.length - pos;
        if (pos == 0 || neg == 0) {
            return 0.5;
        }

        List<int[]> pairs = new ArrayList<>();
        for (int i = 0; i < yTrue.length; i++) {
            int scaled = (int) Math.round(probs[i] * 1_000_000);
            pairs.add(new int[]{scaled, yTrue[i]});
        }
        pairs.sort(Comparator.comparingInt((int[] p) -> p[0]).reversed());

        double tp = 0.0;
        double fp = 0.0;
        double prevTpr = 0.0;
        double prevFpr = 0.0;
        double area = 0.0;

        for (int[] pair : pairs) {
            if (pair[1] == 1) tp++;
            else fp++;

            double tpr = tp / pos;
            double fpr = fp / neg;
            area += (fpr - prevFpr) * (tpr + prevTpr) / 2.0;
            prevTpr = tpr;
            prevFpr = fpr;
        }
        return area;
    }
}


