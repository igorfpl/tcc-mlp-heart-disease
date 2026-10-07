import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

class Divisao {
    public final List<PontoDado> train;
    public final List<PontoDado> val;
    public final List<PontoDado> test;

    public Divisao(List<PontoDado> train, List<PontoDado> val, List<PontoDado> test) {
        this.train = train;
        this.val = val;
        this.test = test;
    }
}

class DadosPreparados {
    public final double[][] xTrain;
    public final int[] yTrain;
    public final double[][] xVal;
    public final int[] yVal;
    public final double[][] xTest;
    public final int[] yTest;
    public final String[] featureNames;
    public final double[] medians;
    public final double[] means;
    public final double[] stds;
    public final boolean standardized;

    public DadosPreparados(double[][] xTrain, int[] yTrain, double[][] xVal, int[] yVal,
                        double[][] xTest, int[] yTest, String[] featureNames,
                        double[] medians, double[] means, double[] stds, boolean standardized) {
        this.xTrain = xTrain;
        this.yTrain = yTrain;
        this.xVal = xVal;
        this.yVal = yVal;
        this.xTest = xTest;
        this.yTest = yTest;
        this.featureNames = featureNames;
        this.medians = medians;
        this.means = means;
        this.stds = stds;
        this.standardized = standardized;
    }
}

public class PreProcessador {
    public static final String[] FEATURE_NAMES = {
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    };

    static Divisao stratifiedSplit(List<PontoDado> all, double trainRatio, double valRatio, long seed) {
        List<PontoDado> pos = new ArrayList<>();
        List<PontoDado> neg = new ArrayList<>();
        for (PontoDado dp : all) {
            if (dp.label == 1) {
                pos.add(dp);
            } else {
                neg.add(dp);
            }
        }

        Random rng = new Random(seed);
        Collections.shuffle(pos, rng);
        Collections.shuffle(neg, rng);

        List<PontoDado> train = new ArrayList<>();
        List<PontoDado> val = new ArrayList<>();
        List<PontoDado> test = new ArrayList<>();

        splitByRatio(pos, trainRatio, valRatio, train, val, test);
        splitByRatio(neg, trainRatio, valRatio, train, val, test);

        Collections.shuffle(train, rng);
        Collections.shuffle(val, rng);
        Collections.shuffle(test, rng);

        return new Divisao(train, val, test);
    }

    private static void splitByRatio(List<PontoDado> src, double trainRatio, double valRatio,
                                     List<PontoDado> train, List<PontoDado> val, List<PontoDado> test) {
        int n = src.size();
        int nTrain = (int) Math.round(n * trainRatio);
        int nVal = (int) Math.round(n * valRatio);
        if (nTrain + nVal > n) {
            nVal = Math.max(0, n - nTrain);
        }
        int nTest = n - nTrain - nVal;

        train.addAll(src.subList(0, nTrain));
        val.addAll(src.subList(nTrain, nTrain + nVal));
        test.addAll(src.subList(nTrain + nVal, nTrain + nVal + nTest));
    }

    static DadosPreparados fitAndTransform(Divisao split) {
        return fitAndTransform(split, true);
    }

    static DadosPreparados fitAndTransformWithoutScaling(Divisao split) {
        return fitAndTransform(split, false);
    }

    private static DadosPreparados fitAndTransform(Divisao split, boolean standardize) {
        double[] medians = computeMedians(split.train);
        double[][] xTrainImputed = impute(split.train, medians);
        double[][] xValImputed = impute(split.val, medians);
        double[][] xTestImputed = impute(split.test, medians);

        double[] means = UtilMatriz.columnMeans(xTrainImputed);
        double[] stds = UtilMatriz.columnStds(xTrainImputed, means);

        double[][] xTrain = standardize ? UtilMatriz.standardize(xTrainImputed, means, stds) : xTrainImputed;
        double[][] xVal = standardize ? UtilMatriz.standardize(xValImputed, means, stds) : xValImputed;
        double[][] xTest = standardize ? UtilMatriz.standardize(xTestImputed, means, stds) : xTestImputed;

        int[] yTrain = labels(split.train);
        int[] yVal = labels(split.val);
        int[] yTest = labels(split.test);

        return new DadosPreparados(xTrain, yTrain, xVal, yVal, xTest, yTest, FEATURE_NAMES, medians, means, stds, standardize);
    }

    static double[][] transformFeatures(List<PontoDado> data, DadosPreparados prepared) {
        double[][] imputed = impute(data, prepared.medians);
        if (!prepared.standardized) {
            return imputed;
        }
        return UtilMatriz.standardize(imputed, prepared.means, prepared.stds);
    }

    private static int[] labels(List<PontoDado> data) {
        int[] y = new int[data.size()];
        for (int i = 0; i < data.size(); i++) {
            y[i] = data.get(i).label;
        }
        return y;
    }

    private static double[][] impute(List<PontoDado> data, double[] medians) {
        double[][] x = new double[data.size()][FEATURE_NAMES.length];
        for (int i = 0; i < data.size(); i++) {
            for (int j = 0; j < FEATURE_NAMES.length; j++) {
                double v = data.get(i).features[j];
                x[i][j] = Double.isNaN(v) ? medians[j] : v;
            }
        }
        return x;
    }

    private static double[] computeMedians(List<PontoDado> data) {
        int d = FEATURE_NAMES.length;
        double[] medians = new double[d];

        for (int j = 0; j < d; j++) {
            List<Double> vals = new ArrayList<>();
            for (PontoDado dp : data) {
                double v = dp.features[j];
                if (!Double.isNaN(v)) {
                    vals.add(v);
                }
            }
            if (vals.isEmpty()) {
                medians[j] = 0.0;
                continue;
            }
            Collections.sort(vals);
            int n = vals.size();
            if (n % 2 == 1) {
                medians[j] = vals.get(n / 2);
            } else {
                medians[j] = (vals.get((n / 2) - 1) + vals.get(n / 2)) / 2.0;
            }
        }
        return medians;
    }
}


