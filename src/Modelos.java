import java.util.Arrays;
import java.util.Random;

interface BinaryClassifier {
    void fit(double[][] x, int[] y);

    double[] predictProba(double[][] x);
}

class LogisticRegressionModel implements BinaryClassifier {
    private final double learningRate;
    private final int epochs;
    private final double l2;
    private final Random rng;

    private double[] w;
    private double b;

    public LogisticRegressionModel(double learningRate, int epochs, double l2, long seed) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.l2 = l2;
        this.rng = new Random(seed);
    }

    @Override
    public void fit(double[][] x, int[] y) {
        int n = x.length;
        int d = x[0].length;
        w = new double[d];
        for (int j = 0; j < d; j++) {
            w[j] = (rng.nextDouble() - 0.5) * 0.01;
        }
        b = 0.0;

        int pos = 0;
        for (int v : y) {
            if (v == 1) pos++;
        }
        int neg = n - pos;
        double posWeight = pos == 0 ? 1.0 : (double) neg / Math.max(1, pos);

        for (int epoch = 0; epoch < epochs; epoch++) {
            double[] gradW = new double[d];
            double gradB = 0.0;

            for (int i = 0; i < n; i++) {
                double z = dot(x[i], w) + b;
                double p = sigmoid(z);
                double yi = y[i];
                double sampleWeight = yi == 1.0 ? posWeight : 1.0;
                double diff = (p - yi) * sampleWeight;

                for (int j = 0; j < d; j++) {
                    gradW[j] += diff * x[i][j];
                }
                gradB += diff;
            }

            for (int j = 0; j < d; j++) {
                gradW[j] = (gradW[j] / n) + l2 * w[j];
                w[j] -= learningRate * gradW[j];
            }
            gradB /= n;
            b -= learningRate * gradB;
        }
    }

    @Override
    public double[] predictProba(double[][] x) {
        double[] probs = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            probs[i] = sigmoid(dot(x[i], w) + b);
        }
        return probs;
    }

    private static double dot(double[] a, double[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    private static double sigmoid(double z) {
        if (z >= 0) {
            double e = Math.exp(-z);
            return 1.0 / (1.0 + e);
        }
        double e = Math.exp(z);
        return e / (1.0 + e);
    }
}

class MlpModel {
    private final int inputSize;
    private final int h1;
    private final int h2;
    private final double lr;
    private final int epochs;
    private final int batchSize;
    private final double l2;
    private final Random rng;

    private double[][] w1;
    private double[] b1;
    private double[][] w2;
    private double[] b2;
    private double[] w3;
    private double b3;

    public MlpModel(int inputSize, int h1, int h2, double lr, int epochs, int batchSize, double l2, long seed) {
        this.inputSize = inputSize;
        this.h1 = h1;
        this.h2 = h2;
        this.lr = lr;
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.l2 = l2;
        this.rng = new Random(seed);
        initWeights();
    }

    private void initWeights() {
        w1 = new double[h1][inputSize];
        b1 = new double[h1];
        w2 = new double[h2][h1];
        b2 = new double[h2];
        w3 = new double[h2];
        b3 = 0.0;

        heInit(w1, inputSize);
        heInit(w2, h1);
        double scale = Math.sqrt(2.0 / h2);
        for (int i = 0; i < h2; i++) {
            w3[i] = rng.nextGaussian() * scale;
        }
    }

    private void heInit(double[][] w, int fanIn) {
        double scale = Math.sqrt(2.0 / fanIn);
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[i].length; j++) {
                w[i][j] = rng.nextGaussian() * scale;
            }
        }
    }

    public void fit(double[][] xTrain, int[] yTrain, double[][] xVal, int[] yVal) {
        int n = xTrain.length;
        int pos = 0;
        for (int v : yTrain) {
            if (v == 1) pos++;
        }
        int neg = n - pos;
        double posWeight = pos == 0 ? 1.0 : (double) neg / Math.max(1, pos);

        double bestValAuc = -1.0;
        int stale = 0;
        MlpSnapshot best = null;

        for (int epoch = 0; epoch < epochs; epoch++) {
            int[] idx = UtilMatriz.shuffledIndexes(n, rng);

            for (int start = 0; start < n; start += batchSize) {
                int end = Math.min(n, start + batchSize);
                int bs = end - start;

                double[][] gw1 = new double[h1][inputSize];
                double[] gb1 = new double[h1];
                double[][] gw2 = new double[h2][h1];
                double[] gb2 = new double[h2];
                double[] gw3 = new double[h2];
                double gb3 = 0.0;

                for (int b = start; b < end; b++) {
                    int i = idx[b];
                    double[] x = xTrain[i];
                    double y = yTrain[i];
                    double sw = y == 1.0 ? posWeight : 1.0;

                    double[] z1 = new double[h1];
                    double[] a1 = new double[h1];
                    for (int j = 0; j < h1; j++) {
                        double s = b1[j];
                        for (int k = 0; k < inputSize; k++) {
                            s += w1[j][k] * x[k];
                        }
                        z1[j] = s;
                        a1[j] = relu(s);
                    }

                    double[] z2 = new double[h2];
                    double[] a2 = new double[h2];
                    for (int j = 0; j < h2; j++) {
                        double s = b2[j];
                        for (int k = 0; k < h1; k++) {
                            s += w2[j][k] * a1[k];
                        }
                        z2[j] = s;
                        a2[j] = relu(s);
                    }

                    double z3 = b3;
                    for (int j = 0; j < h2; j++) {
                        z3 += w3[j] * a2[j];
                    }
                    double p = sigmoid(z3);

                    double dz3 = (p - y) * sw;
                    for (int j = 0; j < h2; j++) {
                        gw3[j] += dz3 * a2[j];
                    }
                    gb3 += dz3;

                    double[] da2 = new double[h2];
                    for (int j = 0; j < h2; j++) {
                        da2[j] = dz3 * w3[j];
                    }

                    double[] dz2 = new double[h2];
                    for (int j = 0; j < h2; j++) {
                        dz2[j] = z2[j] > 0 ? da2[j] : 0.0;
                        for (int k = 0; k < h1; k++) {
                            gw2[j][k] += dz2[j] * a1[k];
                        }
                        gb2[j] += dz2[j];
                    }

                    double[] da1 = new double[h1];
                    for (int k = 0; k < h1; k++) {
                        double s = 0.0;
                        for (int j = 0; j < h2; j++) {
                            s += dz2[j] * w2[j][k];
                        }
                        da1[k] = s;
                    }

                    for (int j = 0; j < h1; j++) {
                        double dz1 = z1[j] > 0 ? da1[j] : 0.0;
                        for (int k = 0; k < inputSize; k++) {
                            gw1[j][k] += dz1 * x[k];
                        }
                        gb1[j] += dz1;
                    }
                }

                double inv = 1.0 / bs;
                for (int j = 0; j < h1; j++) {
                    for (int k = 0; k < inputSize; k++) {
                        gw1[j][k] = gw1[j][k] * inv + l2 * w1[j][k];
                        w1[j][k] -= lr * gw1[j][k];
                    }
                    b1[j] -= lr * gb1[j] * inv;
                }

                for (int j = 0; j < h2; j++) {
                    for (int k = 0; k < h1; k++) {
                        gw2[j][k] = gw2[j][k] * inv + l2 * w2[j][k];
                        w2[j][k] -= lr * gw2[j][k];
                    }
                    b2[j] -= lr * gb2[j] * inv;
                }

                for (int j = 0; j < h2; j++) {
                    gw3[j] = gw3[j] * inv + l2 * w3[j];
                    w3[j] -= lr * gw3[j];
                }
                b3 -= lr * gb3 * inv;
            }

            double[] valProbs = predictProba(xVal);
            double valAuc = ResultadoModelo.auc(yVal, valProbs);
            if (valAuc > bestValAuc + 1e-5) {
                bestValAuc = valAuc;
                best = snapshot();
                stale = 0;
            } else {
                stale++;
                if (stale >= 25) {
                    break;
                }
            }
        }

        if (best != null) {
            restore(best);
        }
    }

    public double[] predictProba(double[][] x) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            double[] a1 = new double[h1];
            for (int j = 0; j < h1; j++) {
                double s = b1[j];
                for (int k = 0; k < inputSize; k++) {
                    s += w1[j][k] * x[i][k];
                }
                a1[j] = relu(s);
            }

            double[] a2 = new double[h2];
            for (int j = 0; j < h2; j++) {
                double s = b2[j];
                for (int k = 0; k < h1; k++) {
                    s += w2[j][k] * a1[k];
                }
                a2[j] = relu(s);
            }

            double s = b3;
            for (int j = 0; j < h2; j++) {
                s += w3[j] * a2[j];
            }
            out[i] = sigmoid(s);
        }
        return out;
    }

    private MlpSnapshot snapshot() {
        return new MlpSnapshot(copy2D(w1), Arrays.copyOf(b1, b1.length), copy2D(w2),
                Arrays.copyOf(b2, b2.length), Arrays.copyOf(w3, w3.length), b3);
    }

    private void restore(MlpSnapshot s) {
        this.w1 = copy2D(s.w1);
        this.b1 = Arrays.copyOf(s.b1, s.b1.length);
        this.w2 = copy2D(s.w2);
        this.b2 = Arrays.copyOf(s.b2, s.b2.length);
        this.w3 = Arrays.copyOf(s.w3, s.w3.length);
        this.b3 = s.b3;
    }

    private static double[][] copy2D(double[][] m) {
        double[][] c = new double[m.length][];
        for (int i = 0; i < m.length; i++) {
            c[i] = Arrays.copyOf(m[i], m[i].length);
        }
        return c;
    }

    private static double relu(double x) {
        return x > 0 ? x : 0.0;
    }

    private static double sigmoid(double z) {
        if (z >= 0) {
            double e = Math.exp(-z);
            return 1.0 / (1.0 + e);
        }
        double e = Math.exp(z);
        return e / (1.0 + e);
    }

    private static class MlpSnapshot {
        final double[][] w1;
        final double[] b1;
        final double[][] w2;
        final double[] b2;
        final double[] w3;
        final double b3;

        MlpSnapshot(double[][] w1, double[] b1, double[][] w2, double[] b2, double[] w3, double b3) {
            this.w1 = w1;
            this.b1 = b1;
            this.w2 = w2;
            this.b2 = b2;
            this.w3 = w3;
            this.b3 = b3;
        }
    }
}


