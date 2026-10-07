import java.util.Random;

public class UtilMatriz {
    public static int[] range(int startInclusive, int endExclusive) {
        int[] r = new int[endExclusive - startInclusive];
        for (int i = 0; i < r.length; i++) {
            r[i] = startInclusive + i;
        }
        return r;
    }

    public static double[][] subsetColumns(double[][] x, int[] columns) {
        double[][] out = new double[x.length][columns.length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < columns.length; j++) {
                out[i][j] = x[i][columns[j]];
            }
        }
        return out;
    }

    public static double[] columnMeans(double[][] x) {
        int n = x.length;
        int d = x[0].length;
        double[] means = new double[d];
        for (double[] row : x) {
            for (int j = 0; j < d; j++) {
                means[j] += row[j];
            }
        }
        for (int j = 0; j < d; j++) {
            means[j] /= n;
        }
        return means;
    }

    public static double[] columnStds(double[][] x, double[] means) {
        int n = x.length;
        int d = x[0].length;
        double[] stds = new double[d];
        for (double[] row : x) {
            for (int j = 0; j < d; j++) {
                double diff = row[j] - means[j];
                stds[j] += diff * diff;
            }
        }
        for (int j = 0; j < d; j++) {
            stds[j] = Math.sqrt(stds[j] / Math.max(1, n - 1));
            if (stds[j] < 1e-9) {
                stds[j] = 1.0;
            }
        }
        return stds;
    }

    public static double[][] standardize(double[][] x, double[] means, double[] stds) {
        double[][] out = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                out[i][j] = (x[i][j] - means[j]) / stds[j];
            }
        }
        return out;
    }

    public static int[] shuffledIndexes(int n, Random rng) {
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) {
            idx[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = idx[i];
            idx[i] = idx[j];
            idx[j] = tmp;
        }
        return idx;
    }
}


