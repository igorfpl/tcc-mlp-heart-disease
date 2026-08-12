import java.util.Arrays;
import java.util.Locale;
import java.util.Random;

class DadosAtributosKMeans {
    public final double[][] xTrain;
    public final double[][] xVal;
    public final double[][] xTest;
    public final int k;
    public final int fittedIterations;
    public final double inertia;
    public final double silhouette;
    public final String[] featureNames;
    public final double[][] centroids;
    public final double[] augmentedMeans;
    public final double[] augmentedStds;

    public DadosAtributosKMeans(double[][] xTrain, double[][] xVal, double[][] xTest,
                             int k, int fittedIterations, double inertia, double silhouette,
                             String[] featureNames, double[][] centroids,
                             double[] augmentedMeans, double[] augmentedStds) {
        this.xTrain = xTrain;
        this.xVal = xVal;
        this.xTest = xTest;
        this.k = k;
        this.fittedIterations = fittedIterations;
        this.inertia = inertia;
        this.silhouette = silhouette;
        this.featureNames = featureNames;
        this.centroids = centroids;
        this.augmentedMeans = augmentedMeans;
        this.augmentedStds = augmentedStds;
    }
}

class EngenhariaAtributosKMeans {
    static DadosAtributosKMeans fitBestAndTransform(double[][] xTrain, double[][] xVal, double[][] xTest,
                                                 String[] baseFeatureNames,
                                                 int minK, int maxK, int maxIterations, long seed) {
        KMeansRun bestRun = null;
        for (int k = minK; k <= maxK; k++) {
            KMeansRun run = fit(xTrain, k, maxIterations, seed + k);
            if (bestRun == null || run.silhouette > bestRun.silhouette) {
                bestRun = run;
            }
        }

        if (bestRun == null) {
            throw new IllegalStateException("Nao foi possivel ajustar o K-Means.");
        }

        double[][] trainAug = augment(xTrain, bestRun.centroids);
        double[][] valAug = augment(xVal, bestRun.centroids);
        double[][] testAug = augment(xTest, bestRun.centroids);

        double[] means = UtilMatriz.columnMeans(trainAug);
        double[] stds = UtilMatriz.columnStds(trainAug, means);

        return new DadosAtributosKMeans(
                UtilMatriz.standardize(trainAug, means, stds),
                UtilMatriz.standardize(valAug, means, stds),
                UtilMatriz.standardize(testAug, means, stds),
                bestRun.k,
                bestRun.iterations,
                bestRun.inertia,
                bestRun.silhouette,
                buildFeatureNames(baseFeatureNames, bestRun.k),
                copy2D(bestRun.centroids),
                Arrays.copyOf(means, means.length),
                Arrays.copyOf(stds, stds.length)
        );
    }

    static DadosAtributosKMeans fitAndTransformForK(double[][] xTrain, double[][] xVal, double[][] xTest,
                                                 String[] baseFeatureNames,
                                                 int k, int maxIterations, long seed) {
        KMeansRun run = fit(xTrain, k, maxIterations, seed);
        double[][] trainAug = augment(xTrain, run.centroids);
        double[][] valAug = augment(xVal, run.centroids);
        double[][] testAug = augment(xTest, run.centroids);

        double[] means = UtilMatriz.columnMeans(trainAug);
        double[] stds = UtilMatriz.columnStds(trainAug, means);

        return new DadosAtributosKMeans(
                UtilMatriz.standardize(trainAug, means, stds),
                UtilMatriz.standardize(valAug, means, stds),
                UtilMatriz.standardize(testAug, means, stds),
                run.k,
                run.iterations,
                run.inertia,
                run.silhouette,
                buildFeatureNames(baseFeatureNames, run.k),
                copy2D(run.centroids),
                Arrays.copyOf(means, means.length),
                Arrays.copyOf(stds, stds.length)
        );
    }

    static double[][] transformWithExistingModel(double[][] standardizedBaseFeatures, DadosAtributosKMeans fittedData) {
        double[][] augmented = augment(standardizedBaseFeatures, fittedData.centroids);
        return UtilMatriz.standardize(augmented, fittedData.augmentedMeans, fittedData.augmentedStds);
    }

    private static KMeansRun fit(double[][] x, int k, int maxIterations, long seed) {
        Random rng = new Random(seed);
        int n = x.length;
        int d = x[0].length;

        double[][] centroids = initKMeansPlusPlus(x, k, rng);
        int[] assignments = new int[n];
        Arrays.fill(assignments, -1);
        int iterations = 0;

        for (int iter = 0; iter < maxIterations; iter++) {
            iterations = iter + 1;
            boolean changed = false;
            for (int i = 0; i < n; i++) {
                int cluster = nearestCentroid(x[i], centroids);
                if (assignments[i] != cluster) {
                    assignments[i] = cluster;
                    changed = true;
                }
            }

            double[][] newCentroids = new double[k][d];
            int[] counts = new int[k];
            for (int i = 0; i < n; i++) {
                int cluster = assignments[i];
                counts[cluster]++;
                for (int j = 0; j < d; j++) {
                    newCentroids[cluster][j] += x[i][j];
                }
            }

            for (int c = 0; c < k; c++) {
                if (counts[c] == 0) {
                    int fallback = rng.nextInt(n);
                    newCentroids[c] = Arrays.copyOf(x[fallback], d);
                    counts[c] = 1;
                } else {
                    for (int j = 0; j < d; j++) {
                        newCentroids[c][j] /= counts[c];
                    }
                }
            }

            double shift = 0.0;
            for (int c = 0; c < k; c++) {
                shift += squaredDistance(centroids[c], newCentroids[c]);
            }
            centroids = newCentroids;

            if (!changed || shift < 1e-8) {
                break;
            }
        }

        double inertia = 0.0;
        for (int i = 0; i < n; i++) {
            inertia += squaredDistance(x[i], centroids[assignments[i]]);
        }
        double silhouette = silhouetteScore(x, assignments, k);
        return new KMeansRun(k, centroids, assignments, iterations, inertia, silhouette);
    }

    private static double[][] initKMeansPlusPlus(double[][] x, int k, Random rng) {
        int n = x.length;
        int d = x[0].length;
        double[][] centroids = new double[k][d];
        int first = rng.nextInt(n);
        centroids[0] = Arrays.copyOf(x[first], d);

        double[] minDist = new double[n];
        Arrays.fill(minDist, Double.POSITIVE_INFINITY);

        for (int c = 1; c < k; c++) {
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                minDist[i] = Math.min(minDist[i], squaredDistance(x[i], centroids[c - 1]));
                sum += minDist[i];
            }

            if (sum <= 1e-12) {
                centroids[c] = Arrays.copyOf(x[rng.nextInt(n)], d);
                continue;
            }

            double r = rng.nextDouble() * sum;
            double cumulative = 0.0;
            int chosen = n - 1;
            for (int i = 0; i < n; i++) {
                cumulative += minDist[i];
                if (cumulative >= r) {
                    chosen = i;
                    break;
                }
            }
            centroids[c] = Arrays.copyOf(x[chosen], d);
        }
        return centroids;
    }

    private static double[][] augment(double[][] x, double[][] centroids) {
        int n = x.length;
        int d = x[0].length;
        int k = centroids.length;
        double[][] out = new double[n][d + k + k];

        for (int i = 0; i < n; i++) {
            System.arraycopy(x[i], 0, out[i], 0, d);
            int nearest = 0;
            double bestDist = Double.POSITIVE_INFINITY;
            for (int c = 0; c < k; c++) {
                double dist = euclideanDistance(x[i], centroids[c]);
                out[i][d + c] = dist;
                if (dist < bestDist) {
                    bestDist = dist;
                    nearest = c;
                }
            }
            out[i][d + k + nearest] = 1.0;
        }
        return out;
    }

    private static String[] buildFeatureNames(String[] baseFeatureNames, int k) {
        String[] names = new String[baseFeatureNames.length + (2 * k)];
        int pos = 0;
        for (String baseFeatureName : baseFeatureNames) {
            names[pos++] = baseFeatureName;
        }
        for (int i = 0; i < k; i++) {
            names[pos++] = String.format(Locale.US, "dist_cluster_%d", i);
        }
        for (int i = 0; i < k; i++) {
            names[pos++] = String.format(Locale.US, "onehot_cluster_%d", i);
        }
        return names;
    }

    private static double silhouetteScore(double[][] x, int[] assignments, int k) {
        int n = x.length;
        int[] counts = new int[k];
        for (int assignment : assignments) {
            counts[assignment]++;
        }

        double total = 0.0;
        for (int i = 0; i < n; i++) {
            int own = assignments[i];
            double a = averageDistanceToCluster(x, assignments, i, own, counts[own]);
            double b = Double.POSITIVE_INFINITY;
            for (int c = 0; c < k; c++) {
                if (c == own || counts[c] == 0) {
                    continue;
                }
                double candidate = averageDistanceToCluster(x, assignments, i, c, counts[c]);
                if (candidate < b) {
                    b = candidate;
                }
            }
            if (!Double.isFinite(b)) {
                b = a;
            }
            double denom = Math.max(a, b);
            if (denom > 1e-12) {
                total += (b - a) / denom;
            }
        }
        return total / Math.max(1, n);
    }

    private static double averageDistanceToCluster(double[][] x, int[] assignments, int index, int cluster, int clusterCount) {
        if (clusterCount <= 1 && assignments[index] == cluster) {
            return 0.0;
        }
        double sum = 0.0;
        int count = 0;
        for (int i = 0; i < x.length; i++) {
            if (assignments[i] != cluster || i == index) {
                continue;
            }
            sum += euclideanDistance(x[index], x[i]);
            count++;
        }
        return count == 0 ? 0.0 : sum / count;
    }

    private static int nearestCentroid(double[] row, double[][] centroids) {
        int best = 0;
        double bestDist = Double.POSITIVE_INFINITY;
        for (int c = 0; c < centroids.length; c++) {
            double dist = squaredDistance(row, centroids[c]);
            if (dist < bestDist) {
                bestDist = dist;
                best = c;
            }
        }
        return best;
    }

    private static double squaredDistance(double[] a, double[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            s += diff * diff;
        }
        return s;
    }

    private static double euclideanDistance(double[] a, double[] b) {
        return Math.sqrt(squaredDistance(a, b));
    }

    private static double[][] copy2D(double[][] data) {
        double[][] out = new double[data.length][];
        for (int i = 0; i < data.length; i++) {
            out[i] = Arrays.copyOf(data[i], data[i].length);
        }
        return out;
    }

    private static class KMeansRun {
        final int k;
        final double[][] centroids;
        final int[] assignments;
        final int iterations;
        final double inertia;
        final double silhouette;

        KMeansRun(int k, double[][] centroids, int[] assignments, int iterations, double inertia, double silhouette) {
            this.k = k;
            this.centroids = centroids;
            this.assignments = assignments;
            this.iterations = iterations;
            this.inertia = inertia;
            this.silhouette = silhouette;
        }
    }
}




