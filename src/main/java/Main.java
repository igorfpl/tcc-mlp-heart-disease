import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) throws Exception {
        Path root = Path.of(".").toAbsolutePath().normalize();
        Path dataDir = root.resolve("Dados");

        List<Path> rawFiles = List.of(
                dataDir.resolve("hungarian.data"),
                dataDir.resolve("switzerland.data"),
                dataDir.resolve("long-beach-va.data")
        );

        List<PontoDado> all = CarregadorDados.loadRawFiles(rawFiles);
        System.out.printf("Registros validos carregados (raw): %d%n", all.size());
        System.out.println("\n=== COMPARACAO COM DADOS ORIGINAIS ===");

        ResultadoExperimento experimento = executarExperimento(all, dataDir);
        GeradorRelatorios.writeOutputs(root, all, experimento.split, experimento.normalPrepared,
                experimento.kMeansPrepared, experimento.normalModel, experimento.results,
                experimento.costModel, experimento.kMeansExperiments);

        System.out.println("\nExperimento com dados originais concluido.");
    }

    private static ResultadoExperimento executarExperimento(List<PontoDado> dados, Path dataDir) throws Exception {
        System.out.printf("\n=== EXECUTANDO DATASET ORIGINAL (%d registros) ===\n", dados.size());

        Divisao split = PreProcessador.stratifiedSplit(dados, 0.70, 0.15, 42L);
        DadosPreparados normalPrepared = PreProcessador.fitAndTransformWithoutScaling(split);
        DadosPreparados kMeansPrepared = PreProcessador.fitAndTransform(split);

        List<ResultadoModelo> results = new ArrayList<>();
        List<ExperimentoKMeans> kMeansExperiments = new ArrayList<>();
        int[] fullIdx = UtilMatriz.range(0, normalPrepared.featureNames.length);

        MlpModel normalModel = new MlpModel(normalPrepared.xTrain[0].length, 32, 16,
                0.0005, 400, 32, 1e-4, 42L);
        normalModel.fit(normalPrepared.xTrain, normalPrepared.yTrain,
                normalPrepared.xVal, normalPrepared.yVal);
        double[] probsMlpVal = normalModel.predictProba(normalPrepared.xVal);
        double[] probsMlp = normalModel.predictProba(normalPrepared.xTest);
        results.add(ResultadoModelo.fromValidationAndTest(
                "MLP-Normal(13f)", fullIdx,
                normalPrepared.yVal, probsMlpVal,
                normalPrepared.yTest, probsMlp
        ));

        for (int k = 2; k <= 10; k++) {
            DadosAtributosKMeans kMeansData = EngenhariaAtributosKMeans.fitAndTransformForK(
                    kMeansPrepared.xTrain, kMeansPrepared.xVal, kMeansPrepared.xTest,
                    kMeansPrepared.featureNames, k, 100, 42L + k
            );

            MlpModel mlpKMeans = new MlpModel(kMeansData.xTrain[0].length, 40, 20,
                    0.01, 400, 32, 1e-4, 42L + k);
            mlpKMeans.fit(kMeansData.xTrain, kMeansPrepared.yTrain,
                    kMeansData.xVal, kMeansPrepared.yVal);
            double[] probsMlpKMeansVal = mlpKMeans.predictProba(kMeansData.xVal);
            double[] probsMlpKMeans = mlpKMeans.predictProba(kMeansData.xTest);
            ResultadoModelo result = ResultadoModelo.fromValidationAndTest(
                    "MLP+KMeans(k=" + kMeansData.k + ")", fullIdx,
                    kMeansPrepared.yVal, probsMlpKMeansVal,
                    kMeansPrepared.yTest, probsMlpKMeans
            );
            results.add(result);
            kMeansExperiments.add(new ExperimentoKMeans(kMeansData, result, mlpKMeans));
        }

        ModeloCusto costModel = ModeloCusto.load(dataDir.resolve("costs"));
        for (ResultadoModelo result : results) {
            result.featureAcquisitionCost = costModel.totalFeatureCost(
                    result.featureIndexes, normalPrepared.featureNames);
            result.delayedTests = costModel.countDelayedTests(
                    result.featureIndexes, normalPrepared.featureNames);
        }

        GeradorRelatorios.printConsoleTable(results);
        System.out.println("\nResumo dos K testados:");
        for (ExperimentoKMeans experiment : kMeansExperiments) {
            DadosAtributosKMeans data = experiment.featureData;
            System.out.printf("k=%d | silhouette=%.4f | inertia=%.4f | iteracoes=%d | F1=%.3f | AUC=%.3f%n",
                    data.k, data.silhouette, data.inertia, data.fittedIterations,
                    experiment.result.f1, experiment.result.auc);
        }

        return new ResultadoExperimento(split, normalPrepared, kMeansPrepared, normalModel,
                results, costModel, kMeansExperiments);
    }

    private static class ResultadoExperimento {
        final Divisao split;
        final DadosPreparados normalPrepared;
        final DadosPreparados kMeansPrepared;
        final MlpModel normalModel;
        final List<ResultadoModelo> results;
        final ModeloCusto costModel;
        final List<ExperimentoKMeans> kMeansExperiments;

        ResultadoExperimento(Divisao split, DadosPreparados normalPrepared,
                             DadosPreparados kMeansPrepared, MlpModel normalModel,
                             List<ResultadoModelo> results, ModeloCusto costModel,
                             List<ExperimentoKMeans> kMeansExperiments) {
            this.split = split;
            this.normalPrepared = normalPrepared;
            this.kMeansPrepared = kMeansPrepared;
            this.normalModel = normalModel;
            this.results = results;
            this.costModel = costModel;
            this.kMeansExperiments = kMeansExperiments;
        }
    }
}
