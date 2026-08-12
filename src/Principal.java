import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class Principal {
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

        // 1. Treinar com dados originais
        System.out.println("\n=== FASE 1: Testando com dados originais ===");
        Map<String, ResultadoVolume> volumeResults = new LinkedHashMap<>();

        // Volume 1x (original 617 registros)
        executarExperimento(all, 1, volumeResults, root, dataDir);

        // Gerar dados sintéticos para 2x
        System.out.println("\n=== GERANDO DADOS SINTÉTICOS 2x ===");
        GeradorDadosSinteticos gerador2x = new GeradorDadosSinteticos(all, 100L);
        List<PontoDado> sinteticos2x = gerador2x.gerarDadosSinteticos(all.size());
        List<PontoDado> all2x = new ArrayList<>(all);
        all2x.addAll(sinteticos2x);
        System.out.printf("Total 2x: %d registros (original: %d + sintético: %d)%n",
                all2x.size(), all.size(), sinteticos2x.size());

        // Volume 2x
        executarExperimento(all2x, 2, volumeResults, root, dataDir);

        // Gerar dados sintéticos para 3x
        System.out.println("\n=== GERANDO DADOS SINTÉTICOS 3x ===");
        GeradorDadosSinteticos gerador3x = new GeradorDadosSinteticos(all, 101L);
        List<PontoDado> sinteticos3x = gerador3x.gerarDadosSinteticos(all.size() * 2);
        List<PontoDado> all3x = new ArrayList<>(all);
        all3x.addAll(sinteticos3x);
        System.out.printf("Total 3x: %d registros (original: %d + sintético: %d)%n",
                all3x.size(), all.size(), sinteticos3x.size());

        // Volume 3x
        executarExperimento(all3x, 3, volumeResults, root, dataDir);

        // Gerar relatório de comparação
        System.out.println("\n=== GERANDO RELATÓRIO DE COMPARAÇÃO ===");
        GeradorRelatorios.writeComparacaoVolumes(root, volumeResults);

        System.out.println("\nTodos os experimentos completos!");
    }

    private static void executarExperimento(List<PontoDado> dados, int volumeMultiplier,
                                           Map<String, ResultadoVolume> volumeResults,
                                           Path root, Path dataDir) throws Exception {
        String volumeLabel = volumeMultiplier + "x";
        System.out.printf("\n=== EXECUTANDO VOLUME %s (%d registros) ===\n", volumeLabel, dados.size());

        Divisao split = PreProcessador.stratifiedSplit(dados, 0.70, 0.15, 42L);
        DadosPreparados normalPrepared = PreProcessador.fitAndTransformWithoutScaling(split);
        DadosPreparados kMeansPrepared = PreProcessador.fitAndTransform(split);

        List<ResultadoModelo> results = new ArrayList<>();
        List<ExperimentoKMeans> kMeansExperiments = new ArrayList<>();
        int[] fullIdx = UtilMatriz.range(0, normalPrepared.featureNames.length);

        // MLP Normal
        MlpModel mlp = new MlpModel(normalPrepared.xTrain[0].length, 32, 16, 0.0005, 400, 32, 1e-4, 42L);
        mlp.fit(normalPrepared.xTrain, normalPrepared.yTrain, normalPrepared.xVal, normalPrepared.yVal);
        double[] probsMlpVal = mlp.predictProba(normalPrepared.xVal);
        double[] probsMlp = mlp.predictProba(normalPrepared.xTest);
        results.add(ResultadoModelo.fromValidationAndTest(
                "MLP-Normal(13f)", fullIdx,
                normalPrepared.yVal, probsMlpVal,
                normalPrepared.yTest, probsMlp
        ));

        // MLP + KMeans (K=2 até 6)
        for (int k = 2; k <= 6; k++) {
            DadosAtributosKMeans kMeansData = EngenhariaAtributosKMeans.fitAndTransformForK(
                    kMeansPrepared.xTrain, kMeansPrepared.xVal, kMeansPrepared.xTest,
                    kMeansPrepared.featureNames,
                    k, 100, 42L + k
            );

            MlpModel mlpKMeans = new MlpModel(kMeansData.xTrain[0].length, 40, 20, 0.01, 400, 32, 1e-4, 42L + k);
            mlpKMeans.fit(kMeansData.xTrain, kMeansPrepared.yTrain, kMeansData.xVal, kMeansPrepared.yVal);
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
            result.featureAcquisitionCost = costModel.totalFeatureCost(result.featureIndexes, normalPrepared.featureNames);
            result.delayedTests = costModel.countDelayedTests(result.featureIndexes, normalPrepared.featureNames);
        }

        GeradorRelatorios.printConsoleTable(results);

        // Armazenar resultados por volume
        ResultadoVolume volumeResult = new ResultadoVolume(volumeLabel, dados.size(), results, kMeansExperiments);
        volumeResults.put(volumeLabel, volumeResult);

        System.out.println("\nResumo dos K testados para volume " + volumeLabel + ":");
        for (ExperimentoKMeans experiment : kMeansExperiments) {
            DadosAtributosKMeans data = experiment.featureData;
            System.out.printf("k=%d | silhouette=%.4f | inertia=%.4f | iteracoes=%d | F1=%.3f | AUC=%.3f%n",
                    data.k, data.silhouette, data.inertia, data.fittedIterations,
                    experiment.result.f1, experiment.result.auc);
        }
    }

    /**
     * Classe auxiliar para armazenar resultados por volume
     */
    static class ResultadoVolume {
        String volumeLabel;
        int totalRecords;
        List<ResultadoModelo> results;
        List<ExperimentoKMeans> kMeansExperiments;

        ResultadoVolume(String volumeLabel, int totalRecords,
                       List<ResultadoModelo> results, List<ExperimentoKMeans> kMeansExperiments) {
            this.volumeLabel = volumeLabel;
            this.totalRecords = totalRecords;
            this.results = results;
            this.kMeansExperiments = kMeansExperiments;
        }
    }
}

