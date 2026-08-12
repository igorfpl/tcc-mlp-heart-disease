import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Compara os resultados do MLP com e sem KMeans em diferentes tamanhos de datasets
 */
public class ComparadorDatasets {

    static class ResultadoComparacao {
        String tamanho;
        int registros;
        double f1SemKMeans;
        double aucSemKMeans;
        double f1ComKMeans;
        double aucComKMeans;
        double melhoriaF1;
        double melhoriaAUC;
        long tempoSemKMeans;
        long tempoComKMeans;

        ResultadoComparacao(String tamanho, int registros) {
            this.tamanho = tamanho;
            this.registros = registros;
        }
    }

    public static void main(String[] args) throws Exception {
        Path root = Path.of(".").toAbsolutePath().normalize();
        Path dataDir = root.resolve("Dados");

        System.out.println("╔════════════════════════════════════════════════════════════════╗");
        System.out.println("║     COMPARAÇÃO DE RESULTADOS: MLP Sem vs Com KMeans            ║");
        System.out.println("║        Em Diferentes Tamanhos de Datasets                      ║");
        System.out.println("╚════════════════════════════════════════════════════════════════╝\n");

        List<ResultadoComparacao> comparacoes = new ArrayList<>();

        // Testar com dados originais
        System.out.println("\n[1/3] Testando com dados ORIGINAIS...");
        ResultadoComparacao orig = testarDataset(
                List.of(
                    dataDir.resolve("hungarian.data"),
                    dataDir.resolve("switzerland.data"),
                    dataDir.resolve("long-beach-va.data")
                ),
                "Original",
                617
        );
        comparacoes.add(orig);

        // Testar com dados 2x
        System.out.println("\n[2/3] Testando com dados 2x...");
        ResultadoComparacao duplo = testarDataset(
                List.of(
                    dataDir.resolve("hungarian_2x.data"),
                    dataDir.resolve("switzerland_2x.data"),
                    dataDir.resolve("long-beach-va_2x.data")
                ),
                "2x",
                617 * 2
        );
        comparacoes.add(duplo);

        // Testar com dados 3x
        System.out.println("\n[3/3] Testando com dados 3x...");
        ResultadoComparacao triplo = testarDataset(
                List.of(
                    dataDir.resolve("hungarian_3x.data"),
                    dataDir.resolve("switzerland_3x.data"),
                    dataDir.resolve("long-beach-va_3x.data")
                ),
                "3x",
                617 * 3
        );
        comparacoes.add(triplo);

        // Exibir resumo comparativo
        exibirResumo(comparacoes);

        // Exibir conclusões
        exibirConclusoes(comparacoes);
    }

    private static ResultadoComparacao testarDataset(List<Path> files, String tamanho, int totalRegistros) throws Exception {
        ResultadoComparacao resultado = new ResultadoComparacao(tamanho, totalRegistros);

        // Carregar dados
        List<PontoDado> all = CarregadorDados.loadRawFiles(files);
        System.out.printf("  Registros válidos carregados: %d%n", all.size());

        // Split
        Divisao split = PreProcessador.stratifiedSplit(all, 0.70, 0.15, 42L);
        DadosPreparados normalPrepared = PreProcessador.fitAndTransformWithoutScaling(split);
        DadosPreparados kMeansPrepared = PreProcessador.fitAndTransform(split);

        // ========== TESTE 1: SEM KMEANS ==========
        System.out.println("    → Treinando MLP SEM KMeans...");
        long inicioSem = System.currentTimeMillis();

        MlpModel mlpNormal = new MlpModel(normalPrepared.xTrain[0].length, 32, 16, 0.0005, 400, 32, 1e-4, 42L);
        mlpNormal.fit(normalPrepared.xTrain, normalPrepared.yTrain, normalPrepared.xVal, normalPrepared.yVal);
        double[] probsMlpVal = mlpNormal.predictProba(normalPrepared.xVal);
        double[] probsMlp = mlpNormal.predictProba(normalPrepared.xTest);

        long fimSem = System.currentTimeMillis();
        resultado.tempoSemKMeans = fimSem - inicioSem;

        ResultadoModelo resultadoSem = ResultadoModelo.fromValidationAndTest(
                "MLP-Normal(13f)", UtilMatriz.range(0, normalPrepared.featureNames.length),
                normalPrepared.yVal, probsMlpVal,
                normalPrepared.yTest, probsMlp
        );

        resultado.f1SemKMeans = resultadoSem.f1;
        resultado.aucSemKMeans = resultadoSem.auc;

        System.out.printf("      F1 = %.4f, AUC = %.4f (tempo: %dms)%n",
                resultado.f1SemKMeans, resultado.aucSemKMeans, resultado.tempoSemKMeans);

        // ========== TESTE 2: COM KMEANS (K=3, mais comum) ==========
        System.out.println("    → Treinando MLP COM KMeans (k=3)...");
        long inicioComKMeans = System.currentTimeMillis();

        DadosAtributosKMeans kMeansData = EngenhariaAtributosKMeans.fitAndTransformForK(
                kMeansPrepared.xTrain, kMeansPrepared.xVal, kMeansPrepared.xTest,
                kMeansPrepared.featureNames,
                3, 100, 45L
        );

        MlpModel mlpKMeans = new MlpModel(kMeansData.xTrain[0].length, 40, 20, 0.01, 400, 32, 1e-4, 45L);
        mlpKMeans.fit(kMeansData.xTrain, kMeansPrepared.yTrain, kMeansData.xVal, kMeansPrepared.yVal);
        double[] probsMlpKMeansVal = mlpKMeans.predictProba(kMeansData.xVal);
        double[] probsMlpKMeans = mlpKMeans.predictProba(kMeansData.xTest);

        long fimComKMeans = System.currentTimeMillis();
        resultado.tempoComKMeans = fimComKMeans - inicioComKMeans;

        ResultadoModelo resultadoComKMeans = ResultadoModelo.fromValidationAndTest(
                "MLP+KMeans(k=3)", UtilMatriz.range(0, normalPrepared.featureNames.length),
                kMeansPrepared.yVal, probsMlpKMeansVal,
                kMeansPrepared.yTest, probsMlpKMeans
        );

        resultado.f1ComKMeans = resultadoComKMeans.f1;
        resultado.aucComKMeans = resultadoComKMeans.auc;

        System.out.printf("      F1 = %.4f, AUC = %.4f (tempo: %dms)%n",
                resultado.f1ComKMeans, resultado.aucComKMeans, resultado.tempoComKMeans);

        // Calcular melhorias
        resultado.melhoriaF1 = ((resultado.f1ComKMeans - resultado.f1SemKMeans) / resultado.f1SemKMeans) * 100;
        resultado.melhoriaAUC = ((resultado.aucComKMeans - resultado.aucSemKMeans) / resultado.aucSemKMeans) * 100;

        System.out.printf("    ✓ Melhoria com KMeans: F1 = %+.2f%%, AUC = %+.2f%%%n",
                resultado.melhoriaF1, resultado.melhoriaAUC);

        return resultado;
    }

    private static void exibirResumo(List<ResultadoComparacao> comparacoes) {
        System.out.println("\n╔════════════════════════════════════════════════════════════════╗");
        System.out.println("║                    RESUMO COMPARATIVO                           ║");
        System.out.println("╚════════════════════════════════════════════════════════════════╝\n");

        System.out.println("Tamanho | Registros |  F1 Sem KM  |  F1 Com KM  |  Melhoria  | Tempo(ms)");
        System.out.println("--------|-----------|------------|------------|------------|----------");
        for (ResultadoComparacao r : comparacoes) {
            System.out.printf(Locale.US,
                    "%7s | %9d | %.4f→%.4f | %.4f→%.4f | %+7.2f%% | %4d→%4d%n",
                    r.tamanho, r.registros,
                    r.f1SemKMeans, r.f1ComKMeans,
                    r.aucSemKMeans, r.aucComKMeans,
                    r.melhoriaF1,
                    r.tempoSemKMeans, r.tempoComKMeans);
        }

        System.out.println("\n\n┌────────────────────────────────────────────────────────────────┐");
        System.out.println("│  PERFORMANCE (F1 e AUC por dataset)                            │");
        System.out.println("└────────────────────────────────────────────────────────────────┘\n");

        System.out.println("Dataset  | Sem KMeans         | Com KMeans         | Diferença");
        System.out.println("         |   F1      AUC      |   F1      AUC      |  F1   AUC");
        System.out.println("---------|--------------------|--------------------|----------");
        for (ResultadoComparacao r : comparacoes) {
            System.out.printf(Locale.US,
                    "%8s | %.4f   %.4f    | %.4f   %.4f    | %+.2f %% %+.2f %%%n",
                    r.tamanho,
                    r.f1SemKMeans, r.aucSemKMeans,
                    r.f1ComKMeans, r.aucComKMeans,
                    r.melhoriaF1, r.melhoriaAUC);
        }
    }

    private static void exibirConclusoes(List<ResultadoComparacao> comparacoes) {
        System.out.println("\n╔════════════════════════════════════════════════════════════════╗");
        System.out.println("║                    ANÁLISE E CONCLUSÕES                        ║");
        System.out.println("╚════════════════════════════════════════════════════════════════╝\n");

        if (comparacoes.isEmpty()) return;

        // Tendência com aumento de dados
        System.out.println("1. IMPACTO DO AUMENTO DE DADOS:");
        System.out.println("   ─────────────────────────────");

        double mejoriaF1Original = comparacoes.get(0).f1SemKMeans;
        double mejoriaF1Duplo = comparacoes.get(1).f1SemKMeans;
        double mejoriaF13x = comparacoes.get(2).f1SemKMeans;

        System.out.printf("   • Original: F1 = %.4f%n", mejoriaF1Original);
        System.out.printf("   • 2x:       F1 = %.4f (%+.2f%%)%n", mejoriaF1Duplo,
                ((mejoriaF1Duplo - mejoriaF1Original) / mejoriaF1Original) * 100);
        System.out.printf("   • 3x:       F1 = %.4f (%+.2f%%)%n", mejoriaF13x,
                ((mejoriaF13x - mejoriaF1Original) / mejoriaF1Original) * 100);

        // KMeans effect
        System.out.println("\n2. EFETIVIDADE DO KMEANS:");
        System.out.println("   ───────────────────────");

        double mediaKMeansOriginal = comparacoes.get(0).melhoriaF1;
        double mediaKMeansDuplo = comparacoes.get(1).melhoriaF1;
        double mediaKMeans3x = comparacoes.get(2).melhoriaF1;

        System.out.printf("   • Original: %+.2f%% (F1)%n", mediaKMeansOriginal);
        System.out.printf("   • 2x:       %+.2f%% (F1)%n", mediaKMeansDuplo);
        System.out.printf("   • 3x:       %+.2f%% (F1)%n", mediaKMeans3x);

        if (mediaKMeansOriginal > 1) {
            System.out.println("\n   ✓ KMeans é EFETIVO em melhorar F1");
        } else {
            System.out.println("\n   ✗ KMeans NÃO melhora F1 significativamente");
        }

        // Tempo de treinamento
        System.out.println("\n3. CUSTO COMPUTACIONAL:");
        System.out.println("   ──────────────────────");

        double tempoAumentoOriginal = (double)comparacoes.get(0).tempoComKMeans / comparacoes.get(0).tempoSemKMeans;
        double tempoAumentoDuplo = (double)comparacoes.get(1).tempoComKMeans / comparacoes.get(1).tempoSemKMeans;
        double tempoAumento3x = (double)comparacoes.get(2).tempoComKMeans / comparacoes.get(2).tempoSemKMeans;

        System.out.printf("   • Original: KMeans é %.2fx mais lento%n", tempoAumentoOriginal);
        System.out.printf("   • 2x:       KMeans é %.2fx mais lento%n", tempoAumentoDuplo);
        System.out.printf("   • 3x:       KMeans é %.2fx mais lento%n", tempoAumento3x);

        // Recomendação
        System.out.println("\n4. RECOMENDAÇÃO:");
        System.out.println("   ───────────────");

        double ganhoMedioKMeans = (mediaKMeansOriginal + mediaKMeansDuplo + mediaKMeans3x) / 3;
        double tempoMedio = (tempoAumentoOriginal + tempoAumentoDuplo + tempoAumento3x) / 3;

        if (ganhoMedioKMeans > 2 && tempoMedio < 2) {
            System.out.println("   ✓ USAR KMeans: ganhos significativos com custo aceitável");
        } else if (ganhoMedioKMeans > 0) {
            System.out.println("   ~ KMeans PODE ser usado conforme a bolsa de custo computacional");
        } else {
            System.out.println("   ✗ NÃO usar KMeans: não compensa o custo computacional");
        }

        System.out.println("\n════════════════════════════════════════════════════════════════\n");
    }
}


