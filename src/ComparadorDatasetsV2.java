import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Compara os resultados do MLP com e sem KMeans em diferentes tamanhos de datasets
 * Simula aumento de dados duplicando o dataset original
 */
public class ComparadorDatasetsV2 {
i
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
        List<PontoDado> dataOriginal = CarregadorDados.loadRawFiles(List.of(
                dataDir.resolve("hungarian.data"),
                dataDir.resolve("switzerland.data"),
                dataDir.resolve("long-beach-va.data")
        ));
        ResultadoComparacao orig = testarDataset(dataOriginal, "Original", dataOriginal.size());
        comparacoes.add(orig);

        // Testar com 2x (duplicar dados)
        System.out.println("\n[2/3] Testando com dados 2x (duplicados)...");
        List<PontoDado> data2x = new ArrayList<>(dataOriginal);
        data2x.addAll(dataOriginal);
        ResultadoComparacao duplo = testarDataset(data2x, "2x", data2x.size());
        comparacoes.add(duplo);

        // Testar com 3x
        System.out.println("\n[3/3] Testando com dados 3x (duplicados)...");
        List<PontoDado> data3x = new ArrayList<>(data2x);
        data3x.addAll(dataOriginal);
        ResultadoComparacao triplo = testarDataset(data3x, "3x", data3x.size());
        comparacoes.add(triplo);

        // Exibir resumo comparativo
        exibirResumo(comparacoes);

        // Exibir conclusões
        exibirConclusoes(comparacoes);
    }

    private static ResultadoComparacao testarDataset(List<PontoDado> all, String tamanho, int totalRegistros) throws Exception {
        ResultadoComparacao resultado = new ResultadoComparacao(tamanho, totalRegistros);

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
                    "%7s | %9d | %.4f->%.4f | %.4f->%.4f | %+7.2f%% | %4d->%4d%n",
                    r.tamanho, r.registros,
                    r.f1SemKMeans, r.f1ComKMeans,
                    r.aucSemKMeans, r.aucComKMeans,
                    r.melhoriaF1,
                    r.tempoSemKMeans, r.tempoComKMeans);
        }

        System.out.println("\n\n┌────────────────────────────────────────────────────────────────┐");
        System.out.println("│  PERFORMANCE (F1 e AUC por dataset)                            │");
        System.out.println("└────────────────────────────────────────────────────────────────┘\n");

        System.out.println("Dataset  | Sem KMeans         | Com KMeans         | Melhoria");
        System.out.println("         |   F1      AUC      |   F1      AUC      |  F1   AUC");
        System.out.println("---------|--------------------|--------------------|----------");
        for (ResultadoComparacao r : comparacoes) {
            System.out.printf(Locale.US,
                    "%8s | %.4f   %.4f    | %.4f   %.4f    | %+.2f%% %+.2f%%%n",
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
        System.out.println("1. IMPACTO DO AUMENTO DE DADOS (SEM KMeans):");
        System.out.println("   ──────────────────────────────────────────");

        double f1Original = comparacoes.get(0).f1SemKMeans;
        double f1Duplo = comparacoes.get(1).f1SemKMeans;
        double f13x = comparacoes.get(2).f1SemKMeans;

        System.out.printf("   • Original: F1 = %.4f%n", f1Original);
        System.out.printf("   • 2x:       F1 = %.4f (%+.2f%%)%n", f1Duplo,
                ((f1Duplo - f1Original) / f1Original) * 100);
        System.out.printf("   • 3x:       F1 = %.4f (%+.2f%%)%n", f13x,
                ((f13x - f1Original) / f1Original) * 100);

        // KMeans effect
        System.out.println("\n2. EFETIVIDADE DO KMEANS:");
        System.out.println("   ───────────────────────");

        double melhoriaOriginal = comparacoes.get(0).melhoriaF1;
        double melhoriaDuplo = comparacoes.get(1).melhoriaF1;
        double melhoria3x = comparacoes.get(2).melhoriaF1;

        System.out.printf("   • Original: %+.2f%% (F1)%n", melhoriaOriginal);
        System.out.printf("   • 2x:       %+.2f%% (F1)%n", melhoriaDuplo);
        System.out.printf("   • 3x:       %+.2f%% (F1)%n", melhoria3x);

        if (melhoriaOriginal > 1) {
            System.out.println("\n   ✓ KMeans é EFETIVO em melhorar F1");
        } else {
            System.out.println("\n   ✗ KMeans NÃO melhora F1 significativamente");
        }

        // Tempo de treinamento
        System.out.println("\n3. CUSTO COMPUTACIONAL (RAZÃO COM/SEM KMeans):");
        System.out.println("   ──────────────────────────────────────────");

        double razaoOriginal = (double)comparacoes.get(0).tempoComKMeans / comparacoes.get(0).tempoSemKMeans;
        double razaoDuplo = (double)comparacoes.get(1).tempoComKMeans / comparacoes.get(1).tempoSemKMeans;
        double razao3x = (double)comparacoes.get(2).tempoComKMeans / comparacoes.get(2).tempoSemKMeans;

        System.out.printf("   • Original: KMeans é %.2fx mais lento ({}ms vs {}ms)%n",
                razaoOriginal, comparacoes.get(0).tempoComKMeans, comparacoes.get(0).tempoSemKMeans);
        System.out.printf("   • 2x:       KMeans é %.2fx mais lento ({}ms vs {}ms)%n",
                razaoDuplo, comparacoes.get(1).tempoComKMeans, comparacoes.get(1).tempoSemKMeans);
        System.out.printf("   • 3x:       KMeans é %.2fx mais lento ({}ms vs {}ms)%n",
                razao3x, comparacoes.get(2).tempoComKMeans, comparacoes.get(2).tempoSemKMeans);

        // Escalabilidade
        System.out.println("\n4. ESCALABILIDADE:");
        System.out.println("   ─────────────────");

        double escalSemKMeans2x = (double)comparacoes.get(1).tempoSemKMeans / comparacoes.get(0).tempoSemKMeans;
        double escalSemKMeans3x = (double)comparacoes.get(2).tempoSemKMeans / comparacoes.get(0).tempoSemKMeans;
        double escalComKMeans2x = (double)comparacoes.get(1).tempoComKMeans / comparacoes.get(0).tempoComKMeans;
        double escalComKMeans3x = (double)comparacoes.get(2).tempoComKMeans / comparacoes.get(0).tempoComKMeans;

        System.out.printf("   MLP Sem KMeans:  2x = %.2fx, 3x = %.2fx%n", escalSemKMeans2x, escalSemKMeans3x);
        System.out.printf("   MLP Com KMeans:  2x = %.2fx, 3x = %.2fx%n", escalComKMeans2x, escalComKMeans3x);

        // Recomendação
        System.out.println("\n5. RECOMENDAÇÃO:");
        System.out.println("   ───────────────");

        double ganhoMedioKMeans = (melhoriaOriginal + melhoriaDuplo + melhoria3x) / 3;
        double tempoMedio = (razaoOriginal + razaoDuplo + razao3x) / 3;

        if (ganhoMedioKMeans > 2 && tempoMedio < 2) {
            System.out.println("   ✓ USAR KMeans: ganhos significativos com custo aceitável");
        } else if (ganhoMedioKMeans > 0 && tempoMedio < 5) {
            System.out.println("   ✓ USAR KMeans: melhoria clara, apesar do custo computacional");
        } else {
            System.out.println("   ~ KMeans PODE ser usado conforme a importância da melhoria marginal");
        }

        System.out.println("\n════════════════════════════════════════════════════════════════\n");
    }
}

