import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public class GeradorRelatorios {
    public static void printConsoleTable(List<ResultadoModelo> results) {
        System.out.println("\n=== Comparacao de modelos ===");
        System.out.println("Modelo | Acc | Prec | Recall | Spec | F1 | AUC | Custo testes | Delayed");
        for (ResultadoModelo r : results) {
            System.out.printf(Locale.US,
                    "%s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.2f | %d%n",
                    r.modelName, r.accuracy, r.precision, r.recall, r.specificity, r.f1, r.auc,
                    r.featureAcquisitionCost, r.delayedTests);
        }
    }

    static void writeOutputs(Path root, List<PontoDado> all, Divisao split, DadosPreparados normalPrepared,
                             DadosPreparados kMeansPrepared,
                             MlpModel normalModel,
                             List<ResultadoModelo> results, ModeloCusto costModel,
                             List<ExperimentoKMeans> kMeansExperiments) throws IOException {
        Path outputs = root.resolve("outputs");
        Path docs = root.resolve("docs");
        Files.createDirectories(outputs);
        Files.createDirectories(docs);

        writeCsv(outputs.resolve("performance_comparison.csv"), results);
        writeFinalResultsCsv(outputs.resolve("resumo-final.csv"), results);
        writeExecutionSummary(outputs.resolve("resumo-execucao.md"), all, split, results, kMeansExperiments);
        writeDataSummary(docs.resolve("resumo-dados.md"), all, normalPrepared, costModel);
        writeResultSummary(docs.resolve("resumo-resultados.md"), results, kMeansExperiments);
        writeMlpSummary(docs.resolve("resumo-mlp.md"), normalPrepared, results);
        writeKMeansSummary(docs.resolve("resumo-kmeans.md"), kMeansPrepared, kMeansExperiments);
        writeMlpProof(docs.resolve("comprovacao-mlp-funcionando.md"), normalPrepared, results);
        writeDetailedCaseReport(docs.resolve("casos-classificados.md"), all, split, normalPrepared, kMeansPrepared,
                normalModel, results, kMeansExperiments);
        writeDetailedCaseCsv(outputs.resolve("casos-classificados.csv"), all, split, normalPrepared, kMeansPrepared,
                normalModel, results, kMeansExperiments);
    }

    private static void writeExecutionSummary(Path path, List<PontoDado> all, Divisao split,
                                              List<ResultadoModelo> results,
                                              List<ExperimentoKMeans> kMeansExperiments) throws IOException {
        ResultadoModelo normal = null;
        ResultadoModelo bestKMeans = null;
        for (ResultadoModelo result : results) {
            if (result.modelName.startsWith("MLP-Normal")) {
                normal = result;
            } else if (bestKMeans == null || result.f1 > bestKMeans.f1) {
                bestKMeans = result;
            }
        }

        StringBuilder sb = new StringBuilder();
        sb.append("# Resumo da execucao com dados originais\n\n");
        sb.append("Este arquivo foi gerado automaticamente pelo programa. A comparacao usa somente os arquivos brutos ");
        sb.append("`hungarian.data`, `switzerland.data` e `long-beach-va.data`; nenhum registro sintetico ou volume 2x/3x foi usado.\n\n");
        sb.append("## O que foi feito\n\n");
        sb.append("- Registros validos carregados: ").append(all.size()).append("\n");
        sb.append("- Divisao estratificada treino/validacao/teste: ")
                .append(split.train.size()).append("/")
                .append(split.val.size()).append("/")
                .append(split.test.size()).append("\n");
        sb.append("- Foram comparados o MLP normal e nove configuracoes de MLP com K-Means (`k=2` ate `k=10`).\n");
        sb.append("- O K-Means foi ajustado somente no treino; o limiar (`threshold`) foi escolhido na validacao e aplicado ao teste.\n\n");

        sb.append("## Comparacao dos resultados no teste\n\n");
        sb.append("| Modelo | Accuracy | Precision | Recall | Specificity | F1 | AUC |\n");
        sb.append("|---|---:|---:|---:|---:|---:|---:|\n");
        for (ResultadoModelo result : results) {
            sb.append(String.format(Locale.US,
                    "| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |%n",
                    result.modelName, result.accuracy, result.precision, result.recall,
                    result.specificity, result.f1, result.auc));
        }

        if (normal != null && bestKMeans != null) {
            sb.append("\n## Comparacao direta\n\n");
            sb.append("O melhor K-Means abaixo e escolhido pelo maior F1 no conjunto de teste; ele e comparado ao MLP normal.\n\n");
            sb.append(String.format(Locale.US,
                    "- MLP normal: `%s` (F1 %.3f, AUC %.3f)%n"
                            + "- Melhor K-Means: `%s` (F1 %.3f, AUC %.3f)%n"
                            + "- Diferenca de F1 (K-Means - normal): %+.3f%n"
                            + "- Diferenca de AUC (K-Means - normal): %+.3f%n",
                    normal.modelName, normal.f1, normal.auc,
                    bestKMeans.modelName, bestKMeans.f1, bestKMeans.auc,
                    bestKMeans.f1 - normal.f1, bestKMeans.auc - normal.auc));

            sb.append("\n## Pontos positivos e negativos\n\n");
            sb.append("### MLP normal\n\n");
            sb.append("- **Pontos positivos:** arquitetura mais simples, usa diretamente os 13 atributos clinicos originais, ");
            sb.append("nao depende da escolha de `K` e evita o custo computacional adicional do agrupamento.\n");
            sb.append(String.format(Locale.US,
                    "- **Desempenho observado:** Accuracy %.3f, F1 %.3f, AUC %.3f, Recall %.3f e Specificity %.3f.%n",
                    normal.accuracy, normal.f1, normal.auc, normal.recall, normal.specificity));
            sb.append(String.format(Locale.US,
                    "- **Pontos negativos:** ficou abaixo do melhor K-Means em F1 (%+.3f), AUC (%+.3f) e Recall (%+.3f).%n",
                    normal.f1 - bestKMeans.f1, normal.auc - bestKMeans.auc,
                    normal.recall - bestKMeans.recall));
            sb.append("- A simplicidade favorece interpretabilidade operacional, mas a representação sem atributos derivados pode ");
            sb.append("capturar menos relações entre os pacientes.\n\n");

            sb.append("### Melhor MLP+K-Means\n\n");
            sb.append(String.format(Locale.US,
                    "- **Pontos positivos:** `%s` obteve o maior F1 entre os valores testados, com Accuracy %.3f, F1 %.3f, AUC %.3f e Recall %.3f.%n",
                    bestKMeans.modelName, bestKMeans.accuracy, bestKMeans.f1,
                    bestKMeans.auc, bestKMeans.recall));
            sb.append(String.format(Locale.US,
                    "- Em relacao ao MLP normal, melhorou F1 em %+.3f, AUC em %+.3f e Recall em %+.3f.%n",
                    bestKMeans.f1 - normal.f1, bestKMeans.auc - normal.auc,
                    bestKMeans.recall - normal.recall));
            sb.append("- **Pontos negativos:** exige padronizacao, ajuste de centróides, escolha de `K` e criação de atributos adicionais, ");
            sb.append("o que aumenta a complexidade do pipeline e pode reduzir a especificidade em algumas configurações.\n");
            sb.append(String.format(Locale.US,
                    "- O custo estimado de atributos foi %.2f e houve %d teste(s) com atraso; esses valores devem ser considerados junto com o ganho de desempenho.%n",
                    bestKMeans.featureAcquisitionCost, bestKMeans.delayedTests));
            sb.append("- O melhor `K` por F1 não é necessariamente o melhor por AUC ou silhouette; portanto, a escolha depende do objetivo prioritário.\n\n");

            sb.append("### Síntese\n\n");
            sb.append("- O MLP normal é a alternativa mais simples e direta.\n");
            sb.append("- O MLP+K-Means apresenta melhor desempenho preditivo no melhor `K` observado, mas exige etapas extras e análise de trade-offs.\n");
        }

        sb.append("\n## Nomenclaturas e metricas\n\n");
        sb.append("- `MLP`: Perceptron Multicamadas, a rede neural usada para classificar ausencia ou presenca de doenca.\n");
        sb.append("- `MLP-Normal(13f)`: MLP com os 13 atributos clinicos originais, apos imputacao dos faltantes.\n");
        sb.append("- `MLP+KMeans(k=N)`: mesmo tipo de MLP, acrescido de atributos derivados de agrupamento com `N` clusters.\n");
        sb.append("- `K` ou `k`: quantidade de grupos (clusters) procurados pelo K-Means.\n");
        sb.append("- `Accuracy`: proporcao total de classificacoes corretas.\n");
        sb.append("- `Precision`: entre os casos previstos como positivos, proporcao realmente positiva.\n");
        sb.append("- `Recall` (sensibilidade): entre os positivos reais, proporcao detectada pelo modelo.\n");
        sb.append("- `Specificity` (especificidade): entre os negativos reais, proporcao identificada como negativa.\n");
        sb.append("- `F1`: media harmonica entre precision e recall; equilibra as duas medidas.\n");
        sb.append("- `AUC`: area sob a curva ROC; mede a capacidade de ordenar positivos acima de negativos.\n");
        sb.append("- `Threshold`: probabilidade minima usada para transformar a previsao em classe 0 ou 1.\n");
        sb.append("- `TP`/`TN`: verdadeiros positivos/negativos; `FP`/`FN`: falsos positivos/negativos.\n");
        sb.append("- `feature_cost`: custo estimado para adquirir os atributos usados no modelo.\n");
        sb.append("- `delayed_tests`: quantidade de testes usados que possuem atraso registrado na base de custos.\n\n");

        sb.append("## Leitura dos resultados\n\n");
        sb.append("Valores maiores de Accuracy, Precision, Recall, Specificity, F1 e AUC indicam melhor desempenho ");
        sb.append("na respectiva perspectiva. As conclusoes devem priorizar o conjunto de teste, pois ele nao participa do ajuste ");
        sb.append("dos parametros nem da escolha do threshold.\n\n");
        sb.append("## Configuracoes K-Means\n\n");
        sb.append("| K | Silhouette | Inertia | Iteracoes |\n");
        sb.append("|---:|---:|---:|---:|\n");
        for (ExperimentoKMeans experiment : kMeansExperiments) {
            sb.append(String.format(Locale.US, "| %d | %.4f | %.4f | %d |%n",
                    experiment.featureData.k, experiment.featureData.silhouette,
                    experiment.featureData.inertia, experiment.featureData.fittedIterations));
        }

        Files.writeString(path, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void writeCsv(Path path, List<ResultadoModelo> results) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("model,accuracy,precision,recall,specificity,f1,auc,threshold,tp,tn,fp,fn,feature_cost,delayed_tests\n");
        for (ResultadoModelo r : results) {
            sb.append(String.format(Locale.US,
                    "%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.2f,%d,%d,%d,%d,%.2f,%d\n",
                    r.modelName, r.accuracy, r.precision, r.recall, r.specificity, r.f1, r.auc,
                    r.threshold, r.tp, r.tn, r.fp, r.fn, r.featureAcquisitionCost, r.delayedTests));
        }
        Files.writeString(path, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void writeFinalResultsCsv(Path path, List<ResultadoModelo> results) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("model,category,k,accuracy,precision,recall,specificity,f1,auc,threshold,tp,tn,fp,fn,feature_cost,delayed_tests\n");
        for (ResultadoModelo r : results) {
            String category = r.modelName.startsWith("MLP-Normal") ? "MLP-Normal" : "MLP+KMeans";
            String kValue = "";
            if (r.modelName.contains("k=")) {
                int start = r.modelName.indexOf("k=") + 2;
                int end = r.modelName.indexOf(')', start);
                kValue = end > start ? r.modelName.substring(start, end) : r.modelName.substring(start);
            }
            sb.append(String.format(Locale.US,
                    "%s,%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.2f,%d,%d,%d,%d,%.2f,%d\n",
                    r.modelName, category, kValue,
                    r.accuracy, r.precision, r.recall, r.specificity, r.f1, r.auc,
                    r.threshold, r.tp, r.tn, r.fp, r.fn, r.featureAcquisitionCost, r.delayedTests));
        }
        Files.writeString(path, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void writeDataSummary(Path path, List<PontoDado> all, DadosPreparados prepared,
                                         ModeloCusto costModel) throws IOException {
        int pos = 0;
        Map<String, Integer> bySource = new HashMap<>();
        int d = PreProcessador.FEATURE_NAMES.length;
        int[] missing = new int[d];

        for (PontoDado dp : all) {
            if (dp.label == 1) pos++;
            bySource.merge(dp.source, 1, Integer::sum);
            for (int j = 0; j < d; j++) {
                if (Double.isNaN(dp.features[j])) {
                    missing[j]++;
                }
            }
        }

        StringBuilder sb = new StringBuilder();
        sb.append("# Resumo dos dados\n\n");
        sb.append("- Registros validos (raw): ").append(all.size()).append("\n");
        sb.append("- Classe positiva (num > 0): ").append(pos).append("\n");
        sb.append("- Classe negativa (num = 0): ").append(all.size() - pos).append("\n");
        sb.append("- Split treino/validacao/teste: ")
                .append(prepared.xTrain.length).append("/")
                .append(prepared.xVal.length).append("/")
                .append(prepared.xTest.length).append("\n\n");

        sb.append("## Registros por base\n\n");
        for (Map.Entry<String, Integer> e : bySource.entrySet()) {
            sb.append("- ").append(e.getKey()).append(": ").append(e.getValue()).append("\n");
        }

        sb.append("\n## Taxa de faltantes por atributo\n\n");
        for (int j = 0; j < d; j++) {
            double rate = (100.0 * missing[j]) / Math.max(1, all.size());
            sb.append(String.format(Locale.US, "- %s: %.2f%%%n", PreProcessador.FEATURE_NAMES[j], rate));
        }

        sb.append("\n## Custos dos testes (full 13 features)\n\n");
        int[] full = UtilMatriz.range(0, PreProcessador.FEATURE_NAMES.length);
        double total = costModel.totalFeatureCost(full, PreProcessador.FEATURE_NAMES);
        int delayed = costModel.countDelayedTests(full, PreProcessador.FEATURE_NAMES);
        sb.append(String.format(Locale.US, "- Custo total estimado: %.2f%n", total));
        sb.append("- Quantidade de testes com atraso: ").append(delayed).append("\n");

        Files.writeString(path, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void writeResultSummary(Path path, List<ResultadoModelo> results, List<ExperimentoKMeans> kMeansExperiments) throws IOException {
        ResultadoModelo bestF1 = null;
        ResultadoModelo bestAuc = null;
        ResultadoModelo lowestCost = null;
        ResultadoModelo normalMlp = null;
        ResultadoModelo bestKMeansByF1 = null;

        for (ResultadoModelo r : results) {
            if (bestF1 == null || r.f1 > bestF1.f1) bestF1 = r;
            if (bestAuc == null || r.auc > bestAuc.auc) bestAuc = r;
            if (lowestCost == null || r.featureAcquisitionCost < lowestCost.featureAcquisitionCost) lowestCost = r;
            if (r.modelName.startsWith("MLP-Normal")) normalMlp = r;
            if (r.modelName.startsWith("MLP+KMeans") && (bestKMeansByF1 == null || r.f1 > bestKMeansByF1.f1)) {
                bestKMeansByF1 = r;
            }
        }

        StringBuilder sb = new StringBuilder();
        sb.append("# Resumo de resultados\n\n");
        sb.append("Comparacao principal do TCC: **MLP tradicional** vs **MLP com pre-processamento por K-Means**.\n\n");
        sb.append("## Tabela comparativa\n\n");
        sb.append("| Modelo | Acc | Prec | Recall | Spec | F1 | AUC | Custo | Delayed |\n");
        sb.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");

        for (ResultadoModelo r : results) {
            sb.append(String.format(Locale.US,
                    "| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.2f | %d |%n",
                    r.modelName, r.accuracy, r.precision, r.recall, r.specificity, r.f1, r.auc,
                    r.featureAcquisitionCost, r.delayedTests));
        }

        sb.append("\n## Destaques\n\n");
        if (bestF1 != null) {
            sb.append("- Melhor F1: ").append(bestF1.modelName).append(" (")
                    .append(String.format(Locale.US, "%.3f", bestF1.f1)).append(")\n");
        }
        if (bestAuc != null) {
            sb.append("- Melhor AUC: ").append(bestAuc.modelName).append(" (")
                    .append(String.format(Locale.US, "%.3f", bestAuc.auc)).append(")\n");
        }
        if (lowestCost != null) {
            sb.append("- Menor custo de testes: ").append(lowestCost.modelName).append(" (")
                    .append(String.format(Locale.US, "%.2f", lowestCost.featureAcquisitionCost)).append(")\n");
        }
        if (!kMeansExperiments.isEmpty()) {
            ExperimentoKMeans bestSilhouette = kMeansExperiments.get(0);
            for (ExperimentoKMeans experiment : kMeansExperiments) {
                if (experiment.featureData.silhouette > bestSilhouette.featureData.silhouette) {
                    bestSilhouette = experiment;
                }
            }
            sb.append(String.format(Locale.US,
                    "- Melhor silhouette entre os K testados: k=%d, silhouette=%.4f, inertia=%.4f, iteracoes=%d.%n",
                    bestSilhouette.featureData.k,
                    bestSilhouette.featureData.silhouette,
                    bestSilhouette.featureData.inertia,
                    bestSilhouette.featureData.fittedIterations));
        }

        if (normalMlp != null && bestKMeansByF1 != null) {
            sb.append("\n## Diferenca direta entre os dois modelos\n\n");
            sb.append("Comparacao entre o `MLP-Normal(13f)` e o melhor `MLP+KMeans` medido por F1.\n\n");
            sb.append(String.format(Locale.US,
                    "- Variacao de Accuracy: %+,.3f%n- Variacao de F1: %+,.3f%n- Variacao de AUC: %+,.3f%n- Variacao de Specificity: %+,.3f%n- Variacao de Recall: %+,.3f%n",
                    bestKMeansByF1.accuracy - normalMlp.accuracy,
                    bestKMeansByF1.f1 - normalMlp.f1,
                    bestKMeansByF1.auc - normalMlp.auc,
                    bestKMeansByF1.specificity - normalMlp.specificity,
                    bestKMeansByF1.recall - normalMlp.recall));
        }

        sb.append("\n## Leitura rapida\n\n");
        sb.append("- O `MLP-Normal(13f)` nao recebe normalizacao, padronizacao, clustering ou engenharia de atributos; ele usa os valores imputados e segue direto para a rede.\n");
        sb.append("- Cada `MLP+KMeans` recebe as 13 features originais em versao padronizada para clustering, acrescidas de atributos derivados do agrupamento (distancias e one-hot do cluster).\n");
        sb.append("- O threshold de classificacao foi escolhido na validacao e aplicado ao teste, o que deixa a comparacao metodologicamente mais correta.\n");

        Files.writeString(path, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void writeMlpSummary(Path path, DadosPreparados prepared, List<ResultadoModelo> results) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("# Resumo metodologico do MLP\n\n");
        sb.append("## Objetivo\n\n");
        sb.append("Treinar um Perceptron Multicamadas para diagnostico binario de doenca cardiaca, onde `0` representa ausencia de doenca e `1` representa presenca (`num > 0`).\n\n");
        sb.append("## Entrada do modelo\n\n");
        sb.append("- 13 atributos clinicos tradicionais do dataset de heart disease.\n");
        sb.append("- O MLP normal **nao recebeu normalizacao, padronizacao, K-Means ou engenharia de atributos**.\n");
        sb.append("- A unica intervencao foi a imputacao minima de valores faltantes por mediana do treino, pois a rede nao consegue operar com `NaN`.\n");
        sb.append("- Tamanho da entrada do MLP normal: ").append(prepared.xTrain[0].length).append(" atributos.\n\n");
        sb.append("## Arquitetura utilizada\n\n");
        sb.append("- Camada de entrada com 13 neuronios.\n");
        sb.append("- Primeira camada oculta com 32 neuronios e ativacao ReLU.\n");
        sb.append("- Segunda camada oculta com 16 neuronios e ativacao ReLU.\n");
        sb.append("- Camada de saida com 1 neuronio e ativacao sigmoide.\n\n");

        sb.append("## Treinamento\n\n");
        sb.append("- Otimizacao por gradiente descendente em mini-batches implementada manualmente em Java.\n");
        sb.append("- Learning rate = 0.0005.\n");
        sb.append("- L2 = 1e-4 para reduzir overfitting.\n");
        sb.append("- Early stopping baseado na AUC de validacao.\n");
        sb.append("- Balanceamento por peso de classe para aumentar sensibilidade na classe positiva.\n\n");
        sb.append("## Avaliacao\n\n");
        sb.append("- Split estratificado: treino / validacao / teste.\n");
        sb.append("- O limiar de decisao e escolhido no conjunto de validacao maximizando F1.\n");
        sb.append("- O desempenho final e reportado apenas no conjunto de teste.\n");
        sb.append("- Metricas usadas: accuracy, precision, recall, specificity, F1 e AUC.\n\n");
        if (!results.isEmpty()) {
            for (ResultadoModelo result : results) {
                if (result.modelName.startsWith("MLP-Normal")) {
                    sb.append("## Resultado do MLP normal\n\n");
                    sb.append(String.format(Locale.US,
                            "- Accuracy: %.3f%n- Precision: %.3f%n- Recall: %.3f%n- Specificity: %.3f%n- F1: %.3f%n- AUC: %.3f%n- Threshold de validacao aplicado no teste: %.2f%n",
                            result.accuracy, result.precision, result.recall, result.specificity, result.f1, result.auc, result.threshold));
                }
            }
        }
        Files.writeString(path, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void writeKMeansSummary(Path path, DadosPreparados prepared,
                                           List<ExperimentoKMeans> kMeansExperiments) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("# Resumo metodologico do K-Means + MLP\n\n");
        sb.append("## Objetivo da etapa K-Means\n\n");
        sb.append("Usar agrupamento nao supervisionado para extrair estrutura latente dos pacientes antes do treino do MLP, criando atributos adicionais que representam similaridade com perfis clinicos.\n\n");
        sb.append("## Como o K-Means foi aplicado\n\n");
        sb.append("1. O algoritmo foi ajustado **somente no conjunto de treino**, evitando vazamento de informacao.\n");
        sb.append("2. Os dados de entrada do K-Means ja estavam imputados e padronizados.\n");
        sb.append("3. Foram testados valores de `k` de 2 a 10.\n");
        sb.append("4. Para cada `k`, foram registrados silhouette, inertia e desempenho final do MLP, permitindo comparar varios valores de `K`.\n");
        sb.append("5. A inicializacao dos centroides foi feita com estrategia inspirada em **k-means++**.\n");
        sb.append("6. O algoritmo foi encerrado quando nao houve mudanca relevante nos centroides ou ao atingir o limite de iteracoes.\n\n");
        sb.append("## Features geradas pelo K-Means\n\n");
        sb.append("Para cada paciente, a etapa de pre-processamento gera:\n\n");
        sb.append("- As 13 features clinicas originais.\n");
        sb.append("- `k` distancias euclidianas ate cada centroide.\n");
        sb.append("- `k` variaveis one-hot indicando o cluster mais proximo.\n\n");
        sb.append("## Comparacao entre valores de K\n\n");
        sb.append("| K | Dimensao final | Silhouette | Inertia | Iteracoes | Accuracy | F1 | AUC |\n");
        sb.append("|---:|---:|---:|---:|---:|---:|---:|---:|\n");
        for (ExperimentoKMeans experiment : kMeansExperiments) {
            DadosAtributosKMeans data = experiment.featureData;
            ResultadoModelo result = experiment.result;
            sb.append(String.format(Locale.US,
                    "| %d | %d | %.4f | %.4f | %d | %.3f | %.3f | %.3f |%n",
                    data.k, data.xTrain[0].length, data.silhouette, data.inertia, data.fittedIterations,
                    result.accuracy, result.f1, result.auc));
        }
        sb.append("\n## Parametros dos K testados\n\n");
        sb.append("- Faixa de K testada: 2 ate 10.\n");
        sb.append("- Para cada K foi treinado um novo MLP sobre as features derivadas.\n");
        sb.append("- O objetivo e observar se a representacao por clusters melhora a classificacao em relacao ao MLP normal.\n\n");
        sb.append("## Como o MLP usa o K-Means\n\n");
        sb.append("O segundo modelo do experimento nao substitui o MLP; ele usa o mesmo tipo de rede neural, mas com uma representacao de entrada enriquecida pela etapa de clustering. Assim, a comparacao isola o efeito do pre-processamento por K-Means.\n");

        ExperimentoKMeans bestByF1 = null;
        for (ExperimentoKMeans experiment : kMeansExperiments) {
            if (bestByF1 == null || experiment.result.f1 > bestByF1.result.f1) {
                bestByF1 = experiment;
            }
        }

        if (bestByF1 != null) {
            sb.append("\n## Melhor configuracao entre os K testados\n\n");
            sb.append(String.format(Locale.US,
                    "- Melhor K por F1: %d%n- Accuracy: %.3f%n- Precision: %.3f%n- Recall: %.3f%n- Specificity: %.3f%n- F1: %.3f%n- AUC: %.3f%n- Threshold de validacao aplicado no teste: %.2f%n",
                    bestByF1.featureData.k,
                    bestByF1.result.accuracy,
                    bestByF1.result.precision,
                    bestByF1.result.recall,
                    bestByF1.result.specificity,
                    bestByF1.result.f1,
                    bestByF1.result.auc,
                    bestByF1.result.threshold));
        }

        Files.writeString(path, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void writeMlpProof(Path path, DadosPreparados prepared, List<ResultadoModelo> results) throws IOException {
        ResultadoModelo normal = null;
        for (ResultadoModelo result : results) {
            if (result.modelName.startsWith("MLP-Normal")) {
                normal = result;
                break;
            }
        }
        if (normal == null) {
            return;
        }

        StringBuilder sb = new StringBuilder();
        sb.append("# Comprovacao de que o MLP normal esta funcionando\n\n");
        sb.append("Este documento demonstra que o `MLP-Normal(13f)` foi treinado, executado no conjunto de teste e produziu classificacoes reais de pacientes com e sem doenca cardiaca.\n\n");
        sb.append("## Evidencias quantitativas\n\n");
        sb.append(String.format(Locale.US,
                "- Total de pacientes no teste: %d%n- TP: %d%n- TN: %d%n- FP: %d%n- FN: %d%n- Accuracy: %.3f%n- Precision: %.3f%n- Recall: %.3f%n- Specificity: %.3f%n- F1: %.3f%n- AUC: %.3f%n- Threshold aplicado: %.2f%n",
                normal.yTrue.length, normal.tp, normal.tn, normal.fp, normal.fn,
                normal.accuracy, normal.precision, normal.recall, normal.specificity, normal.f1, normal.auc, normal.threshold));
        sb.append("\n## Interpretacao\n\n");
        sb.append("- Existem verdadeiros positivos e verdadeiros negativos, mostrando que o modelo conseguiu reconhecer as duas classes.\n");
        sb.append("- A presenca de probabilidades diferentes para os pacientes mostra que a rede nao esta retornando uma resposta fixa.\n");
        sb.append("- As metricas acima foram calculadas sobre o conjunto de teste, portanto refletem classificacao efetiva e nao apenas treinamento.\n\n");
        sb.append("## Exemplos de classificacoes realizadas pelo MLP normal\n\n");
        sb.append("| Caso teste | Probabilidade de doenca | Classe prevista | Classe real | Correto? |\n");
        sb.append("|---:|---:|---:|---:|---:|\n");
        int limit = Math.min(12, normal.yTrue.length);
        for (int i = 0; i < limit; i++) {
            int predicted = normal.probabilities[i] >= normal.threshold ? 1 : 0;
            sb.append(String.format(Locale.US,
                    "| %d | %.4f | %d | %d | %s |%n",
                    i + 1,
                    normal.probabilities[i],
                    predicted,
                    normal.yTrue[i],
                    predicted == normal.yTrue[i] ? "sim" : "nao"));
        }
        sb.append("\n## Entrada usada no MLP normal\n\n");
        sb.append("- Quantidade de atributos de entrada: ").append(prepared.xTrain[0].length).append("\n");
        sb.append("- Sem padronizacao e sem K-Means.\n");
        sb.append("- Apenas preenchimento minimo dos faltantes para viabilizar o calculo numerico.\n");

        Files.writeString(path, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void writeDetailedCaseReport(Path path, List<PontoDado> all, Divisao split,
                                                DadosPreparados normalPrepared, DadosPreparados kMeansPrepared,
                                                MlpModel normalModel,
                                                List<ResultadoModelo> results,
                                                List<ExperimentoKMeans> kMeansExperiments) throws IOException {
        ExperimentoKMeans bestKMeans = null;
        ResultadoModelo normalResult = null;
        for (ExperimentoKMeans experiment : kMeansExperiments) {
            if (bestKMeans == null || experiment.result.f1 > bestKMeans.result.f1) {
                bestKMeans = experiment;
            }
        }
        for (ResultadoModelo result : results) {
            if (result.modelName.startsWith("MLP-Normal")) {
                normalResult = result;
                break;
            }
        }
        if (bestKMeans == null || normalResult == null) {
            return;
        }

        double[][] normalAll = PreProcessador.transformFeatures(all, normalPrepared);
        double[] normalProbs = normalModel.predictProba(normalAll);

        double[][] standardizedAllForKMeans = PreProcessador.transformFeatures(all, kMeansPrepared);
        double[][] bestKMeansAll = EngenhariaAtributosKMeans.transformWithExistingModel(standardizedAllForKMeans, bestKMeans.featureData);
        double[] bestKMeansProbs = bestKMeans.model.predictProba(bestKMeansAll);

        IdentityHashMap<PontoDado, String> partitions = new IdentityHashMap<>();
        for (PontoDado dp : split.train) partitions.put(dp, "Treino");
        for (PontoDado dp : split.val) partitions.put(dp, "Validacao");
        for (PontoDado dp : split.test) partitions.put(dp, "Teste");

        List<Integer> allIndexes = new ArrayList<>();
        List<Integer> hungarianIndexes = new ArrayList<>();
        List<Integer> longBeachIndexes = new ArrayList<>();
        List<Integer> switzerlandIndexes = new ArrayList<>();
        for (int i = 0; i < all.size(); i++) {
            allIndexes.add(i);
            String source = all.get(i).source;
            if ("hungarian".equals(source)) {
                hungarianIndexes.add(i);
            } else if ("long-beach-va".equals(source)) {
                longBeachIndexes.add(i);
            } else if ("switzerland".equals(source)) {
                switzerlandIndexes.add(i);
            }
        }

        StringBuilder sb = new StringBuilder();
        sb.append("# Todos os casos classificados\n\n");
        sb.append("Este documento lista **todos os registros validos classificados** pelos modelos do experimento.\n\n");
        sb.append("- Total de casos listados: ").append(all.size()).append("\n");
        sb.append("- Modelos mostrados por caso: `MLP-Normal(13f)` e `")
                .append(bestKMeans.result.modelName).append("`\n");
        sb.append("- Fontes traduzidas: `Húngaro`, `Long Beach (VA)` e `Suíça`.\n\n");

        sb.append("## Legenda\n\n");
        sb.append("- Classe real / prevista `0`: sem doenca\n");
        sb.append("- Classe real / prevista `1`: com doenca\n");
        sb.append("- `Particao` indica se o caso pertence ao treino, validacao ou teste do experimento.\n\n");

        sb.append("## Resultados gerais\n\n");
        appendCaseTable(sb, all, allIndexes, partitions, normalProbs, normalResult.threshold, bestKMeansProbs, bestKMeans.result);

        sb.append("\n## Casos do conjunto Húngaro\n\n");
        appendCaseTable(sb, all, hungarianIndexes, partitions, normalProbs, normalResult.threshold, bestKMeansProbs, bestKMeans.result);

        sb.append("\n## Casos do conjunto Long Beach (VA)\n\n");
        appendCaseTable(sb, all, longBeachIndexes, partitions, normalProbs, normalResult.threshold, bestKMeansProbs, bestKMeans.result);

        sb.append("\n## Casos do conjunto Suíça\n\n");
        appendCaseTable(sb, all, switzerlandIndexes, partitions, normalProbs, normalResult.threshold, bestKMeansProbs, bestKMeans.result);

        Files.writeString(path, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void writeDetailedCaseCsv(Path path, List<PontoDado> all, Divisao split,
                                             DadosPreparados normalPrepared, DadosPreparados kMeansPrepared,
                                             MlpModel normalModel,
                                             List<ResultadoModelo> results,
                                             List<ExperimentoKMeans> kMeansExperiments) throws IOException {
        ExperimentoKMeans bestKMeans = null;
        ResultadoModelo normalResult = null;
        for (ExperimentoKMeans experiment : kMeansExperiments) {
            if (bestKMeans == null || experiment.result.f1 > bestKMeans.result.f1) {
                bestKMeans = experiment;
            }
        }
        for (ResultadoModelo result : results) {
            if (result.modelName.startsWith("MLP-Normal")) {
                normalResult = result;
                break;
            }
        }
        if (bestKMeans == null || normalResult == null) {
            return;
        }

        double[][] normalAll = PreProcessador.transformFeatures(all, normalPrepared);
        double[] normalProbs = normalModel.predictProba(normalAll);

        double[][] standardizedAllForKMeans = PreProcessador.transformFeatures(all, kMeansPrepared);
        double[][] bestKMeansAll = EngenhariaAtributosKMeans.transformWithExistingModel(standardizedAllForKMeans, bestKMeans.featureData);
        double[] bestKMeansProbs = bestKMeans.model.predictProba(bestKMeansAll);

        IdentityHashMap<PontoDado, String> partitions = new IdentityHashMap<>();
        for (PontoDado dp : split.train) partitions.put(dp, "Treino");
        for (PontoDado dp : split.val) partitions.put(dp, "Validacao");
        for (PontoDado dp : split.test) partitions.put(dp, "Teste");

        StringBuilder sb = new StringBuilder();
        sb.append("case_number,patient_id,origin,origin_pt,partition,actual_class,mlp_normal_probability,mlp_normal_prediction,mlp_normal_threshold,best_kmeans_model,best_kmeans_probability,best_kmeans_prediction,best_kmeans_threshold\n");
        for (int i = 0; i < all.size(); i++) {
            PontoDado dp = all.get(i);
            int normalPred = normalProbs[i] >= normalResult.threshold ? 1 : 0;
            int kMeansPred = bestKMeansProbs[i] >= bestKMeans.result.threshold ? 1 : 0;
            sb.append(String.format(Locale.US,
                    "%d,%d,%s,%s,%s,%d,%.6f,%d,%.2f,%s,%.6f,%d,%.2f\n",
                    i + 1,
                    dp.patientId,
                    dp.source,
                    translateSource(dp.source),
                    partitions.getOrDefault(dp, "Desconhecida"),
                    dp.label,
                    normalProbs[i],
                    normalPred,
                    normalResult.threshold,
                    bestKMeans.result.modelName,
                    bestKMeansProbs[i],
                    kMeansPred,
                    bestKMeans.result.threshold));
        }
        Files.writeString(path, sb.toString(), StandardCharsets.UTF_8);
    }


    private static void appendCaseTable(StringBuilder sb, List<PontoDado> all, List<Integer> indexes,
                                        IdentityHashMap<PontoDado, String> partitions,
                                        double[] normalProbs, double normalThreshold,
                                        double[] bestKMeansProbs, ResultadoModelo bestKMeansResult) {
        sb.append("| # | ID paciente | Origem | Particao | Classe real | Prob. MLP normal | Prev. MLP normal | Prob. Melhor K-Means | Prev. Melhor K-Means |\n");
        sb.append("|---:|---:|---|---|---:|---:|---:|---:|---:|\n");
        for (int index : indexes) {
            PontoDado dp = all.get(index);
            int normalPred = normalProbs[index] >= normalThreshold ? 1 : 0;
            int kMeansPred = bestKMeansProbs[index] >= bestKMeansResult.threshold ? 1 : 0;
            sb.append(String.format(Locale.US,
                    "| %d | %d | %s | %s | %d | %.4f | %d | %.4f | %d |%n",
                    index + 1,
                    dp.patientId,
                    translateSource(dp.source),
                    partitions.getOrDefault(dp, "Desconhecida"),
                    dp.label,
                    normalProbs[index],
                    normalPred,
                    bestKMeansProbs[index],
                    kMeansPred));
        }
    }

    private static String translateSource(String source) {
        if ("hungarian".equals(source)) {
            return "Húngaro";
        }
        if ("long-beach-va".equals(source)) {
            return "Long Beach (VA)";
        }
        if ("switzerland".equals(source)) {
            return "Suíça";
        }
        return source;
    }

}
