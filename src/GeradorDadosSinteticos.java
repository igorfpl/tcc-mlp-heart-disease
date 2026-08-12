import java.util.*;

/**
 * Gerador de dados sintéticos mantendo as propriedades estatísticas dos dados reais.
 * Cria novos registros de PontoDado baseado em distribuições observadas.
 */
public class GeradorDadosSinteticos {
    private Random random;
    private Map<FeatureStats, double[]> featureStatsByLabelAndSource;
    private List<PontoDado> originalData;

    public GeradorDadosSinteticos(List<PontoDado> originalData, long seed) {
        this.originalData = originalData;
        this.random = new Random(seed);
        this.featureStatsByLabelAndSource = new HashMap<>();
        computeStatistics();
    }

    /**
     * Calcula média, desvio padrão e quartis para cada feature
     */
    private void computeStatistics() {
        Map<String, List<double[]>> groupedByLabelAndSource = new HashMap<>();

        for (PontoDado p : originalData) {
            String key = p.label + "_" + p.source;
            groupedByLabelAndSource.computeIfAbsent(key, k -> new ArrayList<>())
                    .add(p.features);
        }

        // Para cada grupo, calcula estatísticas
        for (Map.Entry<String, List<double[]>> entry : groupedByLabelAndSource.entrySet()) {
            List<double[]> records = entry.getValue();
            String[] parts = entry.getKey().split("_");
            int label = Integer.parseInt(parts[0]);
            String source = parts[1];

            for (int f = 0; f < 13; f++) {
                FeatureStats key = new FeatureStats(f, label, source);
                double[] stats = computeFeatureStats(records, f);
                featureStatsByLabelAndSource.put(key, stats);
            }
        }
    }

    /**
     * Retorna [media, desvio, min, q1, mediana, q3, max, contagem_nulos]
     */
    private double[] computeFeatureStats(List<double[]> records, int featureIdx) {
        List<Double> values = new ArrayList<>();
        int nullCount = 0;

        for (double[] features : records) {
            if (!Double.isNaN(features[featureIdx])) {
                values.add(features[featureIdx]);
            } else {
                nullCount++;
            }
        }

        if (values.isEmpty()) {
            return new double[]{0, 1, 0, 0, 0, 0, 0, 1.0};
        }

        Collections.sort(values);
        double mean = values.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double stddev = Math.sqrt(values.stream()
                .mapToDouble(v -> Math.pow(v - mean, 2))
                .average().orElse(0));

        return new double[]{
                mean,
                stddev > 0 ? stddev : 1,
                values.get(0),
                values.get((int)(values.size() * 0.25)),
                values.get(values.size() / 2),
                values.get((int)(values.size() * 0.75)),
                values.get(values.size() - 1),
                (double)nullCount / records.size()
        };
    }

    /**
     * Gera N novos registros sintéticos mantendo proporção de classes e fontes
     */
    public List<PontoDado> gerarDadosSinteticos(int quantidadeTotal) {
        List<PontoDado> sintéticos = new ArrayList<>();
        int patientIdStart = originalData.stream()
                .mapToInt(p -> p.patientId)
                .max()
                .orElse(0) + 1;

        // Calcula distribuição de label e source nos dados originais
        Map<String, Integer> labelSourceCounts = new HashMap<>();
        for (PontoDado p : originalData) {
            String key = p.label + "_" + p.source;
            labelSourceCounts.put(key, labelSourceCounts.getOrDefault(key, 0) + 1);
        }

        // Gera dados sintéticos mantendo proporção
        for (Map.Entry<String, Integer> entry : labelSourceCounts.entrySet()) {
            String[] parts = entry.getKey().split("_");
            int label = Integer.parseInt(parts[0]);
            String source = parts[1];
            int originalCount = entry.getValue();
            double proportion = (double) originalCount / originalData.size();
            int syntheticCountForGroup = (int) Math.round(proportion * quantidadeTotal);

            for (int i = 0; i < syntheticCountForGroup; i++) {
                double[] features = new double[13];
                for (int f = 0; f < 13; f++) {
                    FeatureStats statsKey = new FeatureStats(f, label, source);
                    double[] stats = featureStatsByLabelAndSource.get(statsKey);

                    // 20% chance de valor faltante, caso contrário amostra normal
                    if (random.nextDouble() < stats[7]) {
                        features[f] = Double.NaN;
                    } else {
                        // Amostragem normal truncada entre min e max
                        double value = stats[0] + random.nextGaussian() * stats[1];
                        value = Math.max(stats[2], Math.min(stats[6], value));
                        features[f] = value;
                    }
                }

                sintéticos.add(new PontoDado(
                        patientIdStart + sintéticos.size(),
                        features,
                        label,
                        source
                ));
            }
        }

        return sintéticos;
    }

    /**
     * Classe auxiliar para chave de mapa
     */
    private static class FeatureStats {
        int featureIdx;
        int label;
        String source;

        FeatureStats(int featureIdx, int label, String source) {
            this.featureIdx = featureIdx;
            this.label = label;
            this.source = source;
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof FeatureStats)) return false;
            FeatureStats other = (FeatureStats) o;
            return featureIdx == other.featureIdx && label == other.label && source.equals(other.source);
        }

        @Override
        public int hashCode() {
            return Objects.hash(featureIdx, label, source);
        }
    }
}

