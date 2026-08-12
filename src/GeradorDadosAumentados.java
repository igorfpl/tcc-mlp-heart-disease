import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Gera datasets aumentados com 2x e 3x o tamanho dos dados originais
 * Copia os dados originais múltiplas vezes com pequenas perturbações
 */
public class GeradorDadosAumentados {
    private static final int TOKENS_PER_RECORD = 76;
    private static final int[] FEATURE_INDEXES_1_BASED = {3, 4, 9, 10, 12, 16, 19, 32, 38, 40, 41, 44, 51};
    private static final int TARGET_INDEX_1_BASED = 58;

    public static void main(String[] args) throws IOException {
        Path root = Path.of(".").toAbsolutePath().normalize();
        Path dataDir = root.resolve("Dados");

        System.out.println("=== Gerador de Dados Aumentados ===\n");

        // Ler os arquivos originais
        String hungarian = Files.readString(dataDir.resolve("hungarian.data"), StandardCharsets.UTF_8);
        String switzerland = Files.readString(dataDir.resolve("switzerland.data"), StandardCharsets.UTF_8);
        String longBeachVa = Files.readString(dataDir.resolve("long-beach-va.data"), StandardCharsets.UTF_8);

        int hungarianCount = hungarian.trim().split("\\s+").length / TOKENS_PER_RECORD;
        int switzerlandCount = switzerland.trim().split("\\s+").length / TOKENS_PER_RECORD;
        int longBeachVaCount = longBeachVa.trim().split("\\s+").length / TOKENS_PER_RECORD;

        System.out.printf("Dados originais carregados: %d + %d + %d = %d registros%n%n",
                hungarianCount, switzerlandCount, longBeachVaCount,
                hungarianCount + switzerlandCount + longBeachVaCount);

        // Gerar dados aumentados
        generateAndSave(dataDir, "hungarian", hungarian, hungarianCount, 2);
        generateAndSave(dataDir, "switzerland", switzerland, switzerlandCount, 2);
        generateAndSave(dataDir, "long-beach-va", longBeachVa, longBeachVaCount, 2);

        generateAndSave(dataDir, "hungarian", hungarian, hungarianCount, 3);
        generateAndSave(dataDir, "switzerland", switzerland, switzerlandCount, 3);
        generateAndSave(dataDir, "long-beach-va", longBeachVa, longBeachVaCount, 3);

        System.out.println("\n✓ Datasets gerados com sucesso!");
        System.out.println("  - hungarian_2x.data, switzerland_2x.data, long-beach-va_2x.data");
        System.out.println("  - hungarian_3x.data, switzerland_3x.data, long-beach-va_3x.data");
    }

    private static void generateAndSave(Path dataDir, String source, String data, int originalCount, int multiplier) throws IOException {
        Random rng = new Random(42L + multiplier);
        StringBuilder augmented = new StringBuilder();
        String[] tokens = data.trim().split("\\s+");

        // Repetir os dados múltiplas vezes com perturbações
        for (int rep = 0; rep < multiplier; rep++) {
            for (int rec = 0; rec < originalCount; rec++) {
                for (int i = 0; i < TOKENS_PER_RECORD; i++) {
                    int idx = rec * TOKENS_PER_RECORD + i;
                    if (idx < tokens.length) {
                        String tok = tokens[idx];
                        String modified = tok;

                        // Adicionar pequenas perturbações em valores numéricos (exceto na primeira repetição)
                        if (rep > 0 && isNumericFeatureIndex(i)) {
                            try {
                                double val = Double.parseDouble(tok);
                                if (!Double.isNaN(val) && val != -9) {
                                    // Pequeno ruído: 1-3%
                                    double noise = val * (0.01 + 0.02 * rng.nextGaussian());
                                    modified = String.format("%.1f", val + noise);
                                }
                            } catch (NumberFormatException e) {
                                // Manter original se não for número
                            }
                        }
                        augmented.append(modified).append(" ");
                    }
                }
            }
        }

        String outputPath = dataDir.resolve(source + "_" + multiplier + "x.data").toString();
        Files.writeString(Path.of(outputPath), augmented.toString().trim(), StandardCharsets.UTF_8);
    }

    private static boolean isNumericFeatureIndex(int i) {
        // Verificar se é um índice de feature (1-based)
        for (int idx : FEATURE_INDEXES_1_BASED) {
            if (i == idx - 1) return true;
        }
        // Target
        return i == TARGET_INDEX_1_BASED - 1;
    }
}




