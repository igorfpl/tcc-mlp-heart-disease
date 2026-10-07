import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CarregadorDados {
    private static final int TOKENS_PER_RECORD = 76;
    // 14 atributos usados historicamente: os primeiros 13 sao features, ultimo e o alvo num
    private static final int[] FEATURE_INDEXES_1_BASED = {3, 4, 9, 10, 12, 16, 19, 32, 38, 40, 41, 44, 51};
    private static final int TARGET_INDEX_1_BASED = 58;

    public static List<PontoDado> loadRawFiles(List<Path> files) throws IOException {
        List<PontoDado> all = new ArrayList<>();
        for (Path path : files) {
            all.addAll(loadRawFile(path));
        }
        return all;
    }

    private static List<PontoDado> loadRawFile(Path filePath) throws IOException {
        String source = filePath.getFileName().toString().replace(".data", "");
        String content = Files.readString(filePath, StandardCharsets.UTF_8);
        String[] rawTokens = content.split("\\s+");
        List<String> tokens = new ArrayList<>();
        for (String t : rawTokens) {
            if (!t.isBlank()) {
                tokens.add(t.trim());
            }
        }

        int fullRecords = tokens.size() / TOKENS_PER_RECORD;
        List<PontoDado> data = new ArrayList<>();

        for (int r = 0; r < fullRecords; r++) {
            int base = r * TOKENS_PER_RECORD;
            int patientId = parsePatientId(tokens.get(base));
            double[] features = new double[FEATURE_INDEXES_1_BASED.length];
            Arrays.fill(features, Double.NaN);

            for (int i = 0; i < FEATURE_INDEXES_1_BASED.length; i++) {
                String tok = tokens.get(base + FEATURE_INDEXES_1_BASED[i] - 1);
                features[i] = parseMaybeMissing(tok);
            }

            String targetTok = tokens.get(base + TARGET_INDEX_1_BASED - 1);
            double rawTarget = parseMaybeMissing(targetTok);
            if (Double.isNaN(rawTarget)) {
                continue;
            }
            int label = rawTarget > 0.0 ? 1 : 0;
            data.add(new PontoDado(patientId, features, label, source));
        }
        return data;
    }

    private static int parsePatientId(String token) {
        try {
            return (int) Math.round(Double.parseDouble(token.trim()));
        } catch (NumberFormatException e) {
            return -1;
        }
    }

    private static double parseMaybeMissing(String token) {
        String t = token.trim();
        if (t.equals("?") || t.equals("name")) {
            return Double.NaN;
        }
        // Nos arquivos raw, faltantes aparecem como -9 ou -9.
        if (t.startsWith("-9")) {
            return Double.NaN;
        }
        try {
            return Double.parseDouble(t);
        } catch (NumberFormatException e) {
            return Double.NaN;
        }
    }
}


