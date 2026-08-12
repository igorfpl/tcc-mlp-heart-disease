import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class ModeloCusto {
    private final Map<String, Double> baseCost;
    private final Map<String, ExpensePair> expense;
    private final Map<String, String> group;
    private final Set<String> delayed;

    private ModeloCusto(Map<String, Double> baseCost,
                      Map<String, ExpensePair> expense,
                      Map<String, String> group,
                      Set<String> delayed) {
        this.baseCost = baseCost;
        this.expense = expense;
        this.group = group;
        this.delayed = delayed;
    }

    public static ModeloCusto load(Path costsDir) throws IOException {
        Map<String, Double> baseCost = parseCostFile(costsDir.resolve("heart-disease.cost"));
        Map<String, ExpensePair> expense = parseExpenseFile(costsDir.resolve("heart-disease.expense"));
        Map<String, String> group = parseGroupFile(costsDir.resolve("heart-disease.group"));
        Set<String> delayed = parseDelayFile(costsDir.resolve("heart-disease.delay"));
        return new ModeloCusto(baseCost, expense, group, delayed);
    }

    public double totalFeatureCost(int[] featureIndexes, String[] featureNames) {
        Set<String> openedGroups = new HashSet<>();
        double total = 0.0;

        for (int idx : featureIndexes) {
            String name = featureNames[idx];
            ExpensePair e = expense.get(name);
            String g = group.get(name);

            if (e != null && g != null) {
                if (openedGroups.add(g)) {
                    total += e.full;
                } else {
                    total += e.discount;
                }
            } else if (e != null) {
                total += e.full;
            } else {
                total += baseCost.getOrDefault(name, 0.0);
            }
        }
        return total;
    }

    public int countDelayedTests(int[] featureIndexes, String[] featureNames) {
        int count = 0;
        for (int idx : featureIndexes) {
            if (delayed.contains(featureNames[idx])) {
                count++;
            }
        }
        return count;
    }

    private static Map<String, Double> parseCostFile(Path path) throws IOException {
        Map<String, Double> map = new HashMap<>();
        for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
            String cleaned = cleanLine(line);
            if (cleaned.isEmpty()) continue;
            String[] parts = cleaned.split(":");
            if (parts.length < 2) continue;
            map.put(parts[0].trim(), Double.parseDouble(parts[1].trim()));
        }
        return map;
    }

    private static Set<String> parseDelayFile(Path path) throws IOException {
        Set<String> delayed = new HashSet<>();
        for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
            String cleaned = cleanLine(line);
            if (cleaned.isEmpty()) continue;
            String[] parts = cleaned.split(":");
            if (parts.length < 2) continue;
            String test = parts[0].trim();
            String v = trimSentenceDot(parts[1].trim()).toLowerCase();
            if (v.contains("delayed")) {
                delayed.add(test);
            }
        }
        return delayed;
    }

    private static Map<String, ExpensePair> parseExpenseFile(Path path) throws IOException {
        Map<String, ExpensePair> map = new HashMap<>();
        for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
            String cleaned = cleanLine(line);
            if (cleaned.isEmpty()) continue;
            String[] parts = cleaned.split(":");
            if (parts.length < 2) continue;
            String test = parts[0].trim();
            String[] costs = parts[1].split(",");
            if (costs.length < 2) continue;
            double full = Double.parseDouble(costs[0].trim());
            double discount = Double.parseDouble(costs[1].trim());
            map.put(test, new ExpensePair(full, discount));
        }
        return map;
    }

    private static Map<String, String> parseGroupFile(Path path) throws IOException {
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        Map<String, String> map = new HashMap<>();
        for (String line : lines) {
            String cleaned = cleanLine(line);
            if (!cleaned.contains(":")) continue;
            String[] parts = cleaned.split(":");
            if (parts.length < 2) continue;
            map.put(parts[0].trim(), trimSentenceDot(parts[1].trim()));
        }
        return map;
    }

    private static String cleanLine(String line) {
        return line.replace("\t", " ").trim();
    }

    private static String trimSentenceDot(String v) {
        String out = v.trim();
        while (out.endsWith(".")) {
            out = out.substring(0, out.length() - 1).trim();
        }
        return out;
    }

    private static class ExpensePair {
        final double full;
        final double discount;

        ExpensePair(double full, double discount) {
            this.full = full;
            this.discount = discount;
        }
    }
}



