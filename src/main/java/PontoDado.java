public class PontoDado {
    public final int patientId;
    public final double[] features; // 13 atributos: age..thal
    public final int label; // 0 sem doenca, 1 com doenca
    public final String source;

    public PontoDado(int patientId, double[] features, int label, String source) {
        this.patientId = patientId;
        this.features = features;
        this.label = label;
        this.source = source;
    }
}


