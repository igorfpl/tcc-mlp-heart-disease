class ExperimentoKMeans {
    public final DadosAtributosKMeans featureData;
    public final ResultadoModelo result;
    public final MlpModel model;

    public ExperimentoKMeans(DadosAtributosKMeans featureData, ResultadoModelo result, MlpModel model) {
        this.featureData = featureData;
        this.result = result;
        this.model = model;
    }
}


