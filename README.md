# TCC - Comparacao entre MLP normal e MLP com K-Means

Este projeto le os arquivos **raw** em `Dados/` (`hungarian`, `switzerland`, `long-beach-va`), extrai os 14 atributos classicos do dataset UCI, trata valores faltantes e compara duas abordagens principais para diagnostico binario:

- `MLP-Normal(13f)` - Perceptron Multicamadas treinado sem normalizacao, sem padronizacao e sem clustering
- `MLP+KMeans(k=...)` - o mesmo MLP, mas com uma etapa adicional de clustering para gerar atributos derivados, testando varios valores de `K`

Tambem integra os arquivos de custo em `Dados/costs/` para relatar custo total estimado e numero de testes com atraso.

## Estrutura principal

- `src/Principal.java` - pipeline completo
- `src/CarregadorDados.java` - parser dos dados raw (76 atributos por registro)
- `src/PreProcessador.java` - split estratificado, imputacao por mediana, padronizacao
- `src/Modelos.java` - implementacao do MLP
- `src/EngenhariaAtributosKMeans.java` - K-Means e geracao de features derivadas para diferentes valores de `K`
- `src/ModeloCusto.java` - leitura dos arquivos de custo
- `src/GeradorRelatorios.java` - exportacao de relatorios e CSV de comparacao

## Executar (Java puro)

```powershell
Set-Location "C:\Users\Windows 11\IdeaProjects\Tcc"
$javac = "C:\Users\Windows 11\.jdks\openjdk-26.0.1\bin\javac.exe"
$java = "C:\Users\Windows 11\.jdks\openjdk-26.0.1\bin\java.exe"
New-Item -ItemType Directory -Force out | Out-Null
& $javac -d out src\*.java
& $java -cp out Principal
```

## Executar (Maven opcional)

```powershell
Set-Location "C:\Users\Windows 11\IdeaProjects\Tcc"
mvn -q -DskipTests compile
mvn -q exec:java
```

## Saidas geradas

- `outputs/performance_comparison.csv`
- `docs/resumo-dados.md`
- `docs/resumo-resultados.md`
- `docs/resumo-mlp.md`
- `docs/resumo-kmeans.md`
- `docs/comprovacao-mlp-funcionando.md`

## Observacoes

- O alvo e binario: `num = 0` (sem doenca) e `num > 0` (com doenca).
- O MLP normal nao recebe normalizacao nem K-Means; apenas a imputacao minima de faltantes dos arquivos raw (`-9`, `-9.`, `?`).
- O threshold de classificacao e escolhido na validacao e depois aplicado ao teste.
- O K-Means e ajustado apenas no treino e sao testados varios valores de `K` no intervalo de 2 a 6.

