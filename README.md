# TCC - Comparação entre MLP tradicional e MLP com K-Means

Este projeto lê somente os arquivos brutos originais do dataset UCI Heart Disease em `Dados/`, trata valores faltantes e compara duas abordagens de diagnóstico binário:

- `MLP-Normal(13f)`: perceptron multicamadas treinado com os atributos clínicos clássicos
- `MLP+KMeans(k=2..10)`: o mesmo MLP com atributos adicionais gerados por agrupamento K-Means

Além das métricas preditivas, o projeto usa os arquivos em `Dados/costs/` para calcular custo estimado de aquisição de atributos e quantidade de testes com atraso.

## Estrutura do projeto

```text
Tcc/
├── Dados/
├── docs/
├── outputs/
├── src/
│   └── main/
│       └── java/
│           ├── Main.java
│           ├── CarregadorDados.java
│           ├── EngenhariaAtributosKMeans.java
│           ├── ExperimentoKMeans.java
│           ├── GeradorRelatorios.java
│           ├── ModeloCusto.java
│           ├── Modelos.java
│           ├── PontoDado.java
│           ├── PreProcessador.java
│           ├── ResultadoModelo.java
│           └── UtilMatriz.java
└── pom.xml
```

## Requisitos

- JDK 11 ou superior
- Maven 3.9+
- IntelliJ IDEA (opcional)

Recomendação prática: usar JDK 17 ou JDK 21.

## Executar com Maven

Na raiz do projeto:

```powershell
Set-Location "C:\Users\Windows 11\IdeaProjects\Tcc"
mvn clean compile
mvn exec:java
```

O `pom.xml` já está configurado para iniciar a classe `Main`.

## Executar no IntelliJ IDEA

1. Abra a pasta do projeto no IntelliJ.
2. Garanta que o projeto está usando um JDK válido, de preferência 17 ou 21.
3. Abra `src/main/java/Main.java`.
4. Execute a classe `Main`.

### Configuração importante

Na configuração de execução, use este diretório de trabalho:

```text
C:\Users\Windows 11\IdeaProjects\Tcc
```

Isso é necessário porque a aplicação lê os arquivos a partir da pasta `Dados/` na raiz do projeto.

## Executar com `javac` e `java`

Se quiser rodar sem Maven:

```powershell
Set-Location "C:\Users\Windows 11\IdeaProjects\Tcc"
New-Item -ItemType Directory -Force out | Out-Null
javac -d out src\main\java\*.java
java -cp out Main
```

## Principais arquivos

- `src/main/java/Main.java`: ponto de entrada da aplicação
- `src/main/java/CarregadorDados.java`: leitura e parsing dos datasets
- `src/main/java/PreProcessador.java`: divisão estratificada, imputação e preparação dos dados
- `src/main/java/Modelos.java`: implementação do MLP
- `src/main/java/EngenhariaAtributosKMeans.java`: geração de atributos com K-Means
- `src/main/java/ModeloCusto.java`: leitura dos arquivos de custo
- `src/main/java/GeradorRelatorios.java`: exportação de relatórios, CSVs e `outputs/resumo-execucao.md`

## Saídas geradas

Durante a execução, o projeto atualiza ou gera arquivos como:

- `outputs/performance_comparison.csv`
- `outputs/resumo-final.csv`
- `outputs/resumo-execucao.md`: resumo da execução, comparação dos modelos e explicação das nomenclaturas
- `docs/resumo-dados.md`
- `docs/resumo-resultados.md`

## Observações

- O alvo é binário: `num = 0` representa ausência de doença e `num > 0` representa presença de doença.
- O threshold de classificação é escolhido na validação e depois aplicado ao teste.
- O K-Means é ajustado apenas no conjunto de treino e são testados valores de `K` entre 2 e 10.
- O projeto depende da pasta `Dados/` presente na raiz do repositório.
- A comparação usa apenas `hungarian.data`, `switzerland.data` e `long-beach-va.data`; não são gerados nem usados dados sintéticos ou volumes 2x/3x.
- A execução gera apenas resumos técnicos e tabelas de resultados; não gera texto acadêmico.

## Solução de problemas

### IntelliJ tenta usar um JDK errado

Verifique estes pontos no IntelliJ:

- `File > Project Structure > Project SDK`
- `Run > Edit Configurations > JRE`
- `Settings > Build Tools > Maven > JDK for running Maven`

Todos devem apontar para um JDK funcional, preferencialmente 17 ou 21.

### Maven não encontra a classe principal

Confirme que o `pom.xml` está configurado com:

- `Main` como classe principal

e que o código está em:

- `src/main/java/`
