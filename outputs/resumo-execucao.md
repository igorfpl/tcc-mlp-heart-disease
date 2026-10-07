# Resumo da execucao com dados originais

Este arquivo foi gerado automaticamente pelo programa. A comparacao usa somente os arquivos brutos `hungarian.data`, `switzerland.data` e `long-beach-va.data`; nenhum registro sintetico ou volume 2x/3x foi usado.

## O que foi feito

- Registros validos carregados: 617
- Divisao estratificada treino/validacao/teste: 432/93/92
- Foram comparados o MLP normal e nove configuracoes de MLP com K-Means (`k=2` ate `k=10`).
- O K-Means foi ajustado somente no treino; o limiar (`threshold`) foi escolhido na validacao e aplicado ao teste.

## Comparacao dos resultados no teste

| Modelo | Accuracy | Precision | Recall | Specificity | F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| MLP-Normal(13f) | 0.750 | 0.881 | 0.673 | 0.865 | 0.763 | 0.786 |
| MLP+KMeans(k=2) | 0.826 | 0.820 | 0.909 | 0.703 | 0.862 | 0.872 |
| MLP+KMeans(k=3) | 0.848 | 0.902 | 0.836 | 0.865 | 0.868 | 0.869 |
| MLP+KMeans(k=4) | 0.815 | 0.896 | 0.782 | 0.865 | 0.835 | 0.880 |
| MLP+KMeans(k=5) | 0.804 | 0.794 | 0.909 | 0.649 | 0.847 | 0.891 |
| MLP+KMeans(k=6) | 0.837 | 0.823 | 0.927 | 0.703 | 0.872 | 0.876 |
| MLP+KMeans(k=7) | 0.837 | 0.813 | 0.945 | 0.676 | 0.874 | 0.883 |
| MLP+KMeans(k=8) | 0.848 | 0.836 | 0.927 | 0.730 | 0.879 | 0.888 |
| MLP+KMeans(k=9) | 0.837 | 0.900 | 0.818 | 0.865 | 0.857 | 0.898 |
| MLP+KMeans(k=10) | 0.793 | 0.891 | 0.745 | 0.865 | 0.812 | 0.894 |

## Comparacao direta

O melhor K-Means abaixo e escolhido pelo maior F1 no conjunto de teste; ele e comparado ao MLP normal.

- MLP normal: `MLP-Normal(13f)` (F1 0.763, AUC 0.786)
- Melhor K-Means: `MLP+KMeans(k=8)` (F1 0.879, AUC 0.888)
- Diferenca de F1 (K-Means - normal): +0.116
- Diferenca de AUC (K-Means - normal): +0.102

## Pontos positivos e negativos

### MLP normal

- **Pontos positivos:** arquitetura mais simples, usa diretamente os 13 atributos clinicos originais, nao depende da escolha de `K` e evita o custo computacional adicional do agrupamento.
- **Desempenho observado:** Accuracy 0.750, F1 0.763, AUC 0.786, Recall 0.673 e Specificity 0.865.
- **Pontos negativos:** ficou abaixo do melhor K-Means em F1 (-0.116), AUC (-0.102) e Recall (-0.255).
- A simplicidade favorece interpretabilidade operacional, mas a representação sem atributos derivados pode capturar menos relações entre os pacientes.

### Melhor MLP+K-Means

- **Pontos positivos:** `MLP+KMeans(k=8)` obteve o maior F1 entre os valores testados, com Accuracy 0.848, F1 0.879, AUC 0.888 e Recall 0.927.
- Em relacao ao MLP normal, melhorou F1 em +0.116, AUC em +0.102 e Recall em +0.255.
- **Pontos negativos:** exige padronizacao, ajuste de centróides, escolha de `K` e criação de atributos adicionais, o que aumenta a complexidade do pipeline e pode reduzir a especificidade em algumas configurações.
- O custo estimado de atributos foi 323.97 e houve 9 teste(s) com atraso; esses valores devem ser considerados junto com o ganho de desempenho.
- O melhor `K` por F1 não é necessariamente o melhor por AUC ou silhouette; portanto, a escolha depende do objetivo prioritário.

### Síntese

- O MLP normal é a alternativa mais simples e direta.
- O MLP+K-Means apresenta melhor desempenho preditivo no melhor `K` observado, mas exige etapas extras e análise de trade-offs.

## Nomenclaturas e metricas

- `MLP`: Perceptron Multicamadas, a rede neural usada para classificar ausencia ou presenca de doenca.
- `MLP-Normal(13f)`: MLP com os 13 atributos clinicos originais, apos imputacao dos faltantes.
- `MLP+KMeans(k=N)`: mesmo tipo de MLP, acrescido de atributos derivados de agrupamento com `N` clusters.
- `K` ou `k`: quantidade de grupos (clusters) procurados pelo K-Means.
- `Accuracy`: proporcao total de classificacoes corretas.
- `Precision`: entre os casos previstos como positivos, proporcao realmente positiva.
- `Recall` (sensibilidade): entre os positivos reais, proporcao detectada pelo modelo.
- `Specificity` (especificidade): entre os negativos reais, proporcao identificada como negativa.
- `F1`: media harmonica entre precision e recall; equilibra as duas medidas.
- `AUC`: area sob a curva ROC; mede a capacidade de ordenar positivos acima de negativos.
- `Threshold`: probabilidade minima usada para transformar a previsao em classe 0 ou 1.
- `TP`/`TN`: verdadeiros positivos/negativos; `FP`/`FN`: falsos positivos/negativos.
- `feature_cost`: custo estimado para adquirir os atributos usados no modelo.
- `delayed_tests`: quantidade de testes usados que possuem atraso registrado na base de custos.

## Leitura dos resultados

Valores maiores de Accuracy, Precision, Recall, Specificity, F1 e AUC indicam melhor desempenho na respectiva perspectiva. As conclusoes devem priorizar o conjunto de teste, pois ele nao participa do ajuste dos parametros nem da escolha do threshold.

## Configuracoes K-Means

| K | Silhouette | Inertia | Iteracoes |
|---:|---:|---:|---:|
| 2 | 0.7702 | 5194.5721 | 2 |
| 3 | 0.1523 | 4410.4730 | 11 |
| 4 | 0.1617 | 4081.9649 | 20 |
| 5 | 0.1777 | 3728.3928 | 5 |
| 6 | 0.1771 | 3698.1113 | 7 |
| 7 | 0.1693 | 3108.3610 | 19 |
| 8 | 0.1746 | 3345.3350 | 13 |
| 9 | 0.1498 | 3073.8655 | 20 |
| 10 | 0.1722 | 2999.8120 | 9 |
