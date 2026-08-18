# CENTRO FEDERAL DE EDUCAÇÃO TECNOLÓGICA DE MINAS GERAIS

## CURSO DE ENGENHARIA DE COMPUTAÇÃO

**IGOR FURTADO PEREIRA LOUREIRO**

# APLICAÇÃO DE CLUSTERIZAÇÃO COMO ETAPA DE PRÉ-PROCESSAMENTO EM REDES NEURAIS ARTIFICIAIS

Timóteo  
2025

---

**IGOR FURTADO PEREIRA LOUREIRO**

# APLICAÇÃO DE CLUSTERIZAÇÃO COMO ETAPA DE PRÉ-PROCESSAMENTO EM REDES NEURAIS ARTIFICIAIS

Trabalho de Conclusão de Curso apresentado ao Curso de Engenharia de Computação do Centro Federal de Educação Tecnológica de Minas Gerais, Campus Timóteo, como requisito parcial para a obtenção do título de Bacharel em Engenharia de Computação.  
**Orientador:** Douglas Nunes de Oliveira

Timóteo  
2025

---

**IGOR FURTADO PEREIRA LOUREIRO**

# APLICAÇÃO DE CLUSTERIZAÇÃO COMO ETAPA DE PRÉ-PROCESSAMENTO EM REDES NEURAIS ARTIFICIAIS

Trabalho de Conclusão de Curso apresentado ao Curso de Engenharia de Computação do Centro Federal de Educação Tecnológica de Minas Gerais, Campus Timóteo, como requisito parcial para a obtenção do grau de Bacharel em Engenharia de Computação.

**Aprovado em:** ______ de ______________ de 2025.

___________________________________________________  
**Prof. Me. Douglas Nunes de Oliveira**  
Orientador

___________________________________________________  
**Prof. (Titulação) [Nome do avaliador 2]**  
Professor convidado

___________________________________________________  
**Prof. (Titulação) [Nome do avaliador 3]**  
Professor convidado

Timóteo  
2025

---

# RESUMO

[Resumo a ser inserido na versão final do trabalho.]

**Palavras-chave:** [definir palavras-chave].

---

# ABSTRACT

[Abstract to be inserted in the final version of the work.]

**Keywords:** [define keywords].

---

# LISTA DE ILUSTRAÇÕES

- Figura 1 — Exemplo de RNA simples
- Figura 2 — Exemplo de MLP
- Figura 3 — Atribuição de centróide aleatório aos dados
- Figura 4 — K-means após as iterações

# LISTA DE TABELAS

- [A ser preenchida de acordo com as tabelas incluídas na versão final do trabalho.]

# LISTA DE ABREVIATURAS E SIGLAS

- ABNT — Associação Brasileira de Normas Técnicas
- MLP — Multilayer Perceptron
- RNA — Rede Neural Artificial

# SUMÁRIO

1. [Introdução](#1-introdução)  
   1.1. [Objetivos](#11-objetivos)
2. [Fundamentação Teórica](#2-fundamentação-teórica)  
   2.1. [Redes Neurais Artificiais](#21-redes-neurais-artificiais)  
   2.2. [Perceptron Multicamadas (MLP)](#22-perceptron-multicamadas-mlp)  
   2.3. [Clusterização](#23-clusterização)  
   2.4. [K-means](#24-k-means)  
   2.5. [Otimização de Modelos Preditivos](#25-otimização-de-modelos-preditivos)  
   2.6. [Aplicações Combinadas de Clusterização e Redes Neurais Artificiais](#26-aplicações-combinadas-de-clusterização-e-redes-neurais-artificiais)
3. [Procedimentos Metodológicos](#3-procedimentos-metodológicos)  
4. [Referências](#referências)

---

# 1 Introdução

O avanço das tecnologias digitais e do poder computacional nas últimas décadas ampliou a aplicação de técnicas de inteligência computacional a problemas cada vez mais complexos. Entre esses problemas, destaca-se a interpretação e a classificação de grandes volumes de dados com características não triviais, como aqueles utilizados em diagnósticos médicos automatizados. Nesse contexto, a Engenharia de Computação, por meio da aprendizagem de máquina, tem fornecido modelos preditivos capazes de identificar padrões relevantes nos dados e de contribuir para tomadas de decisão mais precisas, especialmente em sistemas de apoio ao diagnóstico.

Entre os métodos de aprendizado supervisionado, as Redes Neurais Artificiais (RNAs) destacam-se por sua capacidade de modelar relações não lineares e lidar com dados de elevada complexidade. Inspiradas no funcionamento do cérebro humano, essas redes aprendem representações internas dos dados e podem realizar classificações com alto grau de acurácia (HAYKIN, 2001). Entretanto, o desempenho desses modelos depende fortemente da qualidade, da estrutura e da organização dos dados de entrada.

Nesse cenário, as técnicas de clusterização, entendidas como métodos de agrupamento não supervisionado baseados em similaridade, surgem como uma etapa complementar relevante. Ao organizar previamente os dados em subconjuntos com características semelhantes, espera-se reduzir a variabilidade interna e oferecer ao classificador uma estrutura de entrada mais homogênea. Essa abordagem pode favorecer a extração de padrões mais robustos e, consequentemente, contribuir para o desempenho das RNAs, sobretudo em contextos com dados ruidosos ou desbalanceados (JAIN; MURTY; FLYNN, 1999).

Apesar desse potencial, permanece em aberto a questão sobre o quanto a clusterização efetivamente melhora o desempenho de redes neurais em problemas de classificação prática. Em muitos casos, modelos treinados diretamente sobre os dados originais podem apresentar desempenho semelhante ou até superior, a depender da complexidade do problema e da arquitetura adotada. Assim, a comparação entre essas abordagens mostra-se relevante tanto sob o ponto de vista técnico quanto sob o ponto de vista prático.

Dessa forma, este trabalho propõe a comparação entre duas abordagens computacionais para o diagnóstico automatizado de doenças cardíacas: uma baseada em Redes Neurais Artificiais com clusterização prévia dos dados e outra em que a rede opera diretamente sobre os dados originais. Como referência adicional, considera-se também o desempenho de um modelo clássico baseado em regressão logística e função discriminante, proposto por Detrano et al. (1989), que utiliza variáveis clínicas tradicionais. A investigação busca verificar se a clusterização agrega valor significativo ao desempenho do modelo ou se apenas acrescenta complexidade ao pipeline computacional.

## 1.1 Objetivos

### Objetivo geral

Avaliar o impacto da aplicação de técnicas de clusterização como etapa de pré-processamento sobre o desempenho de Redes Neurais Artificiais no diagnóstico de doenças cardíacas.

### Objetivos específicos

- implementar modelos de RNA treinados com dados originais e com dados previamente agrupados por técnicas de clusterização;
- comparar o desempenho dos modelos com base em métricas como acurácia e precisão;
- analisar os benefícios e as limitações da clusterização como etapa de pré-processamento na base de dados utilizada;
- comparar os resultados obtidos com um modelo clínico tradicional baseado em função discriminante.

# 2 Fundamentação Teórica

## 2.1 Redes Neurais Artificiais

As Redes Neurais Artificiais (RNAs) são modelos computacionais inspirados no funcionamento dos neurônios biológicos. Compostas por unidades chamadas neurônios artificiais, essas redes processam informações por meio de conexões ponderadas entre camadas de entrada, camadas ocultas e camada de saída (HAYKIN, 2001, p. 15). Cada neurônio calcula uma combinação linear de suas entradas e aplica uma função de ativação, como ReLU, sigmoide ou tangente hiperbólica, para introduzir não linearidade ao sistema (GOODFELLOW; BENGIO; COURVILLE, 2016, p. 168-174).

*Figura 1 — Exemplo de RNA simples.*

O processo de aprendizado das RNAs é realizado por meio do ajuste iterativo dos pesos das conexões, com o objetivo de minimizar uma função de erro. O algoritmo de retropropagação do erro (*backpropagation*), amplamente utilizado, aplica o método do gradiente descendente para efetuar esse ajuste (HAYKIN, 2001, p. 153-205).

Entre as principais vantagens das RNAs, destacam-se a capacidade de modelar relações não lineares complexas e a flexibilidade para diferentes tipos de tarefas. Em contrapartida, essas redes também apresentam limitações, como a necessidade de grandes volumes de dados para treinamento eficiente, a sensibilidade aos hiperparâmetros e a baixa interpretabilidade dos resultados (GOODFELLOW; BENGIO; COURVILLE, 2016, p. 22-24).

Tais características justificam a adoção de técnicas complementares, como clusterização e seleção de atributos, com o objetivo de melhorar a organização dos dados de entrada e aumentar a eficiência do processo de aprendizado supervisionado.

## 2.2 Perceptron Multicamadas (MLP)

O Perceptron Multicamadas (*Multilayer Perceptron* — MLP) é uma das arquiteturas mais tradicionais e amplamente utilizadas entre as Redes Neurais Artificiais. Trata-se de uma rede do tipo *feedforward*, isto é, a informação flui unidirecionalmente da camada de entrada para uma ou mais camadas ocultas e, por fim, para a camada de saída, sem a existência de conexões de realimentação (HAYKIN, 2001, p. 180).

Cada camada do MLP é composta por neurônios artificiais que realizam operações de soma ponderada das entradas, seguidas da aplicação de uma função de ativação não linear. Essa característica é fundamental para permitir que a rede aprenda relações complexas e não triviais entre os dados de entrada e os alvos desejados (GOODFELLOW; BENGIO; COURVILLE, 2016, p. 168-169).

*Figura 2 — Exemplo de MLP.*

O MLP constitui uma evolução direta do Perceptron simples, desenvolvido para superar suas principais limitações. Enquanto o Perceptron original é capaz de resolver apenas problemas linearmente separáveis, o MLP incorpora uma ou mais camadas ocultas e funções de ativação não lineares, o que lhe permite aprender relações não lineares complexas entre variáveis de entrada e de saída (HAYKIN, 2001, p. 180; GOODFELLOW; BENGIO; COURVILLE, 2016, p. 170).

Além disso, o MLP é capaz de aproximar qualquer função contínua, desde que disponha de neurônios e camadas ocultas suficientes (BISHOP, 2006, p. 226). Esse teorema da universalidade faz dessa arquitetura uma alternativa confiável para tarefas complexas de classificação e regressão.

Entre as principais vantagens do MLP, destacam-se:

- capacidade de modelar funções altamente não lineares, em razão das camadas ocultas e das funções de ativação apropriadas;
- flexibilidade arquitetural, permitindo adaptação a diferentes problemas e conjuntos de dados;
- capacidade de generalização, quando adequadamente regularizado e treinado com técnicas como validação cruzada;
- eficiência computacional, quando comparado a modelos mais complexos de aprendizado profundo.

Adicionalmente, o Perceptron Multicamadas é frequentemente empregado em cenários práticos que demandam análise de padrões complexos, tais como reconhecimento de fala, detecção de falhas, avaliação financeira e diagnóstico médico, inclusive em conjuntos de dados multivariados, como os utilizados neste estudo.

## 2.3 Clusterização

A clusterização é uma técnica de aprendizado não supervisionado cujo objetivo é agrupar elementos com base em suas similaridades, de modo que os itens de um mesmo grupo, ou *cluster*, sejam mais semelhantes entre si do que em relação aos elementos de outros grupos. Trata-se de uma abordagem amplamente utilizada em cenários nos quais os dados não possuem rótulos previamente definidos, sendo útil para revelar estruturas e padrões ocultos em conjuntos complexos (JAIN; MURTY; FLYNN, 1999, p. 265-266).

Entre os algoritmos mais comuns de clusterização, destacam-se:

- **K-means:** método particional que busca dividir os dados em *k* grupos, minimizando a distância intraclasse. É amplamente utilizado por sua simplicidade e eficiência, embora apresente limitações para detectar agrupamentos com formatos arbitrários ou tamanhos muito distintos (JAIN; MURTY; FLYNN, 1999, p. 275);
- **DBSCAN (*Density-Based Spatial Clustering of Applications with Noise*):** algoritmo baseado em densidade que identifica regiões densas como agrupamentos e trata pontos isolados como ruído. Sua principal vantagem consiste na capacidade de detectar *clusters* com formas irregulares e de lidar com *outliers* (XU; WUNSCH, 2005);
- **Clusterização hierárquica:** abordagem que constrói uma árvore de agrupamentos (*dendrograma*) a partir da fusão ou divisão progressiva dos dados. Esse método não exige a definição prévia do número de *clusters* e permite analisar a estrutura dos agrupamentos em diferentes níveis de granularidade (TAN; STEINBACH; KUMAR, 2019).

A escolha do algoritmo de clusterização depende da natureza dos dados e do objetivo analítico. No contexto deste trabalho, a clusterização é empregada como etapa de pré-processamento antes do treinamento do MLP. O objetivo é verificar se a segmentação dos dados em grupos internamente homogêneos favorece o aprendizado do modelo, com possível melhoria de acurácia e de capacidade de generalização.

## 2.4 K-means

O K-means é um dos algoritmos de clusterização não supervisionada mais populares e amplamente utilizados na análise de dados. Seu objetivo é particionar um conjunto de dados em *k* grupos mutuamente exclusivos, de forma que os elementos de cada grupo sejam mais semelhantes entre si do que em relação aos elementos dos demais grupos (AHMED; SERAJ; ISLAM, 2020, p. 2).

O funcionamento do K-means segue uma abordagem iterativa. Inicialmente, são escolhidos *k* centróides, que atuam como centros provisórios dos agrupamentos. Esses centróides podem ser selecionados aleatoriamente ou por métodos mais sofisticados, como o K-means++. Em seguida, cada ponto do conjunto de dados é atribuído ao centróide mais próximo, geralmente com base na distância euclidiana como medida de proximidade.

*Figura 3 — Atribuição de centróide aleatório aos dados.*

Após essa atribuição, os centróides são recalculados com base na média aritmética dos pontos associados a cada grupo. O processo de realocação dos pontos e de atualização dos centróides é repetido até que eles se estabilizem, isto é, até que não ocorram mudanças significativas em suas posições, ou até que um número máximo de iterações seja atingido.

*Figura 4 — K-means após as iterações.*

Apesar de sua simplicidade, o K-means é considerado um algoritmo eficiente e de baixo custo computacional, com complexidade da ordem de O(nkt), em que *n* representa o número de amostras, *k* o número de *clusters* e *t* o número de iterações (AHMED; SERAJ; ISLAM, 2020, p. 4). Essa eficiência torna o método particularmente adequado para aplicações em grandes conjuntos de dados, especialmente quando se busca uma solução prática e de rápida execução.

Entre as principais vantagens do K-means, destacam-se a simplicidade conceitual e de implementação, a eficiência computacional e a flexibilidade de adaptação a diferentes tipos de problemas. No contexto de Redes Neurais Artificiais, sua utilização como etapa de pré-processamento pode contribuir para a organização estrutural dos dados, agrupando instâncias semelhantes antes do treinamento da rede. Essa segmentação pode reduzir a variabilidade interna dos dados, facilitar o processo de aprendizado e potencialmente melhorar a capacidade de generalização da RNA (JING, 2022).

Adicionalmente, a aplicação prévia do K-means pode reduzir o impacto de ruídos e *outliers*, uma vez que os grupos tendem a representar padrões predominantes do conjunto de dados. Outra vantagem relevante é a possibilidade de representar os dados em termos dos centróides ou identificadores de *cluster*, o que pode resultar em entradas mais compactas e informativas para a rede neural. Em consequência, o treinamento pode se tornar mais rápido e menos suscetível ao sobreajuste. Por fim, o K-means é computacionalmente eficiente, o que o torna uma escolha prática para integração em *pipelines* neurais mais complexos (AHMED; SERAJ; ISLAM, 2020).

## 2.5 Otimização de Modelos Preditivos

A otimização de modelos preditivos é uma etapa fundamental no desenvolvimento de sistemas de aprendizado de máquina. Trata-se do processo de ajuste de parâmetros e hiperparâmetros com o objetivo de maximizar o desempenho de um modelo em tarefas como classificação, regressão ou agrupamento. Em redes neurais artificiais e algoritmos de clusterização, a escolha adequada desses elementos influencia diretamente a capacidade de generalização, a velocidade de convergência e a precisão dos resultados.

No caso do Perceptron Multicamadas, os principais hiperparâmetros a serem ajustados incluem o número de camadas ocultas, o número de neurônios por camada, a taxa de aprendizado (*learning rate*), o número de épocas, a função de ativação e o algoritmo de otimização, como SGD ou Adam. Esses elementos definem a forma como a rede aprende a partir dos dados e costumam ser escolhidos por meio de abordagens empíricas ou de algoritmos de busca automatizada (HAYKIN, 2001, p. 245-248).

Além disso, é importante aplicar técnicas de validação cruzada e utilizar conjuntos de teste independentes para garantir que os resultados obtidos não estejam sobreajustados aos dados de treinamento, o que comprometeria a capacidade de generalização do modelo em dados inéditos.

Neste trabalho, tanto os modelos de clusterização quanto o MLP serão otimizados com base em experimentação controlada e validação por métricas de desempenho. O objetivo é garantir que os resultados da comparação entre abordagens com e sem clusterização sejam confiáveis e representativos.

## 2.6 Aplicações Combinadas de Clusterização e Redes Neurais Artificiais

A integração entre técnicas de clusterização e Redes Neurais Artificiais tem se mostrado uma abordagem promissora para a construção de modelos preditivos mais robustos e precisos. Essa combinação busca explorar a capacidade dos algoritmos de agrupamento de organizar os dados em subconjuntos mais homogêneos, o que pode favorecer o processo de aprendizado supervisionado de redes neurais, como o MLP.

Ao aplicar a clusterização como etapa de pré-processamento, os dados de entrada são segmentados com base em padrões de similaridade, contribuindo para a redução de ruído, a mitigação de *outliers* e a melhoria da coesão interna dos grupos. Dessa forma, os modelos supervisionados podem concentrar seus esforços de aprendizado em estruturas mais bem definidas, com potencial para alcançar melhor desempenho preditivo.

Estudos recentes corroboram essa hipótese. Bodyanskiy et al. (2016) propuseram uma rede neural híbrida para o diagnóstico de artrite reativa, na qual a clusterização foi utilizada como filtro inicial antes da etapa de classificação. O modelo apresentou resultados superiores aos obtidos por redes convencionais, evidenciando ganhos na qualidade do processamento das informações.

De forma semelhante, Huang e Bhalla (2021) empregaram K-means como etapa preliminar em um sistema baseado em redes neurais convolucionais para segmentação de imagens médicas relacionadas ao diagnóstico da doença de Parkinson. Os autores observaram que a organização prévia dos dados por meio da clusterização possibilitou ganhos relevantes na acurácia da classificação.

Em um contexto mais próximo ao proposto neste trabalho, Alba et al. (2023) apresentaram um modelo híbrido para o diagnóstico de diferentes tipos de câncer com base em dados de expressão gênica, combinando *spectral clustering* e classificadores neurais. O estudo alcançou resultados expressivos em métricas como acurácia e sensibilidade, reforçando o potencial de abordagens híbridas.

Essas evidências reforçam a pertinência da investigação proposta, que consiste em avaliar se a aplicação de técnicas de clusterização sobre dados estruturados pode contribuir de maneira significativa para a melhoria do desempenho de redes neurais artificiais no diagnóstico de doenças cardíacas. A abordagem experimental adotada buscará comparar o desempenho do MLP treinado diretamente sobre os dados originais com sua versão alimentada por dados previamente agrupados por técnicas de clusterização.

# 3 Procedimentos Metodológicos

Este capítulo apresenta os procedimentos metodológicos adotados no desenvolvimento do trabalho, descrevendo o tipo de pesquisa realizada, a base de dados utilizada, as etapas de pré-processamento, a técnica de clusterização empregada, o modelo de Rede Neural Artificial adotado, os experimentos conduzidos, as métricas de avaliação consideradas e as ferramentas computacionais utilizadas.

## 3.1 Tipo de Pesquisa

A pesquisa desenvolvida neste trabalho possui natureza aplicada, uma vez que busca avaliar uma abordagem computacional voltada ao apoio ao diagnóstico de doenças cardíacas. Quanto à abordagem, trata-se de uma pesquisa quantitativa, pois a análise dos resultados é fundamentada em métricas numéricas de desempenho. Em relação aos procedimentos técnicos, caracteriza-se como experimental, visto que são comparados diferentes cenários de treinamento de uma Rede Neural Artificial, com e sem a aplicação de técnicas de clusterização.

## 3.2 Base de Dados

A base de dados utilizada neste trabalho é composta por um conjunto de dados público amplamente empregado em estudos sobre diagnóstico de doenças cardíacas, originalmente apresentado por Detrano et al. (1989). O conjunto contém registros de pacientes descritos por atributos clínicos e fisiológicos, além de um rótulo associado que indica a presença ou a ausência de doença cardíaca. Por se tratar de um conjunto de dados rotulado, ele é adequado à aplicação de técnicas de aprendizado supervisionado.

## 3.3 Pré-processamento dos Dados

Antes da aplicação dos algoritmos de clusterização e da Rede Neural Artificial, os dados passaram por uma etapa de pré-processamento. Essa etapa incluiu a análise e o tratamento de valores ausentes, bem como a normalização dos atributos numéricos, de modo a evitar que diferenças de escala influenciassem o processo de aprendizado. Após essas etapas, os dados foram organizados em conjuntos de atributos de entrada e rótulos de saída, garantindo sua adequação tanto para a clusterização quanto para o treinamento da rede neural.

## 3.4 Técnica de Clusterização

Como técnica de aprendizado não supervisionado, foi utilizado o algoritmo K-means, com o objetivo de agrupar os dados em subconjuntos mais homogêneos. A clusterização foi aplicada como etapa de pré-processamento, antecedendo o treinamento da Rede Neural Artificial. A partir dos *clusters* gerados, os dados foram reorganizados, permitindo analisar se a estruturação prévia dos dados influencia positivamente o desempenho do modelo supervisionado.

## 3.5 Modelo de Rede Neural Artificial

O modelo adotado para a tarefa de classificação é uma Rede Neural Artificial treinada de forma supervisionada. A rede foi configurada com uma arquitetura adequada ao problema proposto, utilizando funções de ativação não lineares e o algoritmo de retropropagação do erro para o ajuste dos pesos sinápticos. A mesma configuração da rede foi mantida em todos os experimentos, garantindo que as comparações realizadas fossem justas e consistentes.

## 3.6 Experimentos Realizados

Os experimentos foram conduzidos em dois cenários distintos. No primeiro, a Rede Neural Artificial foi treinada utilizando os dados originais, sem a aplicação de técnicas de clusterização. No segundo, a rede foi treinada com dados previamente agrupados pelo algoritmo K-means. Essa estratégia possibilitou a comparação direta entre os dois cenários, permitindo avaliar o impacto da clusterização no desempenho do modelo.

## 3.7 Métricas de Avaliação

A avaliação dos modelos foi realizada por meio das métricas de acurácia e precisão, amplamente utilizadas em problemas de classificação. A acurácia representa a proporção de classificações corretas em relação ao total de amostras analisadas, enquanto a precisão indica a proporção de previsões positivas corretas em relação ao total de previsões positivas realizadas. Os valores obtidos para cada métrica foram comparados entre os dois cenários experimentais, com o objetivo de verificar se a aplicação da clusterização contribui para a melhoria do desempenho da Rede Neural Artificial.

## 3.8 Ferramentas e Tecnologias Utilizadas

A implementação dos algoritmos e a realização dos experimentos foram feitas utilizando a linguagem de programação Java, escolhida por sua robustez, portabilidade e amplo suporte ao desenvolvimento de aplicações computacionais. O ambiente de desenvolvimento adotado permitiu a implementação tanto da etapa de clusterização quanto do treinamento e da avaliação da Rede Neural Artificial.

# Referências

AHMED, M.; SERAJ, R.; ISLAM, S. M. S. The K-means Algorithm: A Comprehensive Survey and Performance Evaluation. *Electronics*, v. 9, n. 8, p. 1295, 2020. Disponível em: https://www.mdpi.com/2079-9292/9/8/1295.

ALBA, E. et al. A Hybrid Model of Cancer Diseases Diagnosis Based on Gene Expression Data Using Spectral Clustering and Neural Networks. *Applied Sciences*, v. 13, n. 10, p. 6022, 2023. Disponível em: https://doi.org/10.3390/app13106022.

BISHOP, C. M. *Pattern Recognition and Machine Learning*. New York: Springer, 2006.

BODYANSKIY, Y. et al. Hybrid Clustering-Classification Neural Network in the Medical Diagnostics of Reactive Arthritis. *International Journal of Intelligent Systems and Applications*, v. 8, n. 8, p. 1-9, 2016. DOI: 10.5815/ijisa.2016.08.01.

DETRANO, R. et al. International Application of a New Probability Algorithm for the Diagnosis of Coronary Artery Disease. *The American Journal of Cardiology*, v. 64, n. 5, p. 304-310, 1989. Disponível em: https://doi.org/10.1016/0002-9149(89)90524-9. Acesso em: 16 jun. 2025.

GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. Cambridge: MIT Press, 2016. Disponível em: https://www.deeplearningbook.org.

HAYKIN, S. *Redes neurais: princípios e prática*. 2. ed. Porto Alegre: Bookman, 2001.

HUANG, Y.-P.; BHALLA, K. Wavelet K-Means Clustering and Fuzzy-Based Method for Segmenting MRI Images Depicting Parkinson’s Disease. *International Journal of Fuzzy Systems*, v. 23, p. 1600-1612, 2021. Disponível em: https://www.researchgate.net/publication/350369622_Wavelet_K-Means_Clustering_and_Fuzzy-Based_Method_for_Segmenting_MRI_Images_Depicting_Parkinson's_Disease.

JAIN, A. K.; MURTY, M. N.; FLYNN, P. J. Data Clustering: A Review. *ACM Computing Surveys*, New York, v. 31, n. 3, p. 264-323, 1999. Disponível em: https://doi.org/10.1145/331499.331504. Acesso em: 16 jun. 2025.

JING, H. Application of Improved K-Means Algorithm in Collaborative Recommendation System. *Journal of Applied Mathematics*, v. 2022, Art. ID 2213173, 2022. Disponível em: https://doi.org/10.1155/2022/2213173.

TAN, P.; STEINBACH, M.; KUMAR, V. *Introduction to Data Mining*. 2. ed. Boston: Pearson, 2019.

XU, R.; WUNSCH, D. Survey of Clustering Algorithms. *IEEE Transactions on Neural Networks*, v. 16, n. 3, p. 645-678, 2005. Disponível em: https://doi.org/10.1109/TNN.2005.845141.
