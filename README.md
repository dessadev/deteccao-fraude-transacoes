# Detecção de fraude em transações com classe rara

Modelo para identificar transações fraudulentas quando a fraude representa **menos de 1%** da base, com o ponto de corte escolhido pelo **custo do erro**.

> ⚠️ Os dados são **sintéticos**, gerados em [`src/gerar_dados.py`](src/gerar_dados.py). Nenhum dado real de cliente ou instituição é usado.

## Problema

Em fraude, a acurácia engana: um modelo que diz sempre *não é fraude* acerta mais de 99% e não pega nenhuma fraude. O desafio é equilibrar dois erros com custos diferentes:

| Erro | O que significa | Custo |
|---|---|---|
| Falso negativo | fraude que passa | perda financeira |
| Falso positivo | cliente bom bloqueado | atrito e risco de perder o cliente |

## Abordagem

1. **EDA** da taxa de fraude por hora, canal e dispositivo
2. **Validação out of time**: treino nos 4 primeiros meses e teste nos 2 últimos
3. **Comparação de modelos**: regressão logística (com e sem `class_weight`), SMOTE, Random Forest e Gradient Boosting
4. **Métricas para classe rara**: PR-AUC, ROC-AUC, KS, Gini, recall e precision
5. **Threshold por custo**: falso negativo custa o valor da transação, falso positivo custa R$ 15
6. **Interpretação** com importância por permutação

## Resultados (teste out of time)

| Modelo | ROC-AUC | Gini | KS | PR-AUC |
|---|---|---|---|---|
| Logística sem balancear | 0,888 | 0,776 | 0,665 | 0,250 |
| Logística + class_weight | 0,901 | 0,803 | 0,676 | 0,212 |
| Logística + SMOTE | 0,900 | 0,799 | 0,679 | 0,213 |
| Random Forest + class_weight | 0,905 | 0,810 | 0,679 | 0,299 |
| **Gradient Boosting** | **0,904** | **0,809** | **0,669** | **0,310** |

Com o Gradient Boosting e o threshold escolhido pelo custo (0,07):

- **Custo total 63,7% menor** que sem modelo
- Recall de 49% e precision de 22,5%
- 1,67% dos clientes bons bloqueados

## Aprendizados

- Com classe rara, a **PR-AUC** é a métrica mais honesta, porque não é inflada pelos negativos.
- O threshold padrão de 0,5 é ruim: o ponto ótimo saiu muito abaixo, porque deixar passar uma fraude custa mais que revisar uma transação boa.
- `class_weight` e SMOTE tiveram efeito parecido. O ganho maior veio de um modelo que capta interações entre variáveis.

## Limitações e próximos passos

- Em dados reais, o fraudador muda de tática (*concept drift*): é preciso monitorar o modelo e retreinar.
- Rótulos de fraude chegam com atraso, como em contestações, o que afeta a janela de treino.
- Próximos passos: variáveis de histórico por cliente em janelas de tempo, calibração das probabilidades e monitoramento de drift.

## Como rodar

```bash
pip install -r requirements.txt
jupyter notebook deteccao_fraude.ipynb
```

---
Feito por [Andressa Prudente](https://github.com/dessadev)
