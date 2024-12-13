# Detecção de anomalias usando Double Deep Q-Learning

Este repositório contém o Trabalho de Conclusão de Curso (TCC) de João Antonio Oldenburg, aluno do curso de Engenharia Elétrica no Instituto Federal Santa Catarina campus Itajaí. O projeto explora a aplicação de técnicas de Deep Reinforcement Learning, especificamente Double Deep Q-Learning, para a detecção de anomalias em séries temporais.

## Requisitos de Instalação

Para executar o projeto, siga os passos abaixo:

1. Clone este repositório em sua máquina local.
2. Crie um ambiente virtual Python.
3. Instale as dependências necessárias utilizando o comando:

   ```bash
   pip install -r requirements.txt

## Execução dos Códigos

O repositório contém scripts Python que implementam a detecção de anomalias em diferentes tipos de sinais:

- **Para sinais artificiais**, execute o script `2024_8_27__DQN_anomaly_detection_artificial_With_Anomaly.py`.
- **Para outros tipos de sinais**, utilize o script `2024_8_27__DQN_anomaly_detection.py`.

### Importante

Cada script contém informações detalhadas sobre os valores a serem utilizados para o verdadeiro negativo em diferentes sinais. Esses valores devem ser inseridos nas linhas 296-297 do código:

```python
elif (action == 0 and temporal_serie_treino['anomaly'].iloc[idx] == 0):
    reward = 'AQUI'
```

Os sinais disponíveis podem ser encontrados na pasta `data`, e seus rótulos de anomalias estão localizados em `labels/combined_windows.json`

## Documentação e Estilo de Código

O código segue as diretrizes da PEP 8, garantindo consistência e legibilidade. Além disso, a principal documentação do projeto está contida na monografia do TCC, que deve ser consultada para um entendimento mais profundo do contexto e das metodologias aplicadas.

## Banco de Dados Utilizado

Este projeto utiliza o banco de dados [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB), uma referência amplamente reconhecida para a avaliação de algoritmos de detecção de anomalias, a qual para maiores informações pode ser consultada. O NAB inclui uma coleção diversa de séries temporais que abrangem dados de diferentes domínios, como redes sociais, tráfego de internet, e sensores IoT. Sua estrutura foi projetada para testar a eficácia dos algoritmos de detecção de anomalias em cenários realistas, sendo essencial para validar a performance dos métodos propostos. 

## Modo de Avaliação

O modo de avaliação utilizado no projeto foi importado da pasta `evaluation`, sendo derivado do projeto [Orion](https://github.com/sintel-dev/Orion). Orion é uma ferramenta de código aberto desenvolvida pelo MIT, projetada para avaliar a performance de algoritmos de detecção de anomalias. Ela oferece uma infraestrutura robusta para testar e comparar diferentes técnicas de detecção em um ambiente controlado, facilitando a análise da eficácia dos métodos empregados.

## Agradecimentos

Gostaria de expressar minha gratidão especial à empresa [Murabei Data Science](https://www.murabei.com/) pelo apoio contínuo ao longo deste projeto. 
