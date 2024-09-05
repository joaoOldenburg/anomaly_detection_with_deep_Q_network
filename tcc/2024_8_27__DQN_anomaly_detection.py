"""Rede DDQN para detecção de anomalias."""

import pandas as pd
import random
import numpy as np
import tensorflow as tf
import plotly.express as px
import plotly.graph_objs as go
from keras import Sequential
from arch import arch_model
from keras.layers import InputLayer
from keras.layers import Dense
from tqdm import tqdm
from sklearn.metrics import f1_score
from packages import template_murabei
from sklearn.preprocessing import MinMaxScaler
from tcc.evaluation.contextual import contextual_f1_score
from tcc.evaluation.utils import from_list_points_timestamps


# 0.05 para verdadeiro negativo
# temporal_serie = pd.read_csv(
#     "data/realTweets/Twitter_volume_AMZN.csv")
# print(f'Twitter_volume_AMZN.csv : {temporal_serie.shape}')

# anomaly_points = [
#         [
#             "2015-03-05 03:22:53.000000",
#             "2015-03-06 12:12:53.000000"
#         ],
#         [
#             "2015-03-11 04:32:53.000000",
#             "2015-03-12 13:22:53.000000"
#         ],
#         [
#             "2015-04-01 05:32:53.000000",
#             "2015-04-02 14:22:53.000000"
#         ],
#         [
#             "2015-04-07 12:27:53.000000",
#             "2015-04-08 21:17:53.000000"
#         ]
#     ]


#  0.3 para verdadeiro negativo
# temporal_serie = pd.read_csv(
#     "data/realAWSCloudwatch/ec2_disk_write_bytes_c0d644.csv")
# print(f'data/realAWSCloudwatch/ec2_disk_write_bytes_c0d644.csv":
# {temporal_serie.shape}')

# anomaly_points = [
#         [
#             "2014-04-08 19:55:00.000000",
#             "2014-04-09 07:05:00.000000"
#         ],
#         [
#             "2014-04-10 09:00:00.000000",
#             "2014-04-10 20:10:00.000000"
#         ],
#         [
#             "2014-04-12 21:25:00.000000",
#             "2014-04-13 08:35:00.000000"
#         ]
#     ]

# 3.6 para verdadeiro negativo
# temporal_serie = pd.read_csv(
#     "data/realAdExchange/exchange-4_cpc_results.csv")
# print(f'realAdExchange/exchange-4_cpc_results.csv": {temporal_serie.shape}')

# anomaly_points = [
#         [
#             "2011-07-15 06:15:01.000000",
#             "2011-07-17 12:15:01.000000"
#         ],
#         [
#             "2011-08-01 07:15:01.000000",
#             "2011-08-03 15:15:01.000000"
#         ],
#         [
#             "2011-08-22 05:15:01.000000",
#             "2011-08-24 11:15:01.000000"
#         ]
#     ]

# 4 para verdadeiro negativo
# temporal_serie = pd.read_csv(
#     "data/realTraffic/speed_7578.csv")
# print(f'realTraffic/speed_7578.csv": {temporal_serie.shape}')

# anomaly_points =  [
#         [
#             "2015-09-11 15:34:00.000000",
#             "2015-09-11 17:54:00.000000"
#         ],
#         [
#             "2015-09-15 13:26:00.000000",
#             "2015-09-15 15:54:00.000000"
#         ],
#         [
#             "2015-09-16 13:04:00.000000",
#             "2015-09-16 15:20:00.000000"
#         ],
#         [
#             "2015-09-16 16:00:00.000000",
#             "2015-09-16 18:20:00.000000"
#         ]
#     ]


temporal_serie['timestamp'] = pd.to_datetime(temporal_serie['timestamp'])

temporal_serie['anomaly'] = 0
for start, end in anomaly_points:
    temporal_serie.loc[((temporal_serie['timestamp'] >= start) &
                        (temporal_serie['timestamp'] <= end)), 'anomaly'] = 1

np.random.seed(42)


def calculate_conditional_std(series):
    """Calcula a volatilidade condicional utilizando um modelo GARCH(1,1).

    A média da diferença precisa ser zero para que a variância condicional
    funcione corretamente. A diferença é a subtração de um número pelo seu
    sucessor; teoricamente, essa diferença no tempo não deve apresentar
    tendências de crescimento ou queda.

    Args:
        series (pd.Series): Série temporal para a qual a volatilidade
        condicional será calculada.

    Returns:
        pd.Series: Volatilidade condicional ajustada pelo modelo GARCH(1,1).
    """
    model = arch_model(series, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp='off')
    return model_fit.conditional_volatility


# diferença para calculo de desvio padrão condicional
temporal_serie['diff'] = temporal_serie['value'].diff()
temporal_serie.loc[0, 'diff'] = 0

#############################
# Desvio padrão condicional #
#############################

temporal_serie['cond_std'] = calculate_conditional_std(temporal_serie['diff'])

#################################
# Adicionando priemira derivada #
#################################

t_derivada = range(0, len(temporal_serie))
y_derivada = temporal_serie['value'].values

# Calcular a primeira derivada
temporal_serie['dy_dt'] = np.gradient(y_derivada, t_derivada)

##########################
# Normalizando os dados #
#########################

data = temporal_serie[['cond_std', 'dy_dt', 'value']]
scaler = MinMaxScaler()
dados_normalizados = scaler.fit_transform(data)

dados_normalizados = pd.DataFrame(
    dados_normalizados, columns=['cond_std', 'dy_dt', 'value'])
temporal_serie[['cond_std', 'dy_dt', 'value']] = dados_normalizados


def create_model(input_shape, possible_actions):
    """Cria a rede neural sequencial para problemas de aprendizado por reforço.

    Args:
        input_shape (int): Dimensão da entrada do modelo.
        possible_actions (int): Número de ações possíveis na saída do modelo.

    Returns:
        tf.keras.Model: Modelo compilado pronto para treinamento.
    """
    model = Sequential()
    model.add(Dense(64, input_dim=input_shape, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(possible_actions, activation='linear'))
    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    )
    return model


def update_target_model():
    """Adiciona os pesos da rede principal na rede alvo."""
    target_model.set_weights(model.get_weights())


def select_action(state, exploration_rate):
    """Seleciona uma ação com base na exploração ou na predição dos valores.

    Args:
        state (np.array): O estado atual do ambiente.
        exploration_rate (float): A taxa de exploração, determinando
                                  a probabilidade
                                  de selecionar uma ação aleatória.

    Returns:
        int: A ação selecionada.
    """

    if np.random.rand() <= exploration_rate:
        # Escolha aleatória (exploração)
        return random.randrange(possible_actions)
    else:
        # Predição dos valores Q
        q_values = model.predict(state, verbose=0)
        # Escolha da ação com o maior valor Q (exploração)
        return np.argmax(q_values[0])


def training(minibatch):
    """Treina o modelo de rede neural usando amostras do minibatch.

    Args:
        minibatch (list): Lista de tlupas (estado, ação, recompensa,
                                           próximo estado).

    Returns:
        None
    """
    if len(memory) < batch_size:
        return

    for state, action, reward, next_state in minibatch:
        # Predição do Q-value atual com a rede principal
        target = model.predict(state, verbose=0)
        # Predição do Q-value do próximo estado com a rede principal
        next_action = np.argmax(model.predict(next_state, verbose=0)[0])
        # Predição do Q-value do próximo estado com a rede alvo
        t = target_model.predict(next_state, verbose=0)[0]
        # Atualização do Q-value usando a rede principal para selecionar a
        # ação e a rede alvo para avaliação
        target[0][action] = reward + gamma * t[next_action]
        # Treinamento da rede principal
        model.fit(state, target, epochs=1, verbose=0)

# Inicialização dos parâmetros
window_size = round(len(temporal_serie)*0.10)
state_size = 3 * window_size
n_iteracoes = 180
gamma = 0.9
possible_actions = 2
exploration_rate = 1
exploration_decay = 0.80
exploration_min = 0.05
replay_memory_size = 10000
batch_size = 32
number_batch_per_training = 2
number_update_per_batch = 3
update_target_freq = 20

# Inicialização das redes
model = create_model(state_size, possible_actions)
target_model = create_model(state_size, possible_actions)
update_target_model()

# Verificando rede
model.summary()
target_model.summary()

temporal_serie_treino = temporal_serie.loc[:len(temporal_serie)*0.70]

memory = []
for i in tqdm(range(n_iteracoes)):
    # Amostragem de estados e próximos estados
    random_indices = random.sample(
        range(window_size,len(temporal_serie_treino)-1), 300)

    # Preparando estados
    for idx in tqdm(random_indices):
        state = temporal_serie_treino\
            .loc[idx-window_size+1:idx, ['cond_std', 'dy_dt', 'value']]\
            .values.flatten().reshape(1, -1)
        next_state = temporal_serie_treino\
            .loc[idx-window_size+2:idx+1, ['cond_std', 'dy_dt', 'value']]\
            .values.flatten().reshape(1, -1)

        # Selecionando a ação
        action = select_action(state, exploration_rate)

        # Verificando recompensas
        if (action == 1 and temporal_serie_treino['anomaly'].iloc[idx] == 1):
            reward = 10
        elif (action == 1 and temporal_serie_treino['anomaly'].iloc[idx] == 0):
            reward = -1
        elif (action == 0 and temporal_serie_treino['anomaly'].iloc[idx] == 1):
            reward = -10
        elif (action == 0 and temporal_serie_treino['anomaly'].iloc[idx] == 0):
            reward = 4

        # Mantendo memória de replay no tamanho desejado
        if(len(memory) > replay_memory_size):
            del memory[0]

        memory.append((state, action, reward, next_state))

    # Separando batches para treinamento
    for batch in tqdm(range(number_batch_per_training)):
        minibatch = random.sample(memory, batch_size)
        for update in tqdm(range(number_update_per_batch)):
            training(minibatch)

    # Atualização da rede alvo
    if i % update_target_freq == 0:
        update_target_model()

    # Ajuste da taxa de exploração
    if exploration_rate > exploration_min:
        exploration_rate *= exploration_decay
        exploration_rate = max(exploration_min, exploration_rate)


################################################
######### TESTE DO TREINAMENTO DA REDE #########
################################################

def predict_full_series(temporal_serie, window_size):
    """Prediz ações para uma série temporal, detectando anomalias
           com base no modelo de aprendizado por reforço.

    Args:
        temporal_serie (pd.DataFrame): Série temporal contendo as
        colunas 'cond_std', 'dy_dt', e 'value'.
        window_size (int): Tamanho da janela de tempo para formar o estado.

    Returns:
        tuple: Uma tupla contendo a lista de predições, os valores corretos
        e os índices de anomalias.
    """
    predictions = []
    linha_correta = []
    indices_anomolos = []

    for idx in tqdm(range(window_size, len(temporal_serie))):
        state = temporal_serie.\
            loc[idx-window_size+1:idx, ['cond_std', 'dy_dt', 'value']]\
            .values.flatten().reshape(1, -1)

        action = np.argmax(model.predict(state, verbose=0)[0])
        predictions.append(action)
        linha_correta.append(temporal_serie['value'].iloc[idx])

        if action == 1:
            indices_anomolos.append(idx)

    return predictions, linha_correta, indices_anomolos

predictions_treino, valores_df_prediciton, anomalias_previstas = (
    predict_full_series(temporal_serie_treino, window_size))
real_anomaly = temporal_serie_treino['anomaly']\
    .loc[window_size:len(temporal_serie_treino)]
df_real_linha = temporal_serie_treino['value']\
    .loc[window_size:len(temporal_serie_treino)]

# Confere se os index entre predição e série temporal estão corretos
comparacao = (valores_df_prediciton == df_real_linha).all()
print('Os dados estão no mesmo index:', comparacao)

# Verifica o F1-Score padrão
dnq_f1 = f1_score(real_anomaly, predictions_treino)
print(f'Rede F1 Score : {dnq_f1}')

# Prepara um describe() das predições
media = np.mean(predictions_treino)
desvio_padrao = np.std(predictions_treino)
minimo = np.min(predictions_treino)
maximo = np.max(predictions_treino)
quartis = np.percentile(predictions_treino, [25, 50, 75])
soma = np.sum(predictions_treino)

# Mostra um describe() das predições
print("Soma:", soma)
print("Média:", media)
print("Desvio padrão:", desvio_padrao)
print("Mínimo:", minimo)
print("Máximo:", maximo)
print("Primeiro Quartil:", quartis[0])
print("Mediana:", quartis[1])
print("Terceiro Quartil:", quartis[2])

#################################
# evaluation MIT - usado no tcc #
#################################

if len(anomalias_previstas) != 0:
    anomalias_previstas_tupla = from_list_points_timestamps(anomalias_previstas)

    # Tuplas de das anomalias reais
    indices_anomolos_corretos = []
    for idx in range(window_size, len(temporal_serie_treino)):
        if temporal_serie_treino['anomaly'].iloc[idx] == 1:
                # Adiciona o índice se a ação for igual a 1
                indices_anomolos_corretos.append(idx)
    anomalias_reais_tupla_teste = from_list_points_timestamps(
        indices_anomolos_corretos)

    # F1 score MIT
    data = pd.DataFrame(
        {'timestamp': range(window_size, len(temporal_serie_treino))})
    start = window_size
    end = len(temporal_serie_treino) - 1
    f1 = contextual_f1_score(anomalias_reais_tupla_teste,
                             anomalias_previstas_tupla, data, start, end,
                             weighted=True)
    print(f'Rede F1 Score MIT: {f1}')
else:
    print("Não houveram anomalias previstas")


#################################
# Prevendo outros dados - teste #
#################################

# 50% pra frente - 40% de teste e 10% de window size
temporal_serie_teste = temporal_serie.loc[len(temporal_serie)*0.50:]\
    .reset_index(drop=True)
predictions_teste, valores_df_prediciton, anomalias_previstas = (
    predict_full_series(temporal_serie_teste, window_size))
real_anomaly = temporal_serie_teste['anomaly']\
    .loc[window_size:len(temporal_serie_teste)]
df_real_linha = temporal_serie_teste['value']\
    .loc[window_size:len(temporal_serie_teste)]


# Confere se os index entre predição e série temporal estão corretos
comparacao = (valores_df_prediciton == df_real_linha).all()
print('Os dados estão no mesmo index:', comparacao)

# F1-Score padrão
dnq_f1 = f1_score(real_anomaly, predictions_teste)
print(f'Rede F1 Score : {dnq_f1}')

# Prepara um describe() das predições
media = np.mean(predictions_treino)
desvio_padrao = np.std(predictions_treino)
minimo = np.min(predictions_treino)
maximo = np.max(predictions_treino)
quartis = np.percentile(predictions_treino, [25, 50, 75])
soma = np.sum(predictions_treino)

# Mostra um describe() das predições
print("Soma:", soma)
print("Média:", media)
print("Desvio padrão:", desvio_padrao)
print("Mínimo:", minimo)
print("Máximo:", maximo)
print("Primeiro Quartil:", quartis[0])
print("Mediana:", quartis[1])
print("Terceiro Quartil:", quartis[2])

#################################
# evaluation MIT - usado no tcc #
#################################

# F1-Score a ser considerado para fins de apresentação resultados é apenas esse.
if len(anomalias_previstas) != 0:
    anomalias_previstas_tupla = from_list_points_timestamps(
        anomalias_previstas)

    # Tuplas de das anomalias reais
    indices_anomolos_corretos = []
    for idx in range(window_size, len(temporal_serie_teste)):
        if temporal_serie_teste['anomaly'].iloc[idx] == 1:
            # Adiciona o índice se a ação for igual a 1
            indices_anomolos_corretos.append(idx)
    anomalias_reais_tupla = from_list_points_timestamps(
        indices_anomolos_corretos)

    # F1 score MIT
    data = pd.DataFrame(
        {'timestamp': range(window_size, len(temporal_serie_teste))})
    start = window_size
    end = len(temporal_serie_teste) - 1
    f1 = contextual_f1_score(anomalias_reais_tupla, anomalias_previstas_tupla,
                             data, start, end, weighted=True)
    print(f'Rede F1 Score MIT: {f1}')
else:
    print("Não houveram anomalias previstas")

############################
# Modelos de gráfico teste #
############################

# Coloca as predições do treino e as predições do teste na mesma lista
predictions = predictions_treino + predictions_teste

trace1 = go.Scatter(
    x=np.arange(len(temporal_serie.loc[window_size:len(temporal_serie)])),
    y=temporal_serie['value'].loc[window_size:len(temporal_serie)],
    mode='lines',
    name='value',
    line=dict(color='royalblue')
)

value_trace = go.Scatter(
   x=np.arange(len(predictions)),
   y=temporal_serie['value'].loc[window_size:len(temporal_serie)],
   mode='markers',
   name='value',
   showlegend=False,
   marker=dict(
        color=['#FF6347' if a == 1 else '#1E90FF' for a in predictions],
        size=8,
        symbol='circle'
    )
)

# Traces para a legenda de anomalia e normal
anomaly_legend = go.Scatter(
    x=[None],
    y=[None],
    mode='markers',
    name='Anômalia',
    marker=dict(
        color='#FF6347',
        size=8,
        symbol='circle'
    )
)

normal_legend = go.Scatter(
    x=[None],
    y=[None],
    mode='markers',
    name='Normal',
    marker=dict(
        color='#1E90FF',
        size=8,
        symbol='circle'
    )
)

data = [trace1, value_trace, anomaly_legend, normal_legend]
layout = go.Layout(
    title='Série Temporal Real Traffic  detectado',
    xaxis=dict(
        title='nº da Leitura',
        showgrid=False,  # Sem grade
        titlefont=dict(size=16),  # Tamanho da fonte do título do eixo x
        tickfont=dict(size=14)  # Tamanho da fonte dos valores do eixo x
    ),
    yaxis=dict(
        title='Valor',
        showgrid=False,  # Sem grade
        titlefont=dict(size=16),  # Tamanho da fonte do título do eixo y
        tickfont=dict(size=14)  # Tamanho da fonte dos valores do eixo y
    ),
    plot_bgcolor='white',  # Fundo branco
    legend=dict(
        font=dict(size=14),  # Tamanho da fonte das legendas
        orientation="h",  # Legendas na horizontal
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig = go.Figure(data=data, layout=layout)
fig.show()


######################################
# Modelos de gráfico anomalias reais #
######################################

trace1 = go.Scatter(
    x=np.arange(len(temporal_serie)),
    y=temporal_serie['value'],
    mode='lines',
    name='value',
    line=dict(color='royalblue')
)

value_trace = go.Scatter(
    x=np.arange(len(temporal_serie)),
    y=temporal_serie['value'],
    mode='markers',
    name='value',
    showlegend=False,
    marker=dict(
        color=['#FF6347' if a == 1 else '#1E90FF' for a in temporal_serie['anomaly']],
        size=8,
        symbol='circle'
    )
)

anomaly_legend = go.Scatter(
    x=[None],
    y=[None],
    mode='markers',
    name='Anômalia',
    marker=dict(
        color='#FF6347',
        size=8,
        symbol='circle'
    )
)

normal_legend = go.Scatter(
    x=[None],
    y=[None],
    mode='markers',
    name='Normal',
    marker=dict(
        color='#1E90FF',
        size=8,
        symbol='circle'
    )
)

data = [trace1, value_trace, anomaly_legend, normal_legend]

layout = go.Layout(
    title='Série Temporal  Real Traffic  real',
    xaxis=dict(
        title='nº da Leitura',
        showgrid=False,  # Sem grade
        titlefont=dict(size=16),  # Tamanho da fonte do título do eixo x
        tickfont=dict(size=14)  # Tamanho da fonte dos valores do eixo x
    ),
    yaxis=dict(
        title='Valor',
        showgrid=False,  # Sem grade
        titlefont=dict(size=16),  # Tamanho da fonte do título do eixo y
        tickfont=dict(size=14)  # Tamanho da fonte dos valores do eixo y
    ),
    plot_bgcolor='white',  # Fundo branco
    legend=dict(
        font=dict(size=14),  # Tamanho da fonte das legendas
        orientation="h",  # Legendas na horizontal
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig = go.Figure(data=data, layout=layout)
fig.show()
