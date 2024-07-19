import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math

class Recomendador:
    def __init__(self, df):
        # Inicializa a classe Recomendador com o DataFrame df
        self.df = df
        self.dividir_treino_teste()  # Divide os dados em conjuntos de treino e teste
        self.criar_matrizes()  # Cria as matrizes de usuário-item e de similaridade

    def dividir_treino_teste(self, test_size=0.1, random_state=42):
        # Divide os dados em conjunto de treino e teste com uma proporção de 10% para teste
        X_treino, X_teste = train_test_split(self.df, test_size=test_size, random_state=random_state)
        # Divide o conjunto de treino em base e validação com a mesma proporção
        self.X_base, self.X_validacao = train_test_split(X_treino, test_size=test_size, random_state=random_state)
        self.X_teste = X_teste  # Define o conjunto de teste
        self.u_medias = self.user_media(X_base)

    def itens_comuns(self, u, v):
        # Retorna os índices dos itens avaliados em comum pelos usuários u e v
        return [i for i in range(len(u)) if u[i] != 0 and v[i] != 0]

    def calc_divisorCos(self, u, v):
        # Calcula o divisor da fórmula de similaridade cosseno (produto dos módulos dos vetores u e v)
        sum_u = sum([u[i]**2 for i in range(len(u)) if u[i] != 0])
        sum_v = sum([v[i]**2 for i in range(len(v)) if v[i] != 0])
        return math.sqrt(sum_u * sum_v)

    def user_media(self,df):
        #calcula a média de usuarios dentro da matriz
        media_usuarios = df.groupby('id_user')['ratings'].mean()
        return media_usuarios

    def similaridade_cosseno(self, matriz, row):
        # Calcula a matriz de similaridade cosseno entre os usuários
        similaridade = np.zeros((row, row), float)  # Inicializa a matriz de similaridade

        for i in range(row):
            for j in range(i, row):
                if i == j:
                    similaridade[i, j] = 1  # Similaridade de um usuário com ele mesmo é 1
                    continue

                sum = 0
                a = matriz.iloc[i, :].values
                b = matriz.iloc[j, :].values
                itens_c = self.itens_comuns(a, b)
                for k in itens_c:
                    sum += a[k] * b[k]

                divisor = self.calc_divisorCos(a, b)
                if divisor == 0:
                    similaridade[i, j] = 0
                else:
                    similaridade[i, j] = sum / divisor
                similaridade[j, i] = similaridade[i, j]  # Matriz é simétrica

        return similaridade

    def criar_matrizes(self):
        # Cria a matriz de usuário-item e a matriz de similaridade entre usuários
        self.matriz_usuario_item = self.X_base.pivot(index='id_user', columns='id_item', values='ratings')
        self.matriz_similaridade_usuario = self.similaridade_cosseno(self.matriz_usuario_item.fillna(0), self.matriz_usuario_item.shape[0])
        
        print(f"Dimensões da matriz usuário-item: {self.matriz_usuario_item.shape}")
        print(f"Dimensões da matriz de similaridade: {self.matriz_similaridade_usuario.shape}")

    def prever(self, usuario, item, min_avaliacoes=5, vizinhos=20):
        # Prever a avaliação de um item por um usuário
        if usuario >= len(self.matriz_similaridade_usuario) or item >= self.matriz_usuario_item.shape[1]:
            return np.nan

        
        
        vetor_similaridade = self.matriz_similaridade_usuario[usuario]
        vetor_similaridade[usuario] = 0  # Similaridade do usuário com ele mesmo é desconsiderada

        usuarios_similares = np.argsort(vetor_similaridade)[-vizinhos:] # Encontra os usuários mais similares

        usuarios_similares = [u for u in usuarios_similares if not np.isnan(self.matriz_usuario_item.iloc[u, item])]
        
        if len(usuarios_similares) < min_avaliacoes:
            return np.nan

        avaliacoes = self.matriz_usuario_item.iloc[usuarios_similares, item]
        similaridades = vetor_similaridade[usuarios_similares]
        ajuste = self.u_medias[usuarios_similares]

        if np.sum(similaridades) == 0:
            return np.nan
        
        # Calcula a previsão como o produto ponto entre avaliações e similaridades, dividido pela soma das similaridades
        return self.u_medias[usuario] + (similaridades*(avaliacoes - ajuste)/ np.sum(similaridades))


class Avaliador:
    def __init__(self, recomendador):
        # Inicializa a classe Avaliador com um objeto Recomendador
        self.recomendador = recomendador

    def avaliar(self, dataset, min_avaliacoes=5, vizinhos=5):
        # Avalia a precisão do recomendador usando MAE e RMSE
        soma_mae = 0
        soma_rmse = 0
        contador = 0

        for _, linha in dataset.iterrows():
            # Obtém o índice do usuário e do item, ajustando para base 0
            usuario = linha['id_user'] - 1
            item = linha['id_item'] - 1
            
            # Verifica se os índices do usuário e do item estão dentro dos limites válidos
            if usuario < 0 or usuario >= len(self.recomendador.matriz_similaridade_usuario) or item < 0 or item >= self.recomendador.matriz_usuario_item.shape[1]:
                continue
            
            # Obtém a avaliação real do usuário para o item
            avaliacao_verdadeira = linha['ratings']
            # Calcula a avaliação prevista pelo sistema de recomendação
            avaliacao_prevista = self.recomendador.prever(usuario, item, min_avaliacoes, vizinhos)
            
            # Se a avaliação prevista não for NaN, calcula os erros
            if not np.isnan(avaliacao_prevista):
                soma_mae += abs(avaliacao_verdadeira - avaliacao_prevista)
                soma_rmse += (avaliacao_verdadeira - avaliacao_prevista) ** 2
                contador += 1

        mae = soma_mae / contador if contador > 0 else np.nan
        rmse = math.sqrt(soma_rmse / contador) if contador > 0 else np.nan

        if contador == 0:
            print("Nenhuma avaliação válida encontrada")

        return mae, rmse


class Utilitarios:
    @staticmethod
    def carregar_dados(caminho_arquivo):
        # Carrega os dados do arquivo especificado
        nomes_colunas = ['id_user', 'id_item', 'ratings', 'timestamp']
        return pd.read_csv(caminho_arquivo, sep='\t', header=None, names=nomes_colunas)


if __name__ == "__main__":
    # Caminho para o arquivo de dados (substituir pelo caminho correto)
    caminho_arquivo_dados = 'u.data'

    # Carrega os dados
    df = Utilitarios.carregar_dados(caminho_arquivo_dados)

    # Inicializa o recomendador e o avaliador
    recomendador = Recomendador(df)
    avaliador = Avaliador(recomendador)

    # Avalia o recomendador usando o conjunto de validação
    mae, rmse = avaliador.avaliar(recomendador.X_validacao)
    print(f'Validação - MAE: {mae}, RMSE: {rmse}')

    # Avalia o recomendador usando o conjunto de teste
    mae, rmse = avaliador.avaliar(recomendador.X_teste)
    print(f'Teste - MAE: {mae}, RMSE: {rmse}')
