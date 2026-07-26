import pandas as pd
import numpy as np 
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierReplacerIQR(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.limite_inferior_ = None
        self.limite_superior_ = None
        self.medianas_ = None

    def fit(self, X, y=None):
        # Converte para DataFrame temporariamente caso venha como array NumPy
        X_df = pd.DataFrame(X)
        
        # Calcula estatísticas coluna por coluna
        Q1 = X_df.quantile(0.25)
        Q3 = X_df.quantile(0.75)
        IQR = Q3 - Q1
        
        self.limite_inferior_ = Q1 - 1.5 * IQR
        self.limite_superior_ = Q3 + 1.5 * IQR
        self.medianas_ = X_df.median()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        
        # Substitui os outliers pelas medianas salvas no fit()
        for i, coluna in enumerate(X_df.columns):
            lim_inf = self.limite_inferior_.iloc[i]
            lim_sup = self.limite_superior_.iloc[i]
            mediana = self.medianas_.iloc[i]
            
            is_outlier = (X_df[coluna] < lim_inf) | (X_df[coluna] > lim_sup)
            X_df.loc[is_outlier, coluna] = mediana
            
        return X_df.values

    def get_feature_names_out(self, input_features=None):
        """Retorna os mesmos nomes das colunas de entrada, já que não criamos novas colunas."""
        return np.array(input_features) if input_features is not None else np.array([])