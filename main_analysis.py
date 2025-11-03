# main_analysis.py
"""
Análise de Dados - Dissertação: IA, ML e Psicologia
Autor: Saulo Santos Menezes de Almeida
Universidade Salvador - UNIFACS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configuração de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class TriagemPsicologicaAnalyzer:
    """
    Classe para análise de dados de triagem psicológica
    """
    
    def __init__(self, data_path=None):
        """
        Inicializa o analisador
        
        Args:
            data_path: caminho para o arquivo de dados
        """
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, file_path):
        """
        Carrega dados de arquivo CSV ou Excel
        
        Args:
            file_path: caminho do arquivo
        """
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path, encoding='utf-8')
            elif file_path.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(file_path)
            else:
                raise ValueError("Formato não suportado. Use CSV ou Excel.")
            
            print(f"✓ Dados carregados com sucesso!")
            print(f"  Shape: {self.data.shape}")
            print(f"  Colunas: {list(self.data.columns)}")
            
        except Exception as e:
            print(f"✗ Erro ao carregar dados: {e}")
    
    def exploratory_analysis(self):
        """
        Realiza análise exploratória dos dados
        """
        if self.data is None:
            print("✗ Carregue os dados primeiro!")
            return
        
        print("\n" + "="*60)
        print("ANÁLISE EXPLORATÓRIA DE DADOS")
        print("="*60)
        
        # Informações gerais
        print("\n1. INFORMAÇÕES GERAIS")
        print("-" * 60)
        print(self.data.info())
        
        # Estatísticas descritivas
        print("\n2. ESTATÍSTICAS DESCRITIVAS")
        print("-" * 60)
        print(self.data.describe())
        
        # Valores ausentes
        print("\n3. VALORES AUSENTES")
        print("-" * 60)
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data)) * 100
        missing_df = pd.DataFrame({
            'Coluna': missing.index,
            'Valores Ausentes': missing.values,
            'Percentual (%)': missing_pct.values
        })
        print(missing_df[missing_df['Valores Ausentes'] > 0])
        
        # Distribuição de variáveis categóricas
        print("\n4. DISTRIBUIÇÃO DE VARIÁVEIS CATEGÓRICAS")
        print("-" * 60)
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            print(f"\n{col}:")
            print(self.data[col].value_counts())
    
    def visualize_distributions(self, save_path='./plots/'):
        """
        Cria visualizações das distribuições dos dados
        
        Args:
            save_path: diretório para salvar os gráficos
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        if self.data is None:
            print("✗ Carregue os dados primeiro!")
            return
        
        # Distribuição de variáveis numéricas
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(
                nrows=(len(numeric_cols) + 2) // 3,
                ncols=3,
                figsize=(15, 5 * ((len(numeric_cols) + 2) // 3))
            )
            axes = axes.flatten() if len(numeric_cols) > 1 else [axes]
            
            for idx, col in enumerate(numeric_cols):
                axes[idx].hist(self.data[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'Distribuição: {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequência')
            
            # Remove eixos extras
            for idx in range(len(numeric_cols), len(axes)):
                fig.delaxes(axes[idx])
            
            plt.tight_layout()
            plt.savefig(f'{save_path}distribuicoes_numericas.png', dpi=300, bbox_inches='tight')
            print(f"✓ Gráfico salvo: {save_path}distribuicoes_numericas.png")
            plt.close()
        
        # Distribuição de variáveis categóricas
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            plt.figure(figsize=(10, 6))
            value_counts = self.data[col].value_counts()
            
            if len(value_counts) <= 10:
                value_counts.plot(kind='bar', edgecolor='black', alpha=0.7)
                plt.title(f'Distribuição: {col}')
                plt.xlabel(col)
                plt.ylabel('Frequência')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(f'{save_path}distribuicao_{col}.png', dpi=300, bbox_inches='tight')
                print(f"✓ Gráfico salvo: {save_path}distribuicao_{col}.png")
                plt.close()
    
    def correlation_analysis(self, save_path='./plots/'):
        """
        Análise de correlação entre variáveis
        
        Args:
            save_path: diretório para salvar os gráficos
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        if self.data is None:
            print("✗ Carregue os dados primeiro!")
            return
        
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            print("✗ Não há variáveis numéricas suficientes para análise de correlação")
            return
        
        # Matriz de correlação
        corr_matrix = numeric_data.corr()
        
        # Visualização
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8}
        )
        plt.title('Matriz de Correlação', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_path}matriz_correlacao.png', dpi=300, bbox_inches='tight')
        print(f"✓ Matriz de correlação salva: {save_path}matriz_correlacao.png")
        plt.close()
        
        # Correlações mais fortes
        print("\n" + "="*60)
        print("CORRELAÇÕES MAIS FORTES (|r| > 0.5)")
        print("="*60)
        
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.5:
                    corr_pairs.append({
                        'Variável 1': corr_matrix.columns[i],
                        'Variável 2': corr_matrix.columns[j],
                        'Correlação': corr_matrix.iloc[i, j]
                    })
        
        if corr_pairs:
            corr_df = pd.DataFrame(corr_pairs).sort_values('Correlação', ascending=False)
            print(corr_df.to_string(index=False))
        else:
            print("Nenhuma correlação forte encontrada.")
    
    def prepare_ml_data(self, target_column, test_size=0.2, random_state=42):
        """
        Prepara dados para machine learning
        
        Args:
            target_column: nome da coluna alvo
            test_size: proporção do conjunto de teste
            random_state: seed para reprodutibilidade
        """
        if self.data is None:
            print("✗ Carregue os dados primeiro!")
            return
        
        if target_column not in self.data.columns:
            print(f"✗ Coluna '{target_column}' não encontrada!")
            return
        
        # Separar features e target
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        # Converter variáveis categóricas
        X = pd.get_dummies(X, drop_first=True)
        
        # Split train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Normalização
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"✓ Dados preparados para ML!")
        print(f"  Train set: {self.X_train.shape}")
        print(f"  Test set: {self.X_test.shape}")
        print(f"  Classes: {np.unique(y)}")
    
    def train_classifiers(self):
        """
        Treina múltiplos classificadores
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.naive_bayes import GaussianNB
        
        if self.X_train is None:
            print("✗ Prepare os dados primeiro usando prepare_ml_data()!")
            return
        
        classifiers = {
            'Regressão Logística': LogisticRegression(max_iter=1000, random_state=42),
            'Árvore de Decisão': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            'Naive Bayes': GaussianNB()
        }
        
        results = []
        
        print("\n" + "="*60)
        print("TREINAMENTO E AVALIAÇÃO DE CLASSIFICADORES")
        print("="*60)
        
        for name, clf in classifiers.items():
            print(f"\n{name}:")
            print("-" * 60)
            
            # Treinar
            clf.fit(self.X_train, self.y_train)
            
            # Predições
            y_pred = clf.predict(self.X_test)
            
            # Métricas
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            
            results.append({
                'Classificador': name,
                'Acurácia': accuracy,
                'Precisão': precision,
                'Recall': recall,
                'F1-Score': f1
            })
            
            print(f"Acurácia: {accuracy:.4f}")
            print(f"Precisão: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            
            print("\nRelatório de Classificação:")
            print(classification_report(self.y_test, y_pred, zero_division=0))
        
        # Resumo
        results_df = pd.DataFrame(results).sort_values('Acurácia', ascending=False)
        
        print("\n" + "="*60)
        print("RESUMO DOS RESULTADOS")
        print("="*60)
        print(results_df.to_string(index=False))
        
        return results_df
    
    def plot_confusion_matrices(self, save_path='./plots/'):
        """
        Plota matrizes de confusão para os classificadores
        
        Args:
            save_path: diretório para salvar os gráficos
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.naive_bayes import GaussianNB
        import os
        
        os.makedirs(save_path, exist_ok=True)
        
        if self.X_train is None:
            print("✗ Prepare os dados primeiro!")
            return
        
        classifiers = {
            'Regressão Logística': LogisticRegression(max_iter=1000, random_state=42),
            'Árvore de Decisão': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            'Naive Bayes': GaussianNB()
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (name, clf) in enumerate(classifiers.items()):
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_test)
            
            cm = confusion_matrix(self.y_test, y_pred)
            
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=axes[idx],
                cbar=False
            )
            axes[idx].set_title(name, fontweight='bold')
            axes[idx].set_xlabel('Predito')
            axes[idx].set_ylabel('Real')
        
        # Remove o último subplot se houver
        if len(classifiers) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(f'{save_path}matrizes_confusao.png', dpi=300, bbox_inches='tight')
        print(f"✓ Matrizes de confusão salvas: {save_path}matrizes_confusao.png")
        plt.close()
    
    def generate_report(self, output_file='relatorio_analise.txt'):
        """
        Gera relatório completo da análise
        
        Args:
            output_file: nome do arquivo de saída
        """
        if self.data is None:
            print("✗ Carregue os dados primeiro!")
            return
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RELATÓRIO DE ANÁLISE DE DADOS\n")
            f.write("Dissertação: IA, ML e Psicologia - Triagem Psicológica\n")
            f.write("Autor: Saulo Santos Menezes de Almeida\n")
            f.write("="*80 + "\n\n")
            
            f.write("1. INFORMAÇÕES GERAIS DO DATASET\n")
            f.write("-"*80 + "\n")
            f.write(f"Número de registros: {len(self.data)}\n")
            f.write(f"Número de variáveis: {len(self.data.columns)}\n")
            f.write(f"Variáveis: {', '.join(self.data.columns)}\n\n")
            
            f.write("2. ESTATÍSTICAS DESCRITIVAS\n")
            f.write("-"*80 + "\n")
            f.write(self.data.describe().to_string())
            f.write("\n\n")
            
            f.write("3. VALORES AUSENTES\n")
            f.write("-"*80 + "\n")
            missing = self.data.isnull().sum()
            for col, count in missing.items():
                if count > 0:
                    pct = (count / len(self.data)) * 100
                    f.write(f"{col}: {count} ({pct:.2f}%)\n")
            f.write("\n")
        
        print(f"✓ Relatório gerado: {output_file}")


def main():
    """
    Função principal para executar análises
    """
    print("="*60)
    print("SISTEMA DE ANÁLISE - TRIAGEM PSICOLÓGICA")
    print("Dissertação: IA, ML e Psicologia")
    print("="*60)
    
    # Exemplo de uso
    analyzer = TriagemPsicologicaAnalyzer()
    
    # Instruções
    print("\nPara usar este sistema:")
    print("1. Prepare seus dados em formato CSV ou Excel")
    print("2. Carregue os dados: analyzer.load_data('caminho/arquivo.csv')")
    print("3. Execute análise exploratória: analyzer.exploratory_analysis()")
    print("4. Visualize distribuições: analyzer.visualize_distributions()")
    print("5. Análise de correlação: analyzer.correlation_analysis()")
    print("6. Prepare para ML: analyzer.prepare_ml_data('coluna_alvo')")
    print("7. Treine classificadores: analyzer.train_classifiers()")
    print("8. Gere relatório: analyzer.generate_report()")
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
