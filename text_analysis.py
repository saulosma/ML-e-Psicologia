# text_analysis.py
"""
Análise de texto para revisão sistemática
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from collections import Counter
import re

class TextAnalyzer:
    """
    Classe para análise de texto da revisão sistemática
    """
    
    def __init__(self):
        # Download de recursos NLTK necessários
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
        except:
            pass
        
        self.stopwords_pt = set([
            'a', 'o', 'e', 'é', 'de', 'da', 'do', 'em', 'um', 'uma',
            'os', 'as', 'dos', 'das', 'para', 'com', 'por', 'que',
            'se', 'na', 'no', 'ao', 'à', 'mais', 'como', 'ser', 'são'
        ])
    
    def extract_keywords(self, text, top_n=20):
        """
        Extrai palavras-chave mais frequentes
        
        Args:
            text: texto para análise
            top_n: número de palavras-chave a retornar
        """
        # Limpar e tokenizar
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        # Remover stopwords
        words = [w for w in words if w not in self.stopwords_pt and len(w) > 3]
        
        # Contar frequências
        word_freq = Counter(words)
        
        return word_freq.most_common(top_n)
    
    def generate_wordcloud(self, text, output_file='wordcloud.png'):
        """
        Gera nuvem de palavras
        
        Args:
            text: texto para análise
            output_file: arquivo de saída
        """
        # Limpar texto
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Gerar wordcloud
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            stopwords=self.stopwords_pt,
            colormap='viridis',
            max_words=100
        ).generate(text)
        
        # Plotar
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Nuvem de Palavras - Revisão Sistemática', fontsize=16, fontweight='bold')
        plt.tight_layout(pad=0)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Nuvem de palavras salva: {output_file}")
        plt.close()
    
    def analyze_systematic_review(self, articles_df):
        """
        Analisa dados da revisão sistemática
        
        Args:
            articles_df: DataFrame com artigos da revisão
        """
        print("\n" + "="*60)
        print("ANÁLISE DA REVISÃO SISTEMÁTICA")
        print("="*60)
        
        # Estatísticas gerais
        print(f"\nTotal de artigos: {len(articles_df)}")
        
        if 'ano' in articles_df.columns:
            print(f"\nDistribuição por ano:")
            print(articles_df['ano'].value_counts().sort_index())
        
        if 'base' in articles_df.columns:
            print(f"\nDistribuição por base de dados:")
            print(articles_df['base'].value_counts())
        
        # Análise de palavras-chave
        if 'keywords' in articles_df.columns:
            all_keywords = ' '.join(articles_df['keywords'].dropna())
            top_keywords = self.extract_keywords(all_keywords, top_n=20)
            
            print(f"\nTop 20 palavras-chave:")
            for word, freq in top_keywords:
                print(f"  {word}: {freq}")


# Exemplo de uso
if __name__ == "__main__":
    analyzer = TextAnalyzer()
    
    # Exemplo com texto da dissertação
    sample_text = """
    inteligência artificial aprendizado máquina psicologia
    triagem psicológica emoções afetos comportamento
    machine learning deep learning classificação
    """
    
    analyzer.generate_wordcloud(sample_text)
