import streamlit as st
import pandas as pd


def main():
    st.title('AceleraDev Data Science')
    st.subheader('Semana 2 - Pré-processamento de Dados em Python')
    file  = st.file_uploader('Escolha a base de dados que deseja analisar (.csv)', type = 'csv')
    if file is not None:
        st.subheader('Estrutura')
        df = pd.read_csv(file)
        st.markdown('**Analisando**')
        select_analise = st.radio('Escolha uma analise abaixo :', ('head', 'info', 'describe'))
        if select_analise == 'head':
            st.table(df.head())
        if select_analise == 'info':
            st.dataframe({'Dtype': df.dtypes, 'Non-Null Count' :df.count()})
        if select_analise == 'describe':
            st.table(df.describe())
        st.markdown('**Q1**')
        st.write('Nº de obserações:', df.shape[0], ' Nº de colunas:', df.shape[1])
        st.write('Sugestão: df.shape')
        st.markdown('**Q2**')
        st.write('Nº de mulheres com idade entre 26 e 35 anos no dataset:', len(df.query('Age == "26-35" & Gender == "F"')))
        st.write('Sugestão: len(df.query(\'Filtro1 \& Filtro2\'))')
        st.markdown('**Q3**')
        st.write('Nº dusuários únicos há no dataset:', len(df.groupby('User_ID')['User_ID']))
        st.write('Sugestão: len(df.groupby(\'ColunaUuários\')[\'ColunaUuários\'])')
        st.markdown('**Q4**')
        st.write('Nº tipos de dados diferentes existem no dataset:', len(df.dtypes.value_counts()))
        st.write('Sugestão: len(df.dtypes.value_counts())')
        st.markdown('**Q5**')
        st.write('Porcentagem dos registros possui ao menos um valor null (None, ǸaN etc)', (df.shape[0] - df.dropna().shape[0]) / df.shape[0])
        st.write('Sugestão: (df.shape[0] - df.dropna().shape[0]) / df.shape[0]')
        st.markdown('**Q6**')
        st.write('Valores null existem na variável (coluna) com o maior número de null:', df.isnull().sum().max())
        st.write('Sugestão: df.isnull().sum().max()')
        st.markdown('**Q7**')
        st.write('Valor mais frequente (sem contar nulls) em Product_Category_3:', int(df['Product_Category_3'].mode().values))
        st.write('Sugestão: df[\'Coluna\'].mode().values')
        st.markdown('**Q8**')
        st.write('Nova média da variável (coluna) Purchase após sua normalização:', float(((df[['Purchase']] - df[['Purchase']].min()) / (df[['Purchase']].max() - df[['Purchase']].min())).mean()))
        st.write('Sugestão: ((df[[\'Coluna\']] - df[[\'Coluna\']].min()) / (df[[\'Coluna\']].max() - df[[\'Coluna\']].min())).mean()')
        st.markdown('**Q9**')
        st.write('Ocorrências entre -1 e 1 inclusive existem da variáel Purchase após sua padronização:', len(((df[['Purchase']] - df[['Purchase']].mean()) / df[['Purchase']].std()).query('Purchase >= -1 & Purchase <= 1')))
        st.write('Sugestão: len(((df[[\'Coluna\']] - df[[\'Coluna\']].mean()) / df[[\'Coluna\']].std()).query(Coluna >= -1 \& Coluna <= 1\'))')
        st.markdown('**Q10**')
        st.write('Podemos afirmar que se uma observação é null em Product_Category_2 ela também o é em Product_Category_3:', len(df.isna().query('Product_Category_2 == True & Product_Category_3 == False')) == False)
        st.write('Sugestão: len(df.isna().query(\'Product_Category_2 == True & Product_Category_3 == False\')) == False')
        
if __name__ == '__main__':
	main()
