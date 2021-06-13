# Aqui vai ter o APP
import streamlit as st
import pickle


# título
st.title("Sistema de Previsão de Risco de Inadimplência - By Makson Vinicio")

# subtítulo
st.markdown("Este é um Aplicativo utilizado para exibir a solução de Ciência de Dados para o problema de predição de Risco de Inadimplência.")


pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)


@st.cache()
def predict(util_linhas_inseguras, idade, vezes_passou_de_30_59_dias, razao_debito, salario_mensal,
            numero_linhas, numero_vezes_passou_90_dias, numero_emprestimos_imobiliarios, numero_de_vezes_que_passou_60_89_dias, numero_de_dependentes):

    if predict:
        result = classifier.predict([[util_linhas_inseguras, idade, vezes_passou_de_30_59_dias, razao_debito, salario_mensal,
                                      numero_linhas, numero_vezes_passou_90_dias, numero_emprestimos_imobiliarios, numero_de_vezes_que_passou_60_89_dias, numero_de_dependentes]])

        result = result[0]

        if result:
            return 'Crédito Negado'
        else:
            return 'Crédito Aprovado'


def main():

    # front end elements of the web page
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1>
    </div>
    """
    idade = st.number_input(
        'Qual a sua idade: ', min_value=18, max_value=80)

    salario_mensal = st.number_input(
        'Qual o seu salário: ', min_value=0)

    numero_linhas = st.number_input(
        'Qual numero de créditos em aberto em sua conta: ', min_value=0)

    vezes_passou_de_30_59_dias = st.number_input(
        'Você já atrasou por mais de 30 dias? ', min_value=0)

    numero_de_vezes_que_passou_60_89_dias = st.number_input(
        'Você já atrasou por mais de 60 dias? ', min_value=0)

    numero_vezes_passou_90_dias = st.number_input(
        'Você já atrasou por mais de 90 dias? ', min_value=0)

    numero_emprestimos_imobiliarios = st.number_input(
        'Número de emprestimos imobiliários: ', min_value=0)

    numero_de_dependentes = st.number_input(
        'Número de dependentes: ', min_value=0)

    if st.button('Predict'):
        util_linhas_inseguras = 0.5

        razao_debito = 0.5

        result = predict(util_linhas_inseguras, idade, vezes_passou_de_30_59_dias, razao_debito, salario_mensal,
                         numero_linhas, numero_vezes_passou_90_dias, numero_emprestimos_imobiliarios, numero_de_vezes_que_passou_60_89_dias, numero_de_dependentes)
        st.success('{}'.format(result))


if __name__ == '__main__':
    main()
