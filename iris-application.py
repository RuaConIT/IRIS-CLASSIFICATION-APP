import streamlit as st
import pandas as pd
import joblib
from sklearn import neighbors
from sklearn.model_selection import train_test_split

st.write("""
# Phân loại hoa Iris một cách dễ dàng
Hỗ trợ điều chỉnh qua giao diện


""")

st.sidebar.header('Thanh điều chỉnh kích cỡ')
st.subheader('Tham số đầu vào')

def input_bar():
    sepal_l = st.sidebar.slider('Chiều dài đài hoa (cm)', 4.3, 7.9)
    sepal_w = st.sidebar.slider('Chiều rộng đài hoa (cm)', 2.0, 4.4)
    petal_l = st.sidebar.slider('Chiều dài cánh hoa (cm)', 1.0, 6.9)
    petal_w = st.sidebar.slider('Chiều rộng cánh hoa (cm)', 0.1, 2.5)
    data = {
        'Chiều dài đài hoa': sepal_l,
        'Chiều rộng đài hoa': sepal_w,
        'Chiều dài cánh hoa': petal_l,
        'Chiều rộng cánh hoa': petal_w,
    }
    features = pd.DataFrame(data, index = [0])
    return features

data = input_bar()

st.write(data)

option = st.selectbox(
'Lựa chọn mô hình thử nghiệm',
('-- Vui lòng chọn --', 'SVM', 'LogisticRegression', 'HCA-KNN'))
 
if option == 'HCA-KNN':
    dataset = pd.read_csv('dataset/new-data.csv')
    y_iris = dataset['Species']
    X_iris = dataset.drop(['Species',  'Id'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, stratify=y_iris, test_size= 0.3)
    model = neighbors.KNeighborsClassifier(n_neighbors = 3, p = 2)
    model.fit(X_train, y_train)
    predict = model.predict(data)
    predict_proba = model.predict_proba(data)

    st.subheader('Kết quả dự đoán')
    if predict == 0:
        st.write('Iris-setosa')
    elif predict == 1 :
        st.write('Iris-versicolor')
    else:
        st.write('Iris-virginica')

    st.subheader('Phần trăm dự đoán')
    st.write(predict_proba)
elif option == '-- Vui lòng chọn --':
    pass

else:
    filename = 'models/' + option + '.sav'
    model = joblib.load(filename)

    predict = model.predict(data)
    predict_proba = model.predict_proba(data)
    st.subheader('Kết quả dự đoán')
    st.write(predict[0])


    data = {
        'Iris-setosa': predict_proba[0][0],
        'Iris-versicolor': predict_proba[0][1],
        'Iris-virginica': predict_proba[0][2],
    }
    features = pd.DataFrame(data, index = [0])

    st.subheader('Phần trăm dự đoán')
    st.write(features)



# st.write("""
#     **Ghi chú về ký hiệu:**
#     - 0: **Hoa Iris-setosa**
#     - 1: **Hoa Iris-versicolor**
#     - 2: **Hoa Iris-virginica**

# """)