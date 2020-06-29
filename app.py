import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

def main():
    st.title("Binary Classification Web App (Created by Adyaan)")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous? 🍄")
    st.sidebar.markdown("Are your mushrooms edible or poisonous? 🍄")
    

    @st.cache(persist=True)
    #@st.cache(persist=True)
    def load_data():
        data = pd.read_csv("E:\Project\mushrooms.csv")
        labelEncoder = LabelEncoder()
        for col in data.columns:
            data[col] = labelEncoder.fit_transform(data[col])
        return data
    

    @st.cache(persist=True)
    def split(df):
        y = df.type
        X = df.drop("type", axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        return X_train, X_test, y_train, y_test

    def plot_metrics(metrics_list):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, X_test, y_test, display_labels=class_names)
            st.pyplot()
        
        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, X_test, y_test)
            st.pyplot()

        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, X_test, y_test)
            st.pyplot()

    df = load_data()
    class_names = ["edible", "poisonous"]

    if st.sidebar.checkbox("Show row data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)
        st.write("The shape of data", df.shape)
        st.markdown("This [data set](https://archive.ics.uci.edu/ml/datasets/Mushroom) includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms "
        "in the Agaricus and Lepiota Family (pp. 500-525). Each species is identified as definitely edible, definitely poisonous, "
        "or of unknown edibility and not recommended. This latter class was combined with the poisonous one.")

    X_train, X_test, y_train, y_test = split(df)

    st.sidebar.subheader("Choose a Classifier")
    Classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regession","Random Forest"))

    if Classifier=='Support Vector Machine (SVM)':
        st.sidebar.subheader('Model Hyperparameters')
        ##Choose Parameters\
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
        
        metrics = st.sidebar.multiselect("Whitch metrics to plot?", ("Confusion Matrix", "Roc-Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if Classifier=='Logistic Regession':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum Number of iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("Which metrics to plot?", ("Confusion Matrix",'ROC-Curve', 'precision-Recall Curve'))

        if st.sidebar.button("Classify", key='Classify'):
            st.subheader("Logistc Regression Results")
            #model = LogisticRegression(C=C, penalty='12', max_iter=max_iter)
            model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
            #model.fit(X_train, y_train)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write('Recall: ', recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if Classifier=='Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 10, 5000, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='n_estimators')
        bootstrap = st.sidebar.radio("Bootstrap samples when builidng tees", ("True", 'False'), key='bootstrap')
        
        metrics = st.sidebar.multiselect("Which metrics to plot?", ("Confusion Matrix",'ROC-Curve', 'precision-Recall Curve'))
        
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Classifer Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    # if st.sidebar.checkbox("Show row data", False):
    #     st.subheader("Mushroom Data Set (Classification)")
    #     st.write(df)
    #     st.markdown("This [data set](https://archive.ics.uci.edu/ml/datasets/Mushroom) includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms "
    #     "in the Agaricus and Lepiota Family (pp. 500-525). Each species is identified as definitely edible, definitely poisonous, "
    #     "or of unknown edibility and not recommended. This latter class was combined with the poisonous one.")

if __name__ == '__main__':
    main()