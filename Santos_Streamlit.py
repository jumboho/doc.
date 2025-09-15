import streamlit as st # pyright: ignore[reportMissingImports]
import pandas as pd # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]
from sklearn.svm import SVC # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from sklearn.linear_model import LogisticRegression # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from sklearn.ensemble import RandomForestClassifier # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from sklearn.preprocessing import LabelEncoder # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from sklearn.model_selection import train_test_split # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from sklearn.metrics import precision_score, recall_score # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource, reportMissingImports]


def main():
    st.title("Binary Classification Web App")   
    st.sidebar.title("Model Settings ⚙️") 
    st.markdown("Are your mushrooms edible or poisonous? 🍄🧪︎")
    st.sidebar.markdown("Select a classifier and adjust its hyperparameters.")

    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv('/Users/alexandrasantos/Downloads/mushroom_data_all.csv')
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data
    
    @st.cache_data(persist=True)
    def split(df):
        y = df.type
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test
    
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=class_names, ax=ax, cmap='Blues')
            st.pyplot(fig)

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")  
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            fig, ax = plt.subplots()
            PrecisionRecallDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)
    
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisonous']
    st.sidebar.subheader("Choose Classifier ⌕")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))  
    
    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key="gamma")  

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("See Results", key="classify"):
            st.subheader("Support Vector Machine (SVM) Results ⚡")
            model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)  
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision_score(y_test, y_pred), 2))  
            st.write("Recall: ", round(recall_score(y_test, y_pred), 2))  
            plot_metrics(metrics)

    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_lr')
        solver = st.sidebar.selectbox("Solver", ("lbfgs", "liblinear", "newton-cg"), key='solver')
        
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("See Results", key="classify_lr"):
            st.subheader("Logistic Regression Results 📈")
            model = LogisticRegression(C=C, solver=solver, max_iter=1000)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision_score(y_test, y_pred), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred), 2))
            plot_metrics(metrics)

    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("Number of trees", 10, 500, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("Max depth", 1, 20, step=1, key='max_depth')
        
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("See Results", key="classify_rf"):
            st.subheader("Random Forest Results 🌳")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision_score(y_test, y_pred), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred), 2))
            plot_metrics(metrics)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)

if __name__ == '__main__':
    main()
