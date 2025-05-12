import streamlit as st
import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Heart Disease Prediction")
with open('model_heart_disease.pkl', 'rb') as f:
    model = pkl.load(f)
age = st.number_input("Age", min_value=0, max_value=120, value=25)
sex = st.selectbox("Sexe", options=["Homme","Femme"])
chest_pain = st.selectbox("Type of chest pain", options=["Asymptomatic", "Atypical angina", "Non-anginal pain", "Typical angina"])
resting_bp = st.number_input("Resting blood pressure (in mm Hg)", min_value=0, max_value=300, value=120)
cholesterol = st.number_input("Cholesterol (in mg/dl)", min_value=0, max_value=600, value=200)
fasting_bs = st.selectbox("Fasting blood sugar > 120 mg/dl", options=["Oui", "Non"])
resting_ecg = st.selectbox("Resting electrocardiographic results", options=["Normal", "Having ST-T wave abnormality", "Showing probable or definite left ventricular hypertrophy"])
max_heart_rate = st.number_input("Maximum heart rate achieved", min_value=0, max_value=250, value=150)
exercise_angina = st.selectbox("Exercise induced angina", options=["Oui", "Non"])
oldpeak = st.number_input("Oldpeak (depression induced by exercise relative to rest)", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of the peak exercise ST segment", options=["Upsloping", "Flat", "Downsloping"])

# Convertir les catégories en variables numériques
sex_value = 1 if sex == "Homme" else 0
fasting_bs_value = 1 if fasting_bs == "Oui" else 0
exercise_angina_value = 1 if exercise_angina == "Oui" else 0
cpt_Atypical_angina = 1 if chest_pain == "Atypical angina" else 0
cpt_Non_anginal_pain = 1 if chest_pain == "Non-anginal pain" else 0
cpt_Typical_angina = 1 if chest_pain == "Typical angina" else 0
recg_ST_abnormality = 1 if resting_ecg == "Having ST-T wave abnormality" else 0
recg_LVH = 1 if resting_ecg == "Showing probable or definite left ventricular hypertrophy" else 0
slope_Flat = 1 if slope == "Flat" else 0
slope_Downsloping = 1 if slope == "Downsloping" else 0
data = np.array([[age, resting_bp, cholesterol, max_heart_rate, oldpeak,
                  sex_value, fasting_bs_value, exercise_angina_value,
                  cpt_Atypical_angina, cpt_Non_anginal_pain, cpt_Typical_angina,
                  recg_ST_abnormality, recg_LVH,
                  slope_Flat, slope_Downsloping]])
if st.button("Prédire"):
    prediction = model.predict(data)
    if prediction[0] == 1:
        st.error("⚠️ La personne est à risque de maladie cardiaque.")
    else:
        st.success("✅ La personne n'est pas à risque de maladie cardiaque.")
if st.button("Analyse et Visualisation"):
    st.subheader("Chargement des données")
    try:
        df = pd.read_csv("heart.csv")

        # Distribution des variables numériques
        st.write("### Distribution des variables numériques")
        numerical_columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        fig = plt.figure(figsize=(15, 10))
        plt.suptitle('Distribution des variables numériques', fontsize=16)
        for i, col in enumerate(numerical_columns):
            plt.subplot(2, 3, i+1)
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribution de {col}')
            plt.grid(alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Distribution des variables catégorielles
        st.write("### Distribution des variables catégorielles")
        categorical_columns = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'HeartDisease']
        fig = plt.figure(figsize=(15, 12))
        plt.suptitle("Compter les effectifs d'une variable catégorielle:", fontsize=16)
        for i, col in enumerate(categorical_columns):
            plt.subplot(3, 3, i+1)
            sns.countplot(x=df[col], color='teal', alpha=0.7, edgecolor='black')
            plt.title(f'Répartition de {col}')
            plt.grid(alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Heatmap des corrélations
        st.write("### Heatmap des corrélations")
        fig = plt.figure(figsize=(10,7))
        sns.heatmap(df[numerical_columns + ['HeartDisease']].corr(), annot=True, cmap='coolwarm')
        plt.title('Matrice de corrélation')
        st.pyplot(fig)
        plt.close()

        # Répartition des classes de maladie cardiaque
        st.write("### La répartition des classes de maladie cardiaque")
        fig = plt.figure()
        df['HeartDisease'].value_counts().plot(kind='bar', color=['blue', 'orange'], grid=0.2, legend=True, edgecolor='black')
        plt.xlabel('Maladie cardiaque')
        plt.ylabel('Effectif')
        plt.title('Répartition des classes de maladie cardiaque')
        st.pyplot(fig)
        plt.close()
        # Nombre de malades par sexe
        st.write("### Nombre de cas de maladie cardiaque par sexe")
        st.dataframe(df.groupby('Sex')['HeartDisease'].sum().reset_index().rename(columns={'HeartDisease':'Nombre de malades'}))

        # Nombre de malades par type de douleur thoracique
        st.write("### Nombre de cas de maladie cardiaque par type de douleur thoracique")
        st.dataframe(df.groupby('ChestPainType')['HeartDisease'].sum().reset_index().rename(columns={'HeartDisease':'Nombre de malades'}))

        # Moyenne d'âge des malades et non malades
        st.write("### Moyenne d'âge des malades et non malades")
        st.dataframe(df.groupby('HeartDisease')['Age'].mean().reset_index().rename(columns={'Age':'Âge moyen'}))

        # Tous les patients malades
        st.write("### Liste de tous les patients atteints de maladie cardiaque")
        st.dataframe(df[df['HeartDisease'] == 1])

        # Patients malades âgés de moins de 40 ans
        st.write("### Patients malades âgés de moins de 40 ans")
        st.dataframe(df[(df['HeartDisease'] == 1) & (df['Age'] < 40)])

        # Selectbox : choisir une variable catégorielle
        st.write("### Choisir une variable catégorielle pour afficher le nombre de malades")
        selected_col = st.selectbox("Sélectionner une variable:", ['Sex', 'ChestPainType', 'ExerciseAngina', 'RestingECG', 'ST_Slope'])
        st.dataframe(df.groupby(selected_col)['HeartDisease'].sum().reset_index().rename(columns={'HeartDisease':'Nombre de malades'}))

        # Statistiques descriptives des variables numériques
        st.write("### Statistiques descriptives des variables numériques")
        numerical_columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        st.dataframe(df[numerical_columns].describe())
        ### Patients selon l'intervalle d'âge sélectionné
        age_filter = st.slider("Filtrer les patients par âge", min_value=int(df['Age'].min()), max_value=int(df['Age'].max()), value=(30, 60))
        st.write("### Patients selon l'intervalle d'âge sélectionné")
        st.dataframe(df[(df['Age'] >= age_filter[0]) & (df['Age'] <= age_filter[1])])

    except Exception as e:
        st.error(f"Erreur lors du chargement ou de l'analyse des données: {e}")


