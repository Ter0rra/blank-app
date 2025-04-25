import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# URL brute du fichier CSV sur GitHub
csv_url = 'https://raw.githubusercontent.com/Ter0rra/chatbot_predict_examscore/main/student_habits_performance.csv'

# Charger le CSV directement depuis GitHub dans un DataFrame
df_student = pd.read_csv(csv_url)
df_student = df_student.drop(['student_id'], axis=1)
df_student['media_hours'] = df_student['netflix_hours'] + df_student['social_media_hours']
df_student = df_student.reindex(['age', 'gender', 'study_hours_per_day', 'social_media_hours','netflix_hours', 'media_hours','part_time_job','attendance_percentage','sleep_hours','diet_quality','exercise_frequency','parental_education_level','internet_quality','mental_health_rating','extracurricular_participation','exam_score'], axis=1)

# Sélectionner les colonnes numériques
df_num_value = df_student.select_dtypes(include=['number'])
df_num_value_final = df_num_value.drop(['age', 'netflix_hours', 'social_media_hours'], axis=1)

# Train a simple model
X = df_num_value_final.drop('exam_score', axis=1)
y = df_num_value_final['exam_score']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modèle de régression linéaire
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Fonction de conversion de la note
def get_grade_color(score):
    if score >= 90:
        return 'A', 'green'
    elif score >= 80:
        return 'B', 'lightgreen'
    elif score >= 70:
        return 'C', 'yellow'
    elif score >= 60:
        return 'D', 'orange'
    elif score >= 50:
        return 'E', 'orangered'
    else:
        return 'F', 'red'

# Interface de l'application
st.title("Prédicteur de score d'examen avec recommandations")

# Nom de l'étudiant
name = st.text_input("Nom de l'étudiant :")

# Sélection des variables d'entrée
inputs = {}
for col in X.columns:
    inputs[col] = st.number_input(f"Entrez la valeur pour {col} :", min_value=0.0, step=0.1)

# Bouton de prédiction
if st.button("Prédire"):
    input_data = []
    for col in X.columns:
        input_data.append(inputs[col])
    
    # Prédiction avec le modèle de régression linéaire
    input_scaled = scaler.transform([input_data])
    prediction = lr_model.predict(input_scaled)[0]
    grade, color = get_grade_color(prediction)

    # Affichage des résultats
    st.write(f"**Score prédit :** {prediction:.2f}")
    st.write(f"**Note :** {grade}")
    st.write(f"**Couleur associée :**")
    st.markdown(f"<h3 style='color:{color};'>{grade}</h3>", unsafe_allow_html=True)
    
    # Affichage de la barre horizontale du score
    fig, ax = plt.subplots()
    ax.barh([name], [prediction], color=color)
    ax.set_xlim(0, 100)
    ax.set_title(f"Score prédit: {prediction:.2f} - Note: {grade}")
    plt.xlabel("Score (0-100)")
    st.pyplot(fig)
