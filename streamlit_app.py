import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Charger les données (assurez-vous de définir df_student_ml correctement ou de le charger depuis un CSV)
csv_url = "https://raw.githubusercontent.com/Ter0rra/blank-app/6813973fbe231ac40da9129ab94dca649ee09702/student_habits_performance.csv"
df_student = pd.read_csv(csv_url)
df_student = df_student.drop(['student_id'], axis=1)
df_student['media_hours'] = df_student['netflix_hours'] + df_student['social_media_hours']
df_student = df_student.reindex(['age', 'gender', 'study_hours_per_day', 'social_media_hours','netflix_hours', 'media_hours','part_time_job','attendance_percentage','sleep_hours','diet_quality','exercise_frequency','parental_education_level','internet_quality','mental_health_rating','extracurricular_participation','exam_score'], axis=1)
df_num_value = df_student.select_dtypes(include=['number'])
df_student_ml = df_num_value.drop(['age', 'social_media_hours', 'netflix_hours'], axis=1)

# Séparation des variables
X = df_student_ml.drop('exam_score', axis=1)
y = df_student_ml['exam_score']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = X.select_dtypes(include=np.number).columns.to_list()
categorical_features = X.select_dtypes(include='object').columns.to_list()

# Pipelines
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

x_train_processed = preprocessor.fit_transform(x_train)
x_test_processed = preprocessor.transform(x_test)

model = LinearRegression()
model.fit(x_train_processed, y_train)

# Fonction de notation
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

# Suggestions dynamiques
def get_dynamic_suggestions(df_input):
    current_score = model.predict(preprocessor.transform(df_input))[0]
    improvements = []
    for col in numeric_features:
        better_value = df_input[col].values[0]
        test_up = df_input.copy()
        test_up[col] = better_value + 1
        score_up = model.predict(preprocessor.transform(test_up))[0]

        test_down = df_input.copy()
        test_down[col] = better_value - 1
        score_down = model.predict(preprocessor.transform(test_down))[0]

        score_up = min(score_up, 100)
        score_down = min(score_down, 100)

        delta_up = score_up - current_score
        delta_down = score_down - current_score

        best_delta = max(delta_up, delta_down)
        if best_delta > 0:
            direction = "augmenter" if delta_up > delta_down else "réduire"
            improvements.append((col, direction, best_delta))

    improvements.sort(key=lambda x: x[2], reverse=True)
    return improvements[:3]

# Interface utilisateur Streamlit
st.title("Prédicteur de Score d'Examen avec Suggestions Dynamiques")

# Entrée de l'utilisateur
name = st.text_input("Nom de l'étudiant :")

# Entrées des caractéristiques
input_data = {}
for col in X.columns:
    if col in categorical_features:
        input_data[col] = st.selectbox(f"Choix pour {col} :", df_student_ml[col].dropna().unique())
    else:
        input_data[col] = st.number_input(f"Entrez la valeur pour {col} :", min_value=0.0, step=0.1)

# Bouton de prédiction
if st.button("Prédire la note d'examen"):
    try:
        df_input = pd.DataFrame([input_data])
        df_processed = preprocessor.transform(df_input)

        # Prédiction du score
        score = model.predict(df_processed)[0]
        score = min(score, 100)  # S'assurer que le score ne dépasse pas 100
        grade, color = get_grade_color(score)

        # Affichage du score prédit
        st.write(f"**Score prédit pour {name}:** {score:.2f}")
        st.write(f"**Note :** {grade}")

        # Affichage de la barre horizontale du score
        fig, ax = plt.subplots()
        ax.barh([name], [score], color=color)
        ax.set_xlim(0, 100)
        ax.set_title(f"Score prédit: {score:.2f} - Note: {grade}")
        st.pyplot(fig)

        # Suggestions d'amélioration
        if score >= 90:
            st.success("🎉 Continue comme ça, c'est parfait ! 🎉")
        else:
            suggestions = get_dynamic_suggestions(df_input)
            st.subheader("Suggestions pour améliorer votre score :")
            for var, direction, gain in suggestions:
                st.write(f"- {direction} {var} → gain estimé de {gain:.2f} points")

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
