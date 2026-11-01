import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.title("Analisis Konsumsi Alkohol Mahasiswa (student-por.csv)")

# ============================
# LOAD DATASET
# ============================
uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(
            io.BytesIO(uploaded_file.getvalue()),
            sep=None,
            engine="python"
        )
        st.success(f"Dataset {uploaded_file.name} berhasil diunggah.")
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        st.error(f"File CSV tidak dapat dibaca: {error}")
        st.stop()
else:
    try:
        df = pd.read_csv("student-por.csv")
        st.info("Menggunakan student-por.csv. Upload file CSV untuk memakai dataset lain.")
    except FileNotFoundError:
        st.error("File student-por.csv tidak ditemukan. Silakan upload dataset CSV.")
        st.stop()

required_columns = {
    "school", "sex", "address", "famsize", "Pstatus",
    "age", "health", "Walc"
}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    st.error(
        "Dataset tidak memiliki kolom yang diperlukan: "
        + ", ".join(sorted(missing_columns))
    )
    st.stop()

# ============================
# PREVIEW & FILTER
# ============================
st.subheader("Dataset Awal")
st.dataframe(df)

st.subheader("Filter Data")

# Filter: Kategorikal
filter_school = st.multiselect("Pilih Sekolah", options=df['school'].unique(), default=list(df['school'].unique()))
filter_sex = st.multiselect("Pilih Gender", options=df['sex'].unique(), default=list(df['sex'].unique()))
filter_address = st.multiselect("Tipe Alamat", options=df['address'].unique(), default=list(df['address'].unique()))
filter_famsize = st.multiselect("Ukuran Keluarga", options=df['famsize'].unique(), default=list(df['famsize'].unique()))
filter_pstatus = st.multiselect("Status Orang Tua", options=df['Pstatus'].unique(), default=list(df['Pstatus'].unique()))

# Filter: Numerik (pakai rentang usia)
filter_age = st.slider("Rentang Usia", int(df['age'].min()), int(df['age'].max()), (int(df['age'].min()), int(df['age'].max())))
filter_health = st.slider("Kesehatan (1 buruk - 5 baik)", 1, 5, (1, 5))

# Terapkan filter
df_filtered = df[
    (df['school'].isin(filter_school)) &
    (df['sex'].isin(filter_sex)) &
    (df['address'].isin(filter_address)) &
    (df['famsize'].isin(filter_famsize)) &
    (df['Pstatus'].isin(filter_pstatus)) &
    (df['age'].between(filter_age[0], filter_age[1])) &
    (df['health'].between(filter_health[0], filter_health[1]))
]

st.write(f"Jumlah data setelah filter: {df_filtered.shape[0]}")
st.dataframe(df_filtered)

# ============================
# INFO & VISUALISASI
# ============================
st.subheader("Descriptive Statistics")
st.write(df_filtered.describe())

st.subheader("Distribusi Konsumsi Alkohol di Akhir Pekan (Walc)")
fig, ax = plt.subplots()
sns.countplot(x='Walc', data=df_filtered, ax=ax)
ax.set_title('Distribusi Konsumsi Alkohol di Akhir Pekan')
st.pyplot(fig)

# ============================
# MODELING
# ============================
categorical_cols = df_filtered.select_dtypes(include='object').columns
df_encoded = df_filtered.copy()
le = LabelEncoder()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

df_encoded['high_alcohol'] = (df_filtered['Walc'] > 3).astype(int)
X = df_encoded.drop(columns=['Walc', 'high_alcohol'])
y = df_encoded['high_alcohol']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
param_grid_lr = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear']
}
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr, cv=5, scoring='accuracy')
grid_lr.fit(X_train, y_train)

# Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy')
grid_rf.fit(X_train, y_train)

# KNN
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring='accuracy')
grid_knn.fit(X_train, y_train)

# ============================
# TUNING RESULT
# ============================
st.subheader("Hasil Hyperparameter Tuning")

st.write("**Logistic Regression**")
st.write("Best Params:", grid_lr.best_params_)
st.write("Best Accuracy:", grid_lr.best_score_)

st.write("**Random Forest**")
st.write("Best Params:", grid_rf.best_params_)
st.write("Best Accuracy:", grid_rf.best_score_)

st.write("**KNN**")
st.write("Best Params:", grid_knn.best_params_)
st.write("Best Accuracy:", grid_knn.best_score_)

# ============================
# EVALUASI MODEL
# ============================
st.subheader("Evaluasi Model pada Data Test")

models = {
    'Logistic Regression': grid_lr.best_estimator_,
    'Random Forest': grid_rf.best_estimator_,
    'KNN': grid_knn.best_estimator_
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.markdown(f"### {name}")
    st.write(f"Accuracy: {acc:.4f}")
    st.text(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Low', 'High'], yticklabels=['Low', 'High'], ax=ax)
    ax.set_title(f'{name} - Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
