import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv("players_22.csv")
df = df[['short_name', 'overall', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']]
df.dropna(inplace=True)

# Sidebar input
st.sidebar.title("Custom Player Stats")
pace = st.sidebar.slider("Pace", 30, 99, 70)
shooting = st.sidebar.slider("Shooting", 30, 99, 70)
passing = st.sidebar.slider("Passing", 30, 99, 70)
dribbling = st.sidebar.slider("Dribbling", 30, 99, 70)
defending = st.sidebar.slider("Defending", 30, 99, 70)
physic = st.sidebar.slider("Physic", 30, 99, 70)

# Prepare data
X = df[['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']]
y = df['overall']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
custom_input = [[pace, shooting, passing, dribbling, defending, physic]]
predicted_rating = model.predict(custom_input)[0]

st.title("⚽ FIFA Rating Predictor")
st.write("### Your player's predicted overall rating:")
st.success(f"{predicted_rating:.2f}")

