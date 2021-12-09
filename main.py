import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# string = "Knowledge shelf"
st.set_page_config(page_title='Salary Prediciton', page_icon="💵")

st.write("""
# Salary prediction model
Salary vs. *Experience*
""")

data = pd.read_csv('Salary_Data.csv')

x = data.iloc[:, [0]].values
y = data.iloc[:, [-1]].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)

exp = st.sidebar.slider('Experience', 1, 10, 2)

reg = LinearRegression()
reg.fit(x_train, y_train)

y_pred = reg.predict([[exp]])
st.write(f"Experience: ", exp)
st.write(f"Salary: ", round(float(y_pred), 2))

st.write("""
# Scatter plot
Salary vs. *Experience*
""")

fig = plt.figure()
plt.scatter(x, y, alpha=0.8, cmap='viridis')

plt.xlabel('Experience')
plt.ylabel('Salary')
plt.colorbar()

st.pyplot(fig)
