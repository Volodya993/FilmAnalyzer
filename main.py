import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('C:/Users/skrip/PycharmProjects/pythonProject/imdb_top_1000.csv')

df['IMDB_Rating'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Movie Ratings')
plt.xlabel('IMDB_Rating')
plt.ylabel('Number of Movies')
plt.show()

df['Gross'] = pd.to_numeric(df['Gross'].str.replace(',', '').fillna('0'), errors='coerce')

plt.figure()
df['Gross'].plot(kind='hist', bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Gross Revenue')
plt.xlabel('Gross')
plt.ylabel('Number of Movies')
plt.show()

df['Runtime'] = pd.to_numeric(df['Runtime'].str.replace(' min', ''), errors='coerce')

df = df.dropna(subset=['Runtime', 'Gross'])

features = df[['Runtime', 'Gross']]
target = df['IMDB_Rating']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


plt.scatter(X_test['Runtime'], y_test, color='black', label='Actual Ratings')
plt.scatter(X_test['Runtime'], y_pred, color='blue', label='Predicted Ratings')
plt.title('Actual vs Predicted Ratings')
plt.xlabel('Runtime')
plt.ylabel('IMDB Rating')
plt.legend()
plt.show()


df = df.dropna(subset=['Runtime', 'Gross', 'IMDB_Rating'])


features = df[['Runtime', 'Gross']]
target = df['IMDB_Rating']


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


correlation_matrix = df[['Runtime', 'Gross', 'IMDB_Rating']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()