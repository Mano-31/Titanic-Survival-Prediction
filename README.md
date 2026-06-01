# Titanic-Survival-Prediction

# 🚢 Titanic Survival Prediction
![Show Image](https://github.com/Mano-31/Titanic-Survival-Prediction/blob/main/eda.png)
![Show Image](https://github.com/Mano-31/Titanic-Survival-Prediction/blob/main/Correlation%20Heatmap.png)
![Show Image](https://github.com/Mano-31/Titanic-Survival-Prediction/blob/main/PCA%20Dimensionality%20Reduction.png)
![Show Image](https://github.com/Mano-31/Titanic-Survival-Prediction/blob/main/K-Means%20Clustering.png)
![Show Image](https://github.com/Mano-31/Titanic-Survival-Prediction/blob/main/K-Means%20Clustering%20pca.png)
![show image](https://github.com/Mano-31/Titanic-Survival-Prediction/blob/main/Random%20Forest%20Classification.png)

# 📌 Project Overview
An intermediate-level end-to-end data science project that analyzes the 

Titanic passenger dataset to discover survival patterns and predict survival using Logistic Regression.

"Females survived at 74% vs males at 19% — the model discovered the 'women and children first' policy purely from data."

🧠 What is this project?
The RMS Titanic sank on April 15, 1912, after colliding with an iceberg. 

Of the 2,224 passengers and crew aboard, more than 1,500 died — making it one of the deadliest peacetime maritime disasters in history. 

This project uses real passenger data to uncover who survived and why.
The dataset contains 891 passenger records with 15 features including age, gender, ticket class, fare, cabin deck, and port of embarkation. 

The target variable is survived — 0 = did not survive, 1 = survived. 

Only 38% of passengers survived, making this a classic imbalanced binary classification problem.

# 🎯 What does this project do?
This project follows a complete 9-step data science pipeline:

Loads the Titanic dataset (891 rows × 15 columns) and explores its structure

Cleans the data — handles 178 missing age values, drops the 77%-null deck column, fills 2 missing embarked values

Engineers features — encodes sex and embarked columns, creates family_size and is_alone variables

Visualizes 4 key survival patterns using a Seaborn dashboard (by gender, class, count, age)

Plots a correlation heatmap to quantify relationships between all features and survival

Trains a Logistic Regression model using StandardScaler and an 80/20 stratified split

Evaluates using accuracy score, classification report, confusion matrix, and feature coefficients

Interprets results — gender (r = −0.49) is the strongest predictor, confirming the historical evacuation policy

Documents everything in a Jupyter Notebook, PPT presentation, and PDF report

# 💡 Why Logistic Regression?
Logistic Regression is the ideal algorithm for this intermediate-level project because:

It is designed for binary classification (survived = 0 or 1)

Its coefficients are fully interpretable — you can see exactly which features drive predictions

It requires StandardScaler preprocessing — teaching an essential ML skill

It achieves realistic 73% accuracy — strong but not suspiciously perfect

It is the right complexity for intermediate level — not a black box like Random Forest

# 📊 What patterns were discovered?
PatternFinding
👩 GenderFemales: 74% survival vs Males: 19% survival

🎫 Class1st class: 62% → 2nd: 47% → 3rd: 24%

💰 FareHigher fare = higher survival (r = +0.26)

👨‍👩‍👧 FamilySolo travelers had lower survival rates

🔢 Correlationsex_enc r = −0.49 (strongest single predictor)

# 🚢 The Most Powerful Insight
The Logistic Regression model discovered the "women and children first" evacuation policy purely from data — without ever being told about it. 

The model assigned the highest coefficient to sex_enc, meaning gender alone was the dominant driver of its predictions.

This perfectly matches the historical record: officers prioritized women and children when loading lifeboats, which is exactly what the model learned.

# 🎯 Key Results
MetricValueDataset891 rows × 15 columns

ModelLogistic RegressionAccuracy73%

Precision0.76

Recall0.73

F1-Score0.72

Top Predictorsex_enc (r = -0.49)

# 🗂️ Project Steps
StepTaskDescription1📦 Import LibrariesPandas, NumPy, Seaborn, Matplotlib, Sklearn2

📊 Load & ExploreShape, dtypes, missing values, stats3

🧹 Data CleaningFill nulls, drop 77%-null deck column4

⚙️ Feature EngineeringEncode sex/embarked, create family_size, is_alone5

📈 EDA Visualization4-chart dashboard: count, sex, class, age6

🔥 Correlation HeatmapQuantify feature-survival relationships7

🤖 Logistic RegressionStandardScaler + stratified 80/20 split8

📋 Model EvaluationAccuracy, confusion matrix, coefficients9

✅ SummaryKey findings and future improvements

# 🔑 Key Findings

Gender is #1 predictor → Females survived at 74% vs males at 19%

Class matters → 1st class: 62% survival, 3rd class: 24% survival

Fare correlated → Higher fare = higher survival (r = +0.26)

Solo travelers at risk → is_alone shows r = -0.20 with survival

Model accuracy: 73% → 131 out of 179 test passengers correctly predicted

# Preprocessing
X_sc = StandardScaler().fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_sc, y, test_size=0.2, random_state=42, stratify=y
)

# Train
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)

# Result: 73% Accuracy

# 🔮 Future Improvements

 Try Random Forest or XGBoost for better accuracy
 
 Apply k-fold Cross Validation
 
 Use GridSearchCV for hyperparameter tuning
 
 Extract passenger titles from Name column (Mr., Mrs., Miss)
 
 Apply SMOTE to handle class imbalance


# 📊 Sample Outputs
EDA Dashboard

Four-chart panel showing survival count, survival by sex, by class, and age distribution.

Correlation Heatmap

sex_enc = -0.49 (strongest), pclass = -0.34, fare = +0.26

Confusion Matrix

True Negatives: 85 (correctly predicted non-survivors)

True Positives: 46 (correctly predicted survivors)

Overall Accuracy: 73%


# 👤 About
Level: Intermediate Data Science

Duration: 6 weeks, 1 hour/day

Tools: Python, Pandas, Seaborn, Scikit-learn, Jupyter

# 📄 License
This project is open source and available under the MIT License.

⭐ If you found this helpful, please star the repository!
