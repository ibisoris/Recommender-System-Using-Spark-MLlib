# 🎮 Steam Game Recommender System using Spark MLlib

## 📖 Project Overview
This project builds a **collaborative filtering recommender system** using **Apache Spark MLlib** on a dataset collected from the **Steam gaming platform**.  
The system leverages **implicit feedback data** (purchases and playtime) to uncover latent characteristics of both users and games, generating **personalized game recommendations**.

Developed in **Databricks Community Edition**, the project demonstrates **big data processing, distributed machine learning, and recommendation system design**.

---

## 🗂️ Dataset
- **Source**: Steam game interaction dataset.  
- **Features**:
  - `member_id` → unique user ID  
  - `game` → game title  
  - `behavior` → purchase or play indicator  
  - `value` → implicit rating (hours played or purchase flag)  
- Data was cleaned, transformed, and formatted into a **user–item–rating matrix** for ALS.

---

## 🔑 Methodology
The project follows a standard **machine learning pipeline**:

1. **Data Import & Preparation**
   - Load CSV into Spark DataFrame.
   - Clean and preprocess data (remove nulls, rename columns, encode users/games).
   - Split dataset into **training** and **testing** sets.

2. **Model Training**
   - Implement **Alternating Least Squares (ALS)** from Spark MLlib.  
   - ALS is chosen for its suitability for **implicit feedback recommendation** tasks.

3. **Evaluation**
   - Use **Root Mean Square Error (RMSE)** to evaluate prediction accuracy.  
   - Hyperparameter tuning performed on:
     - Rank (latent factors)  
     - Regularization parameter (λ)  
     - Number of iterations  

4. **Recommendation Generation**
   - Top-N game recommendations generated for each user.  
   - Predictions compared against test data.

---

## ✨ Features
- **Big Data Handling**: Uses Apache Spark for distributed processing of large datasets.  
- **Collaborative Filtering**: ALS algorithm captures hidden patterns between users and games.  
- **Implicit Feedback Modeling**: Works with playtime and purchase data instead of explicit ratings.  
- **Hyperparameter Tuning**: Optimizes performance for best RMSE score.  
- **Personalized Recommendations**: Produces tailored suggestions per user.  

---

## ⚙️ Usage

### Prerequisites
- [Databricks Community Edition](https://community.cloud.databricks.com/) OR local Spark setup.  
- Python 3.x with PySpark.

### Running the Notebook
1. Import the provided notebook file into **Databricks**.  
2. Attach to a Spark cluster.  
3. Run cells sequentially:
   - Data import and preprocessing  
   - Model training with ALS  
   - Evaluation and recommendation generation  

### Example Code Snippet
```python
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# Train ALS model
als = ALS(
    userCol="user_id",
    itemCol="game_id",
    ratingCol="value",
    rank=10,
    maxIter=10,
    regParam=0.1,
    coldStartStrategy="drop"
)

model = als.fit(training)

# Evaluate
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="value", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"Root-mean-square error = {rmse}")
```

---

## 📊 Results
- ALS successfully generated **personalized recommendations** for Steam users.  
- Model performance measured by **RMSE**, with tuned hyperparameters improving accuracy.  
- Example recommendation output:  
  - User A → Recommended games: *Game X, Game Y, Game Z*  
  - User B → Recommended games: *Game P, Game Q*  

---

## 🔒 Data Privacy & Ethics
- Dataset used for **educational purposes** only.  
- No personally identifiable information (PII) is included.  
- Recommendations respect the principle of **implicit user behavior analysis**.

---

## ✅ Conclusion
This project demonstrates how **Spark MLlib’s ALS algorithm** can be applied to real-world datasets to build a scalable recommender system.  
It shows:
- How to preprocess large datasets in Spark.  
- How to implement and tune ALS for collaborative filtering.  
- How to evaluate recommendations using RMSE.  

**Future Improvements**:
- Integrate content-based filtering (hybrid recommender).  
- Deploy as an API for real-time recommendations.  
- Extend evaluation with ranking metrics (Precision@K, MAP).  

---

## 👨‍💻 Author
**Ibinabo Orifama**  
_Module: Big Data Tools & Techniques (BDTT) – Task 2_  
