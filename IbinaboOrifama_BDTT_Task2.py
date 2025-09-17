# Databricks notebook source
# MAGIC %md
# MAGIC ## Recommender System Using Spark MLlib on Steam Dataset
# MAGIC
# MAGIC **Name: Ibinabo Orifama**
# MAGIC
# MAGIC **StudentID: 00749582**
# MAGIC
# MAGIC ### Introduction
# MAGIC This task involves building a collaborative filtering recommender system with Apache Spark's MLlib on a dataset collected from Steam, an online video game distribution platform. The dataset comprises implicit feedback in the form of game purchases and playtime for various users, which is use to determine user preferences. We want to identify latent characteristics that represent people and games using the Alternating Least Squares (ALS) algorithm, which will allow us to make personalized game suggestions. The assignment entails using MLflow for 
# MAGIC
# MAGIC - data preparation, 
# MAGIC - model training, 
# MAGIC - evaluating the model,
# MAGIC - hyperparameter tuning,
# MAGIC - recommendation, and 
# MAGIC - experiment tracking in the Databricks environment.
# MAGIC
# MAGIC In this task, several approaches were explored and experimented for training and evaluating the collaborative filtering recommender system, which including 
# MAGIC
# MAGIC - using only play behavior 
# MAGIC - combining purchase and play behavior and 
# MAGIC - using purchase and play behavior with log scale playtimes to reduce the effect of outliers.
# MAGIC
# MAGIC The diiferent approaches were used to be able to identify the configuration that produce the best RMSE value.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Import
# MAGIC **Import mlflow**
# MAGIC
# MAGIC The first code snippet automates the tracking of Spark MLlib experiments using MLflow. By importing MLflow and activating ```mlflow.pyspark.ml.autolog()```, the system may automatically log model parameters, metrics, and artifacts during training, eliminating the need for manual logging. Setting the MLflow logger to ERROR also guarantees that only important messages are displayed, resulting in a cleaner and more focused output.
# MAGIC
# MAGIC **Loading the Dataset**
# MAGIC
# MAGIC The code sample ```df = spark.read.csv("/FileStore/tables/steam_200k.csv", header=False, inferSchema=True)``` loads the Steam user interaction dataset into a Spark DataFrame and prepares it for analysis. It reads the CSV file without a header and automatically infers the data type of each column. The default column names (_c0 to _c3) are then replaced with more descriptive labels: **member_id** (unique user ID), **game** (game title), **behaviour** (either 'purchase' or 'play'), and **value** (1.0 for purchases or number of hours played). Finally, the first five rows are shown to ensure that the data was properly loaded and renamed.

# COMMAND ----------

#Import mlflow and Autologs ML runs
import mlflow
import logging

# Set the logging level for MLflow to ERROR to suppress info and warning messages in the output
logging.getLogger('mlflow').setLevel(logging.ERROR)

# Enable Spark MLlib autologging
mlflow.pyspark.ml.autolog()

# COMMAND ----------

from pyspark.sql import SparkSession

# Load the dataset
df = spark.read.csv("/FileStore/tables/steam_200k.csv", header=False, inferSchema=True)

# Rename columns for clarity
df = df.withColumnRenamed("_c0", "member_id") \
       .withColumnRenamed("_c1", "game") \
       .withColumnRenamed("_c2", "behaviour") \
       .withColumnRenamed("_c3", "value")

# View the first few rows
df.show(10)


# COMMAND ----------

# MAGIC %md
# MAGIC In the Spark DataFrame output shown above, each row represent a user action in a single game. The rows are unique intereaction of purchase and hours played. The **member_id** column identifies the user, **game** provides the game's title, **behaviour** describes whether the user purchased or played the game, and **value** displays the numeric outcome of that behavior - 1.0 for a purchase and number of hours spent for a play action. This format collects implicit feedback that can be used to train a recommender system by examining trends in how users engage with various games.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploratory Data Analysis
# MAGIC
# MAGIC The line ```df.printSchema()``` below displays the structure (schema) of the DataFrame. It displays the name of each column, as well as its data type and whether or not null values are allowed. In this dataset, the member_id: integer, game: string, behaviour: string, value: double. This helps in understand how Spark has interpreted the dataset. 
# MAGIC
# MAGIC The line ```df.count()``` below returns the total number of rows or records in the DataFrame df. It does a full scan of the dataset to count all entries, which is beneficial for quickly determining the dataset size. In the context of the Steam dataset, this indicates how many user-game interactions (both purchases and play sessions) are captured in the data. The output of the count is 200000
# MAGIC
# MAGIC The line ```df.groupBy("behavior").count().show()``` groups the DataFrame by behaviour column and counts the number of times each unique behavior appears. In this dataset, the typical behaviour values are 'purchase' and 'play'. This command helps to analyze the distribution of user actions by displaying the number of purchase and play events present.
# MAGIC
# MAGIC This line of code ```df.select("member_id").distinct().count(), df.select("game").distinct().count()```computes the total number of distinct users and unique games in the dataset by picking distinct values from the member_id and game columns. It contributes to determining the scale of the user-item interaction matrix, which is essential for developing a recommender system. Knowing how many users and games are involved allows you to analyze data sparsity, prepare for computational resources, and anticipate potential obstacles such as cold-start problems, which occur when new users or games have little interaction history.
# MAGIC
# MAGIC The line ```df.select([_sum(col(c).isNull().cast("int")).alias(c + "_nulls") for c in df.columns]).show()``` checks for null values in the dataset.

# COMMAND ----------

#Check the schemas
df.printSchema()

# COMMAND ----------

# Count total records
df.count()

# COMMAND ----------

# Check the distribution of behaviours
df.groupBy("behaviour").count().show()


# COMMAND ----------

# MAGIC %md
# MAGIC The output above displays the results indicating that there are 129,511 records of customers purchasing games and 70,489 records of users playing games. This indicates that the dataset contains more purchase actions than play actions. This knowledge is useful for understanding the data distribution and determining how to use it in a recommender system.

# COMMAND ----------

# Total number of users and games
df.select("member_id").distinct().count(), df.select("game").distinct().count()

# COMMAND ----------

# MAGIC %md
# MAGIC From the output above, the distinct users is 12393 and unique game is 5155 in the steam dataset.

# COMMAND ----------

from pyspark.sql.functions import col, sum as _sum

# Count nulls in each column
df.select([_sum(col(c).isNull().cast("int")).alias(c + "_nulls") for c in df.columns]).show()



# COMMAND ----------

# MAGIC %md
# MAGIC The result above shows that there are no null values in the dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC **Using Spark SQL for analysis**
# MAGIC
# MAGIC The line ```df.createOrReplaceTempView("steam")``` registers the Spark DataFrame df as a temporary SQL view called "steam". This allows the running of SQL queries directly on the DataFrame with Spark SQL.

# COMMAND ----------

#Creating a temporary view for SQL queries
df.createOrReplaceTempView("steam")


# COMMAND ----------

# MAGIC %md
# MAGIC **Visualization of the 10 most played games**
# MAGIC
# MAGIC The SQL query below extracts the top ten most played games in terms of total playtime from the temporary view steam. It filters the data to only include rows with the behaviour 'play' and then groups the remaining records by the game column. It sums the value column for each game to calculate the overall number of hours played. The results are arranged in descending order by total_playtime, with the highest cumulative play hours at the top. Using user activity, this query identifies the most engaging or popular games in the dataset.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Select the top 10 most played games based on total playtime
# MAGIC SELECT game, SUM(value) AS total_playtime  
# MAGIC FROM steam
# MAGIC WHERE behaviour = 'play'                   -- Filter for rows where the user played the game
# MAGIC GROUP BY game                              -- Group by game title
# MAGIC ORDER BY total_playtime DESC               -- Sort games by total playtime in descending order
# MAGIC LIMIT 10                                   -- Return only the top 10 games
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC From the visualization, **Dota 2** has the most cumulative playtime, nearing about 1 million hours, indicating that it is by far the most engaging or popular game among users in this dataset. Other games with high playtime include **Counter-Strike: Global Offensive**, **Team Fortress 2**, and **Counter-Strike**, all of which have strong multiplayer communities. This information aids in the identification of games with the most devoted or active player bases and can be useful in proposing trending or highly engaging titles to new players.

# COMMAND ----------

# MAGIC %md
# MAGIC **Visualization of 10 most purchased games**
# MAGIC
# MAGIC The SQL query below returns the top ten most purchased games by adding purchase counts (values) for each game with the behaviour 'purchase' and sorting the result in descending order.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Select the top 10 most purchased games based on total puchased
# MAGIC SELECT game, SUM(value) AS total_Purchased
# MAGIC FROM steam
# MAGIC WHERE behaviour = 'purchase'                    -- Filter for rows where the user purchased the game
# MAGIC GROUP BY game                                   -- Group by game title
# MAGIC ORDER BY total_purchased DESC                   -- Sort games by total purchased in descending order
# MAGIC LIMIT 10                                        -- Return only the top 10 games

# COMMAND ----------

# MAGIC %md
# MAGIC The chart above displays the top ten most purchased games, with **Dota 2** leading by a significant margin, followed by **Team Fortress 2** and **Unturned**, showing the most popular titles based on purchase count.

# COMMAND ----------

# MAGIC %md
# MAGIC **Checking for outliers**

# COMMAND ----------

#  
# Filter only play behavior
play_df = df.filter(df.behaviour == 'play')

# Describe playtime stats
play_df.select("value").describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC The playtime statistics above show a highly skewed distribution: the average playtime is approximately 48.88 hours, but the standard deviation is very high (229.34), and the maximum playtime is 11,754 hours. Because of the vast range, which includes many small values and a few extreme outliers, log transformation will be a highly successful strategy to use. Applying log1p() compresses these huge numbers, reduces skewness, and makes the data more suited for recommender system training.
# MAGIC
# MAGIC Before applying the log transform strategy, the data will be experimented first without the log to see the performance and RMSE values.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Preprocessing 
# MAGIC Data preprocessing is the process of converting raw data into a clean, structured, and useful format before passing it through a machine learning model or analytic pipeline. It guarantees that the data is consistent, full, and in the appropriate format for modeling.

# COMMAND ----------

# MAGIC %md
# MAGIC **Case 1 - Using play behaviours only**
# MAGIC
# MAGIC The line of code below generates a new DataFrame named play_df and filters the original df to contain only rows with the behaviour 'play', indicating that the user has played the game. It then selects just three columns: member_id, game, and value. The value column, which displays the number of hours a user has spent playing the game, is renamed rating. This reformatted DataFrame is designed exclusively for training a collaborative filtering model (such as ALS). Play behaviour indicates true involvement because it tracks how long a user spent playing a game, whereas purchases do not always imply interest. Using play data allows the ALS model to learn user preferences more accurately, which leads to better recommendations.
# MAGIC

# COMMAND ----------

play_df = df.filter(df.behaviour == "play") \
               .select("member_id", "game", "value") \
               .withColumnRenamed("value", "rating")


# COMMAND ----------

# MAGIC %md
# MAGIC **Adding unique user_id and game_id**
# MAGIC
# MAGIC This code sample indexes the **member_id** and **game** columns using PySpark's StringIndexer and converts them into numerical columns **user_id** and **game_id**, which are required for training collaborative filtering models such as ALS. By default, these indexed columns are of DoubleType, thus the code employs the ```withColumn()``` function in conjunction with ```col()``` to explicitly cast both **user_id** and **game_id** to IntegerType for simpler and more exact data processing. Finally, it shows the first five rows of the modified DataFrame to ensure that the indexing and type conversion were successful.

# COMMAND ----------

from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col

# Index users and games
user_indexer = StringIndexer(inputCol="member_id", outputCol="user_id")
game_indexer = StringIndexer(inputCol="game", outputCol="game_id")

# Fit and transform
play_df = user_indexer.fit(play_df).transform(play_df)
play_df = game_indexer.fit(play_df).transform(play_df)

#Convert datatype to integer
play_df = play_df.withColumn("user_id", col("user_id").cast("int")) \
                       .withColumn("game_id", col("game_id").cast("int"))


# Show transformed data
play_df.select("member_id", "user_id", "game", "game_id", "rating").show(10)


# COMMAND ----------

# MAGIC %md
# MAGIC In the result above, all interactions are associated with a single user (member_id = 151603712), who is assigned the indexed user_id = 585. Each row represents a game that the user has played, with the rating column showing the number of hours spent playing that game. This indexed and organized format is necessary for collaborative filtering because it enables the ALS algorithm to learn user preferences based on numeric user and item IDs.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Training - Case 1
# MAGIC **Split the Dataset**:
# MAGIC
# MAGIC The line ```play_df.randomSplit([0.8, 0.2], seed=42)``` splits the play_df DataFrame into two subsets: training and testing datasets, using an 80/20 split. The ```randomSplit([0.8, 0.2], seed=42)``` technique ensures that 80% of the data is allocated randomly to the training set and 20% to the test set. The seed=42 option ensures reproducibility by utilizing a fixed random seed, which means the split will be the same every time the code is run. This stage is critical for evaluating machine learning models like ALS since it allows you to train the model on a subset of the data and then test its performance on unknown data.
# MAGIC
# MAGIC **ALS Algorithm**:
# MAGIC
# MAGIC The code below uses Alternating Least Squares (ALS) technique to train a recommendation model, which is ideal for collaborative filtering with implicit feedback. It instructs ALS to use the user_id, game_id, and rating columns, with rating indicating how many hours a user has spent playing a game. The model is configured to accept implicit preferences, remove rows with cold-start concerns, utilize non-negative factors, and run for 5 iterations with a regularization value of 0.01 and a latent factor rank of 10. The model is then trained on the training dataset with ```als.fit(training)```, which allows it to learn patterns in user-game interactions and generate suggestions.

# COMMAND ----------

# Split the dataset
(training, test) = play_df.randomSplit([0.8, 0.2], seed=42)


# COMMAND ----------

# use the ALS Algorithm to train the model
from pyspark.ml.recommendation import ALS

# Initialize ALS
als = ALS(
    userCol="user_id", itemCol="game_id", ratingCol="rating", implicitPrefs=True,
    coldStartStrategy="drop", nonnegative=True, maxIter=5, regParam=0.01, rank=10, seed=42
)

# Train model
model = als.fit(training)

# COMMAND ----------

# MAGIC %md
# MAGIC From the above output, a model is connected to one experiment. An experiment is a collection of runs. It serves as a folder that stores and arranges runs to train and test model

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluating the Model
# MAGIC **Make prediction**
# MAGIC
# MAGIC The ```.transform()``` method is used in the code below, to predict on the test dataset using the trained ALS model. The output is a new DataFrame called predictions, which retains the original columns from the test data (such as user_id, game_id, and rating), as well as an additional column called prediction, which has the model's estimated rating (i.e., expected playtime) for each user-game combination. The ```predictions.show()``` command shows the first few rows, allowing you to see how well the model predicts user preferences based on test data.

# COMMAND ----------

# make predictions using the transform() method
predictions = model.transform(test)

predictions.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC The displayed output above shows the outcomes of utilizing the trained ALS model to make predictions on test data. Each row represents a user-game interaction and contains the original values: user_id, game_id, and actual rating (number of hours played), as well as the newly created prediction column. The model internally uses the numeric indices user_id and game_id.
# MAGIC The prediction column contains the model's estimated rating (i.e., expected playtime) for a particular user-game pair. For example, for **member_id** 2083767 or **user_id** 471 and the game **CastleStorm**, the actual rating is 0.7 hours played, but the model projected roughly 0.26 hours

# COMMAND ----------

# MAGIC %md
# MAGIC **Evaluating Accuracy  with RMSE**
# MAGIC
# MAGIC This code below assesses the trained ALS recommendation model's efficacy by computing the Root Mean Square Error (RMSE), which is the average difference between predicted and real playtime values in the test dataset. The prediction column (model output) and the rating column (actual values) are compared using PySpark's RegressionEvaluator. A lower RMSE shows that the model's predictions are closer to the actual values, implying better performance, whereas a higher RMSE suggests less accurate predictions. This stage is critical for assessing how effectively the model generalizes to previously unseen data.
# MAGIC

# COMMAND ----------

# Check the effectiveness of the model
from pyspark.ml.evaluation import RegressionEvaluator

# Evaluate using RMSE
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

rmse = evaluator.evaluate(predictions)
print(f"[Play Behavior]- Root-mean-square error = {rmse:.4f}")
print('rmse = %g ' % (rmse))


# COMMAND ----------

# MAGIC %md
# MAGIC The results show RMSE value is 211.1568, indicating that the predicted playtime values differ from the actual playtimes by around 211 hours. While this may appear to be a high number, it is crucial to realize that huge variances are prevalent in implicit feedback datasets such as Steam (where some users may play games for hundreds or thousands of hours), which might inflate the RMSE. This value provides a quantitative measure of model performance; however, in recommendation systems, the ranking quality is frequently more important than the projected rating. The next step is to combine both play and purchased behaviour to see if RMSE value will reduce or improve.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Preprocessing 
# MAGIC **Case 2 - Using Purchase and Play Behaviour without log**
# MAGIC
# MAGIC Before combining purchase and play behaviour, it's crucial to note that each presents a unique but complimentary view of user preferences. Purchase behaviour reveals a user's interest or purpose, whereas play behaviour reflects actual engagement and enjoyment. By conbining both, building a more robust rating signal that captures both the decision to purchase a game and the degree to which it was enjoyed becomes possible. This method helps the model understand more complex patterns and increases the quality of personalized recommendations.
# MAGIC
# MAGIC The dataset is first prepared by assigning a constant rating (e.g., 5.0) to purchases and the actual playtime value to plays. These are all integrated into a single rating column. User and game identities are then indexed into numeric values using StringIndexer, and the resulting data is converted to integer types. 

# COMMAND ----------

from pyspark.sql.functions import when

# Combine behaviors into a single 'rating' column
full_df = df.withColumn(
    "rating",
    when(df.behaviour == "purchase", 1.0).otherwise(df.value)
)


# COMMAND ----------

from pyspark.ml.feature import StringIndexer

#Convert 'member_id' column to a numeric index for ALS
user_indexer = StringIndexer(inputCol="member_id", outputCol="user_id")

#Convert 'game' names column to a numeric index for ALS
game_indexer = StringIndexer(inputCol="game", outputCol="game_id")

#Fit the user indexer and apply it to the full dataset
indexed_df = user_indexer.fit(full_df).transform(full_df)

#Fit the game indexer and apply it to the user-indexed data
indexed_df = game_indexer.fit(indexed_df).transform(indexed_df)



# COMMAND ----------

from pyspark.sql.functions import col

# Convert user_id and game_id to integers
combined_df = indexed_df.select("user_id","game", "game_id", "rating") \
                        .withColumn("user_id", col("user_id").cast("int")) \
                        .withColumn("game_id", col("game_id").cast("int"))

#Display the first 10 rows with the original and indexed values
combined_df.select("user_id","game", "game_id", "rating").show(10)


# COMMAND ----------

# MAGIC %md
# MAGIC In the table above, For example, user_id 635 purchased **The Elder Scrolls** (with a 1.0 rating) and played it for 273.0 hours. This trend is consistent across other games, such as **Fallout 4** (1.0 for purchase, 87.0 for play) and Spore (1.0 for purchase, 14.9 for play), indicating that each interaction type is logged independently, providing deeper information to the recommendation model.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Training - Case 2
# MAGIC The dataset is then split into training and testing sets, and an ALS model is trained using implicit feedback parameters. Finally, RMSE is used to assess how accurate the model predicts user preferences based on both behaviours.

# COMMAND ----------

#Split the combined_df dataset
(training_combined, test_combined) = combined_df.randomSplit([0.8, 0.2], seed=42)


# COMMAND ----------

# use the ALS Algorithm to train the model
from pyspark.ml.recommendation import ALS

#initialize ALS
als_combined = ALS(userCol="user_id", itemCol="game_id", ratingCol="rating", implicitPrefs=True, coldStartStrategy="drop",nonnegative=True, maxIter=5, regParam=0.01, rank=10
)

#Train the combined model
model_combined = als_combined.fit(training_combined)


# COMMAND ----------

# MAGIC %md
# MAGIC **Evaluating Accuracy  with RMSE**

# COMMAND ----------

# Import RegressionEvaluator for evaluating model accuracy
from pyspark.ml.evaluation import RegressionEvaluator

# Use the trained model to generate predictions on the test dataset
predictions_combined = model_combined.transform(test_combined)

# Initialize an RMSE evaluator to measure prediction error
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

# Calculate and print the RMSE score for the combined behaviour model
rmse_combined = evaluator.evaluate(predictions_combined)
print(f"[Combined Behavior] RMSE: {rmse_combined:.4f}")


# COMMAND ----------

# MAGIC %md
# MAGIC **Explanation of Result**
# MAGIC
# MAGIC The combined behaviour model had a Root Mean Square Error (RMSE) of 123.2005, which was significantly better than the play-only model, which had an RMSE of around 211.1568. This suggests that combining purchase and play behaviours yields a little better result. The final case will be to apply log scale to the playtime behaviour to look for better improvement of the RMSE values before making recommendations.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Preprocessing 
# MAGIC **CASE 3: Combine Purchase and Play with Log-Scaled Playtime**
# MAGIC
# MAGIC The first two cases above yielded relatively high RMSE values of 211.157 and 123.1619, respectively, when play-only behavior was used and combined purchase and play (without transformation). The significant RMSE, especially in the play-only model, shows that excessive playtime outliers may have affected model performance. To solve this, a third experiment was carried out that integrated purchase and play behavior but used log-scaling for playtime. As seen in the following section, this modification reduces the impact of large outliers and ensures more balanced input for training the recommendation model.
# MAGIC
# MAGIC

# COMMAND ----------


from pyspark.sql.functions import when, col, log1p

# Create a rating column: 1.0 for purchases, log-scaled playtime otherwise
combined_df_log = df.withColumn(
    "rating",
    when(col("behaviour") == "purchase", 1.0)
    .otherwise(log1p(col("value")))
)

# Check a few rows to verify
combined_df_log.select("member_id", "game", "behaviour", "value", "rating").show(10)

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

# Index user and game columns
user_indexer = StringIndexer(inputCol="member_id", outputCol="user_id")
game_indexer = StringIndexer(inputCol="game", outputCol="game_id")
indexed_log_df = user_indexer.fit(combined_df_log).transform(combined_df_log)
indexed_log_df = game_indexer.fit(indexed_log_df).transform(indexed_log_df)

# Cast user_id and game_id to int
indexed_log_df = indexed_log_df.withColumn("user_id", col("user_id").cast("int"))
indexed_log_df = indexed_log_df.withColumn("game_id", col("game_id").cast("int"))

#Display the first 10 rows with the original and indexed values
indexed_log_df.select("user_id","game", "game_id", "rating").show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Training - Case 3

# COMMAND ----------

# use the ALS Algorithm to train the model
from pyspark.ml.recommendation import ALS

# Split the data
training_log, test_log = indexed_log_df.randomSplit([0.8, 0.2], seed=42)

# Train ALS model using log-scaled combined data
als_log = ALS(
    userCol="user_id",
    itemCol="game_id",
    ratingCol="rating",
    implicitPrefs=True,
    coldStartStrategy="drop",
    nonnegative=True,
    maxIter=5,
    rank=15,
    regParam=0.001
)

model_log = als_log.fit(training_log)

# COMMAND ----------

# MAGIC %md
# MAGIC **Evaluating Accuracy  with RMSE**

# COMMAND ----------

# Import RegressionEvaluator for evaluating model accuracy
from pyspark.ml.evaluation import RegressionEvaluator

# Predict and evaluate
predictions_log = model_log.transform(test_log)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse_log = evaluator.evaluate(predictions_log)
print(f"[Combined (Log-Scaled)] RMSE: {rmse_log:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hyperparameter Tuning - Case 3
# MAGIC Before using hyperparameter tuning, it is important to note that ALS performance can vary greatly based on the values of key parameters such as rank, regParam, and alpha. Hyperparameter tuning entails evaluating various combinations of these parameters to determine which arrangement produces the best accurate model.

# COMMAND ----------

# -------------------------------------------------------
# Hyperparameter Tuning with MLflow for Log-Scaled Model
# -------------------------------------------------------

import mlflow
import mlflow.spark

# Define the grid of hyperparameters to try
ranks = [5, 10, 15]                 # Number of latent features
reg_params = [0.001, 0.01, 0.1]     # Regularization parameters

# Initialize variables to track the best model
best_rmse = float("inf")           # Start with a very high RMSE
best_params = {}                   # Store best-performing hyperparameters
best_model = None                  # Store the best ALS model

# Loop through each combination of rank and regParam
for rank in ranks:
    for reg in reg_params:
        with mlflow.start_run():  # Start an MLflow run to log this experiment
            # Initialize the ALS model with current hyperparameters
            als = ALS(
                userCol="user_id",
                itemCol="game_id",
                ratingCol="rating",
                implicitPrefs=True,          # Use implicit feedback (e.g., playtime)
                coldStartStrategy="drop",    # drop NaN predictions by dropping unseen data
                nonnegative=True,            # Force non-negative latent factors
                maxIter=5,                   # Number of ALS iterations
                rank=rank,                   # Latent factors
                regParam=reg                 # Regularization strength
            )

            # Train the ALS model using the training set
            model = als.fit(training_log)

            # Make predictions on the test set
            predictions = model.transform(test_log)

            # Evaluate the model using RMSE
            evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
            rmse = evaluator.evaluate(predictions)

            # Log parameters and metrics to MLflow
            mlflow.log_param("rank", rank)
            mlflow.log_param("regParam", reg)
            mlflow.log_metric("rmse", rmse)
            mlflow.spark.log_model(model, "ALSModel")  # Log the trained model

            # Print the result of this run
            print(f"[rank={rank}, regParam={reg}] → RMSE: {rmse:.4f}")

            # Update the best model if this one has lower RMSE
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {"rank": rank, "regParam": reg}
                best_model = model

# Print the best hyperparameter combination and its RMSE
print("\n-Best Log-Scaled Model Parameters-:")
print(f"Rank: {best_params['rank']}, RegParam: {best_params['regParam']}")
print(f"Lowest RMSE: {best_rmse:.4f}")


# COMMAND ----------

# MAGIC %md
# MAGIC The results of a manual hyperparameter tuning experiment with MLflow for a log-scaled ALS (Alternating Least Squares) model shown above, explored various combinations of rank and regParam values to determine the best-performing model. The performance metric employed was the Root Mean Square Error (RMSE). Among all runs, the combination rank=15 and regParam=0.001 had the lowest RMSE of 1.4618, indicating that it is the most accurate model in the experiment. This findings validates the use of log-scaling for playtime data the best case, as the results will give higher prediction accuracy than previous cases.
# MAGIC
# MAGIC **Link for tracking experiment with MLflow**
# MAGIC
# MAGIC https://community.cloud.databricks.com/ml/experiments/3968102192852453?viewStateShareKey=f3d8b61be25cc02a484bd908ea2fa5af357aa57172239dee136c59c935bee414&compareRunsMode=TABLE&o=2812931347034770

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generating Recommendation
# MAGIC **10 Recommended Games for each user**
# MAGIC
# MAGIC The code below uses the trained ALS model to generate the top ten recommended games for each user by using ```recommendForAllUsers(10)```, which returns a DataFrame with each user_id and a list of game suggestions with projected ratings. The ```display(user_recs.head(5))``` line then shows the first five rows of this DataFrame, allowing you to evaluate how the model offers unique games for distinct users based on their prior activity.

# COMMAND ----------

# Recommend top 10 games for each user
user_recs = best_model.recommendForAllUsers(10)

# Show first few recommendations
display(user_recs.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC In the result above, each row corresponds to a distinct user_id and includes a recommendations column, which is an array of dictionaries. Each dictionary has a game_id and a rating, which indicate the model's prediction of how much the user would enjoy the game. For example, user 26 is more likely to prefer games with game_ids 22, 63, and 65, which have the highest expected ratings of 1.35, 1.34, and 1.31, respectively. These scores show the ALS model's confidence in how well each game matches the user's preferences based on previous user activity patterns.

# COMMAND ----------

# MAGIC %md
# MAGIC **Explode Recommendation column and get game names**
# MAGIC
# MAGIC This code convert the nested recommendation structure to a readable format and uses StringIndexer to map game_id values back to game names. It pulls each recommended game per user, combines it with the original game titles, and shows the top 10 recommendations sorted by user and predicted score.

# COMMAND ----------

from pyspark.sql.functions import explode, col

# Flatten the recommendation structure
exploded = user_recs.withColumn("rec", explode("recommendations"))
flat_recs = exploded.select("user_id", col("rec.game_id").alias("game_id"), col("rec.rating").alias("score"))

# Generate mapping from game and game_id (manually indexed if not in df)
game_indexer = StringIndexer(inputCol="game", outputCol="game_id")
game_mapping_df = game_indexer.fit(df).transform(df).select("game", "game_id").distinct()

# Map game_id back to game names
final_recs = flat_recs.join(game_mapping_df, on="game_id", how="left")

# Show final recommendations for top users
top_recs = final_recs.select("user_id", "game", "score").orderBy("user_id", "score", ascending=False)
top_recs.show(20, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC The result shows the top game recommendations for users. For example, user 12392 has been recommended games such as Tomb Raider, Metro 2033, and Far Cry 3, with predicted scores showing how much the ALS model thinks each user will enjoy them. Similarly, user 12391 has the same top recommendations, but with slightly better expected scores, such as 0.0388 for Tomb Raider. 

# COMMAND ----------

# MAGIC %md
# MAGIC **Recommendation for a specific user**
# MAGIC
# MAGIC The code below provides game recommendations for a given user by first specifying a user_id (in this case, 0) and then building a single-row DataFrame with that ID. Using the trained ALS model, it uses recommendForUserSubset() to provide the top ten recommended games for that user based on projected preference ratings.

# COMMAND ----------

#Generate Recommendations for a Specific User

# Specify a user_id 
specific_user_id = 0 
#create a dataframe
single_user_df = spark.createDataFrame([(specific_user_id,)], ["user_id"])
#Recommend for single user
user_top_recs = best_model.recommendForUserSubset(single_user_df, 10)
#Display first few result
display(user_top_recs.head(5))



# COMMAND ----------

# MAGIC %md
# MAGIC The ALS model predicts that user 0 will most likely enjoy game 189, which has the highest predicted rating of 3.70, followed by game 260 with a score of 3.38 and game 265 with 3.21. This format displays the top recommended games for a user before converting the IDs to game names for the final output.

# COMMAND ----------

# MAGIC %md
# MAGIC **Explode recommendation column to get game names**
# MAGIC
# MAGIC This code uses explode to convert the nested array of recommended games, resulting in each game appearing in its own row. It then takes the game_id and expected score (rating) for each recommendation and combines them with game_mapping_df to map game_ids back to their original game names.

# COMMAND ----------

# Flatten and map game titles
single_exploded = user_top_recs.withColumn("rec", explode("recommendations"))
single_flat = single_exploded.select("user_id", col("rec.game_id").alias("game_id"), col("rec.rating").alias("score"))
single_final = single_flat.join(game_mapping_df, on="game_id", how="left")

# Show top 10 recommended games for the user
single_final.select("user_id", "game", "score").orderBy("score", ascending=False).show(10, truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC This result shows the top ten recommended games for user_id 0, arranged by predicted preference score. Games with the highest predicted scores, such as Football Manager 2013  and Football Manager 2012, indicate that the user is most likely to enjoy them.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusion
# MAGIC The project successfully developed a collaborative filtering recommender system using the ALS algorithm on Steam user data, which included both play and purchase behaviour. Initial studies with play-only and combined raw data produced relatively high RMSE values, indicating the presence of outliers, particularly those resulting from long play durations. To remedy this, a third experiment was conducted in which playtime was log-transformed, yielding a more normalized rating distribution. This method considerably improved model performance and decreased RMSE, suggesting that log-scaling is a useful tool for dealing with skewed implicit feedback. Hyperparameter tuning and MLflow tracking helped in determining the best model design. Overall, the log-scaled combined behavior approach provided the most accurate and balanced recommendations.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reference
# MAGIC
# MAGIC 1. https://docs.databricks.com/aws/en
# MAGIC
# MAGIC 2. **_Lecture materials - Big Data Tools and Techniques_**
# MAGIC
# MAGIC