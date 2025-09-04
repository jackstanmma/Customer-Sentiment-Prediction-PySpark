# Databricks notebook source
# MAGIC %md
# MAGIC ## **TAO ZHANG and GAURAV NEMANI ,2025, BIG DATA TOOLS**

# COMMAND ----------

# Training data paths
path_products = "/FileStore/tables/products.csv"
path_orders = "/FileStore/tables/orders.csv"
path_order_items = "/FileStore/tables/order_items.csv"
path_payments = "/FileStore/tables/order_payments.csv"
path_reviews = "/FileStore/tables/order_reviews.csv"

# Test data paths            
path_test_products = "/FileStore/tables/Holdout data-20250120/test_products.csv"
path_test_orders = "/FileStore/tables/Holdout data-20250120/test_orders.csv"
path_test_order_items = "/FileStore/tables/Holdout data-20250120/test_order_items.csv"
path_test_payments = "/FileStore/tables/Holdout data-20250120/test_order_payments.csv"

# COMMAND ----------

# MAGIC %md
# MAGIC Creating basetable for both training set and test set
# MAGIC

# COMMAND ----------


from pyspark.sql.functions import *
from pyspark.sql.window import Window

# ========================
# Products Table Processing
# ========================

def process_products(spark, path, is_test=False):
    """
    Process products data:
    1. Load and deduplicate products
    2. Calculate volumetric weight
    3. Drop dimension columns after calculation
    4. Remove rows with null values
    
    """
    # Load products data
    products = spark.read.format("csv")\
        .option("header", "true")\
        .option("inferSchema", "true")\
        .option("escape", "\"")\
        .load(path)
    
    # Remove duplicates and calculate volumetric weight
    products = products.dropDuplicates()
    products = products.withColumn(
        "volumetric_weight", 
        round((col("product_length_cm") * col("product_height_cm") * col("product_width_cm")) / 5000, 2)
    )
    
    # Clean up: drop original dimension columns and null values
    products = products.drop("product_length_cm", "product_height_cm", "product_width_cm")
    products = products.na.drop("all")
    
    if is_test:
        print(f"Test Products count: {products.count()}")
    
    return products

# =====================
# Orders Table Processing
# =====================

def process_orders(spark, path, is_test=False):
    """
    Process orders data:
    1. Convert timestamp columns
    2. Add customer behavior features
    3. Handle missing values in delivery deviation
    4. Clean up unnecessary columns
    
    """
    # Load orders data
    orders = spark.read.format("csv")\
        .option("header", "true")\
        .option("inferSchema", "true")\
        .option("escape", "\"")\
        .load(path)
    
    # Convert timestamp columns
    timestamp_columns = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]
    for col_name in timestamp_columns:
        orders = orders.withColumn(
            col_name, 
            to_timestamp(col(col_name), "yyyy-MM-dd HH:mm:ss")
        )
    
    # Calculate customer behavior metrics
    customer_order_counts = orders.groupBy("customer_id").agg(
        count("order_id").alias("order_count"),
        min("order_purchase_timestamp").alias("first_order_date")
    )
    
    # Join and add features
    orders = orders.join(customer_order_counts, "customer_id")
    orders = orders.withColumn(
        "is_repeat_customer", 
        when(col("order_count") > 1, 1).otherwise(0)
    ).withColumn(
        "days_since_first_order",
        datediff(col("order_purchase_timestamp"), col("first_order_date"))
    ).withColumn(
        "Delivery_Date_Deviation", 
        datediff(col("order_estimated_delivery_date"), col("order_delivered_customer_date"))
    )
    
    # Handle missing values in delivery deviation
    orders = orders.withColumn(
        "Delivery_Date_Deviation_Missing", 
        when(col("Delivery_Date_Deviation").isNull(), 1).otherwise(0)
    )
    orders = orders.na.fill(-9999, subset=["Delivery_Date_Deviation"])
    
    # Drop unnecessary columns and null values
    columns_to_drop = [
        "customer_id", "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date", "order_count", "first_order_date"
    ]
    orders = orders.drop(*columns_to_drop)
    orders = orders.na.drop("any")
    
    if is_test:
        print(f"Test Orders count: {orders.count()}")
    
    return orders

# ==========================
# Order Items Table Processing
# ==========================

def process_order_items(spark, path, products_df, is_test=False):
    """
    Process order items data:
    1. Join with products data
    2. Calculate order-level aggregations
    3. Add price/shipping ratio
    
    """
    # Load order items data
    items = spark.read.format("csv")\
        .option("header", "true")\
        .option("inferSchema", "true")\
        .option("escape", "\"")\
        .load(path)
    
    # Join with products
    items_product = items.join(products_df, ["product_id"], "inner")
    
    # Calculate order-level aggregations
    items_product = items_product.groupBy("order_id").agg(
        count("order_item_id").alias("item_count"),
        sum("price").alias("total_price"),
        sum("shipping_cost").alias("total_shipping_cost"),
        avg("product_name_lenght").alias("avg_name_length"),
        avg("product_description_lenght").alias("avg_description_length"),
        sum("product_photos_qty").alias("total_photos"),
        sum("product_weight_g").alias("total_weight_g"),
        sum("volumetric_weight").alias("total_volumetric_weight"),
        concat_ws(", ", collect_set("product_category_name")).alias("categories"),
        countDistinct("product_category_name").alias("unique_category_count")
    )
    
    # Add price/shipping ratio
    items_product = items_product.withColumn(
        "price_shipping_ratio",
        when(col("total_shipping_cost") != 0, 
             round(col("total_price") / col("total_shipping_cost"), 2))
        .otherwise(0)
    )
    
    if is_test:
        print(f"Test Order Items count: {items_product.count()}")
    
    return items_product

# ==========================
# Payments Table Processing
# ==========================

def process_payments(spark, path, is_test=False):
    """
    Process payments data:
    1. Drop unnecessary columns
    2. Aggregate payment information at order level
    
    """
    # Load payments data
    payments = spark.read.format("csv")\
        .option("header", "true")\
        .option("inferSchema", "true")\
        .option("escape", "\"")\
        .load(path)
    
    # Drop unnecessary columns and aggregate
    payments = payments.drop("payment_sequential")
    payments = payments.groupBy("order_id").agg(
        concat_ws(", ", collect_set("payment_type")).alias("payment_type"),
        max("payment_installments").alias("max_installments"),
        countDistinct("payment_type").alias("unique_payment_type"),
        sum("payment_value").alias("order_payment")
    )
    
    if is_test:
        print(f"Test Payments count: {payments.count()}")
    
    return payments

# ==========================
# Reviews Table Processing
# ==========================

def process_reviews(spark, path):
    """
    Process reviews data:
    1. Keep only the most recent review per order
    2. Convert scores to binary sentiment
    3. Drop unnecessary columns
    
    """
    # Load reviews data
    reviews = spark.read.format("csv")\
        .option("header", "true")\
        .option("inferSchema", "true")\
        .option("escape", "\"")\
        .load(path)
    
    # Drop unnecessary columns
    reviews = reviews.drop("review_id")
    
    # Keep only the most recent review per order
    window_spec = Window.partitionBy("order_id")\
        .orderBy(col("review_answer_timestamp").desc())
    
    reviews = reviews.withColumn("row_number", row_number().over(window_spec))\
        .filter(col("row_number") == 1)\
        .drop("row_number", "review_creation_date", "review_answer_timestamp")
    
    # Convert scores to sentiment
    reviews = reviews.withColumn(
        "review",
        when((col("review_score") >= 4), "positive").otherwise("negative")
    ).drop('review_score')
    
    return reviews

# ==========================
# Main Processing Pipeline
# ==========================

def create_basetable(spark, is_test=False):
    """
    Main function to create the final basetable by:
    1. Processing all individual tables
    2. Joining them together
    3. Saving the final result

    """
    if is_test:
        print("Processing test dataset...")
        products_df = process_products(spark, path_test_products, is_test=True)
        orders_df = process_orders(spark, path_test_orders, is_test=True)
        items_df = process_order_items(spark, path_test_order_items, products_df, is_test=True)
        payments_df = process_payments(spark, path_test_payments, is_test=True)
        
        # Create final test basetable (without reviews for test data)
        basetable = orders_df.join(items_df, ["order_id"], "inner")\
            .join(payments_df, ["order_id"], "inner")
        
        # Save the final test basetable
        basetable.write.mode("overwrite").csv("/tmp/test_basetable", header=True)
    else:
        print("Processing training dataset...")
        products_df = process_products(spark, path_products)
        orders_df = process_orders(spark, path_orders)
        items_df = process_order_items(spark, path_order_items, products_df)
        payments_df = process_payments(spark, path_payments)
        reviews_df = process_reviews(spark, path_reviews)
        
        # Create final training basetable
        basetable = orders_df.join(items_df, ["order_id"], "inner")\
            .join(payments_df, ["order_id"], "inner")\
            .join(reviews_df, ["order_id"], "inner")
        
        # Save the final training basetable
        basetable.write.mode("overwrite").csv("/tmp/train_basetable", header=True)
    
    print(f"Final {'test' if is_test else 'training'} basetable count: {basetable.count()}")
    return basetable

# Run the pipeline for both training and test datasets
if __name__ == "__main__":
    # Process training data
    train_basetable = create_basetable(spark, is_test=False)
    
    # Process test data
    test_basetable = create_basetable(spark, is_test=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Training and evaluating Random Forest model

# COMMAND ----------

from pyspark.ml.feature import RFormula, ChiSqSelector, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col, when, lit, count
from pyspark.sql.types import StringType, NumericType

def evaluate_model(predictions, name):
    """Evaluate model performance including AUC"""
    metrics = ['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    results = {}
    
    print(f"\n{'-'*50}")
    print(f"{name} Metrics:")
    print(f"{'-'*50}")
    
    # Calculate confusion matrix counts
    confusion_counts = predictions.groupBy("label", "prediction").agg(count("*").alias("count")).collect()
    
    # Initialize confusion matrix values
    tp = fp = tn = fn = 0
    for row in confusion_counts:
        if row.label == 1.0 and row.prediction == 1.0:
            tp = row["count"]
        elif row.label == 0.0 and row.prediction == 1.0:
            fp = row["count"]
        elif row.label == 0.0 and row.prediction == 0.0:
            tn = row["count"]
        elif row.label == 1.0 and row.prediction == 0.0:
            fn = row["count"]

    # Calculate rates
    total_pos = tp + fn
    total_neg = tn + fp
    total = total_pos + total_neg

    # Metrics calculations
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # True Negative Rate (Specificity)
    tnr = tn / total_neg if total_neg > 0 else 0
    
    # False Positive Rate
    fpr = fp / total_neg if total_neg > 0 else 0
    
    print("\nConfusion Matrix:")
    print(f"{'':15} Predicted Negative  Predicted Positive")
    print(f"Actual Negative  {tn:^16d} {fp:^18d}")
    print(f"Actual Positive  {fn:^16d} {tp:^18d}")
    
    print(f"\nKey Metrics:")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"Specificity: {tnr:.4f}")
    print(f"FPR:         {fpr:.4f}")
    
    # Add AUC-ROC
    binary_evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    auc_roc = binary_evaluator.evaluate(predictions)
    print(f"AUC-ROC:     {auc_roc:.4f}")
    
    results.update({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': tnr,
        'fpr': fpr,
        'auc_roc': auc_roc,
        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
    })
    
    return results

def display_confusion_rates(predictions, name):
    """Display detailed confusion matrix rates"""
    confusion_counts = predictions.groupBy("label", "prediction").agg(count("*").alias("count")).collect()
    
    tp = fp = tn = fn = 0
    for row in confusion_counts:
        if row.label == 1.0 and row.prediction == 1.0:
            tp = row["count"]
        elif row.label == 0.0 and row.prediction == 1.0:
            fp = row["count"]
        elif row.label == 0.0 and row.prediction == 0.0:
            tn = row["count"]
        elif row.label == 1.0 and row.prediction == 0.0:
            fn = row["count"]

    total_pos = tp + fn
    total_neg = tn + fp

    # Calculate all rates
    tnr = tn / total_neg if total_neg > 0 else 0  # True Negative Rate (Specificity)
    tpr = tp / total_pos if total_pos > 0 else 0  # True Positive Rate (Sensitivity)
    fpr = fp / total_neg if total_neg > 0 else 0  # False Positive Rate
    fnr = fn / total_pos if total_pos > 0 else 0  # False Negative Rate
    
    print("\n" + "="*80)
    print(f"DETAILED CONFUSION MATRIX RATES FOR {name}")
    print("="*80)
    
    print("\nConfusion Matrix with Totals:")
    print("-"*80)
    print(f"{'':20} {'Predicted Negative':^20} {'Predicted Positive':^20} {'Row Total':^15}")
    print("-"*80)
    print(f"{'Actual Negative':20} {tn:^20d} {fp:^20d} {tn+fp:^15d}")
    print(f"{'Actual Positive':20} {fn:^20d} {tp:^20d} {tp+fn:^15d}")
    print(f"{'Column Total':20} {tn+fn:^20d} {tp+fp:^20d} {tp+tn+fp+fn:^15d}")
    
    print("\nDetailed Rates:")
    print("-"*80)
    print(f"{'Rate Type':30} {'Formula':^25} {'Value':^20}")
    print("-"*80)
    print(f"{'True Negative Rate (TNR)':30} {'TN/(TN+FP)':^25} {tnr:^20.4f}")
    print(f"{'True Positive Rate (TPR)':30} {'TP/(TP+FN)':^25} {tpr:^20.4f}")
    print(f"{'False Positive Rate (FPR)':30} {'FP/(TN+FP)':^25} {fpr:^20.4f}")
    print(f"{'False Negative Rate (FNR)':30} {'FN/(TP+FN)':^25} {fnr:^20.4f}")
    
    print("\nRate Interpretations:")
    print("-"*80)
    print(f"TNR (Specificity): {tnr:.2%} of negative cases correctly identified")
    print(f"TPR (Sensitivity): {tpr:.2%} of positive cases correctly identified")
    print(f"FPR: {fpr:.2%} of negative cases incorrectly classified as positive")
    print(f"FNR: {fnr:.2%} of positive cases incorrectly classified as negative")
    
    return {'tnr': tnr, 'tpr': tpr, 'fpr': fpr, 'fnr': fnr}

# Load and prepare data
print("Loading data...")
basetable = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("dbfs:/tmp/basetable")

basetable_with_label = basetable.withColumn(
    "label",
    when(col("review") == "positive", 1.0)
    .when(col("review") == "negative", 0.0)
    .otherwise(None)
)

# Feature engineering pipeline
formula = RFormula(
    formula="label ~ . - order_id - review",
    featuresCol="raw_features",
    labelCol="label",
    handleInvalid="skip"
)

selector = ChiSqSelector(
    numTopFeatures=40,
    featuresCol="raw_features",
    outputCol="selected_features",
    labelCol="label"
)

scaler = StandardScaler(
    inputCol="selected_features",
    outputCol="features",
    withStd=True,
    withMean=True
)

feature_pipeline = Pipeline(stages=[formula, selector, scaler])
pipeline_model = feature_pipeline.fit(basetable_with_label)
final_data = pipeline_model.transform(basetable_with_label)

# Split data
train, temp = final_data.select("features", "label", "review").randomSplit([0.7, 0.3], seed=42)
val, test = temp.randomSplit([0.5, 0.5], seed=42)

print(f"\nData Split:")
print(f"Training set:   {train.count():,}")
print(f"Validation set: {val.count():,}")
print(f"Test set:       {test.count():,}")

# Add class weights
neg_weight = 2.5
pos_weight = 1.0

class_weight = train.groupBy("label").agg(count("*").alias("row_count")).withColumn(
    "weight",
    when(col("label") == 0.0, lit(neg_weight))
    .otherwise(lit(pos_weight))
)

train = train.join(class_weight.select("label", "weight"), on="label", how="left").withColumn("classWeight", col("weight"))

# Random Forest setup
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    weightCol="classWeight",
    numTrees=100,
    maxDepth=12,
    minInstancesPerNode=10,
    seed=42
)

rf_paramGrid = ParamGridBuilder()\
    .addGrid(rf.numTrees, [100, 150])\
    .addGrid(rf.maxDepth, [10, 12, 15])\
    .addGrid(rf.minInstancesPerNode, [5, 10])\
    .build()

rf_cv = CrossValidator(
    estimator=rf,
    estimatorParamMaps=rf_paramGrid,
    evaluator=BinaryClassificationEvaluator(metricName="areaUnderPR"),
    numFolds=3,
    seed=42
)

# Train Random Forest
print("\nTraining Random Forest...")
rf_model = rf_cv.fit(train)
rf_predictions = rf_model.transform(val)

# Basic evaluation
rf_metrics = evaluate_model(rf_predictions, "Random Forest (Validation)")

# Get feature names and importance
formula_output = pipeline_model.stages[0].transform(basetable_with_label)
feature_attrs = formula_output.schema["raw_features"].metadata["ml_attr"]["attrs"]
feature_names = []
for attr_type in ["numeric", "binary", "nominal"]:
    if attr_type in feature_attrs:
        feature_names.extend([attr["name"] for attr in feature_attrs[attr_type]])

selected_indices = pipeline_model.stages[1].selectedFeatures
selected_feature_names = [feature_names[i] for i in selected_indices]

# Print Random Forest Feature Importance
print("\nRandom Forest - Top 10 Most Important Features:")
if hasattr(rf_model.bestModel, "featureImportances"):
    importances = rf_model.bestModel.featureImportances.toArray()
    rf_feature_importance = list(zip(selected_feature_names, importances))
    rf_sorted_importance = sorted(rf_feature_importance, key=lambda x: x[1], reverse=True)
    for feature_name, importance in rf_sorted_importance[:10]:
        print(f"{feature_name}: {importance:.4f}")

# Evaluate on test set
print("\nTest Set Results:")
rf_test = rf_model.transform(test)
rf_test_metrics = evaluate_model(rf_test, "Random Forest (Test)")

# Print best parameters
print("\nBest Random Forest Parameters:")
best_rf = rf_model.bestModel
print(f"Number of Trees: {best_rf._java_obj.getNumTrees()}")
print(f"Max Depth: {best_rf._java_obj.getMaxDepth()}")
print(f"Min Instances Per Node: {best_rf._java_obj.getMinInstancesPerNode()}")

# Display detailed confusion matrix rates
val_rates = display_confusion_rates(rf_predictions, "Random Forest (Validation)")
test_rates = display_confusion_rates(rf_test, "Random Forest (Test)")



# COMMAND ----------

# MAGIC %md
# MAGIC Make predictions on the test basetable and save them as a CSV file

# COMMAND ----------

# Load the test set
print("Loading test data...")
test_data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/tmp/test_basetable")

# Create temporary label column (needed for pipeline)
test_data_with_label = test_data.withColumn("label", lit(0.0))

# Apply feature engineering pipeline to test data
print("Processing test data...")
processed_test = pipeline_model.transform(test_data_with_label)

# Generate predictions
print("Making predictions...")
predictions = rf_model.transform(processed_test)

# Select and format output
output = predictions.select(
    "order_id",
    col("prediction").cast("double").alias("pred_review_score")
)

# Save predictions to a single CSV file
print("Saving predictions...")
# Remove dbfs: from the path
output.coalesce(1).write.mode("overwrite").option("header", "true").csv("/FileStore/predictions_output")

# Get the download path
files = dbutils.fs.ls("/FileStore/predictions_output")
csv_file = [x.path for x in files if x.path.endswith(".csv")][0]

# Print the download URL
print("\nDownload your predictions from:")
print(f"/files{csv_file}")

# Display the predictions
print("\nLoading saved predictions to display:")
saved_predictions = spark.read.csv("/FileStore/predictions_output", header=True)
display(saved_predictions)
