import pandas as pd
import numpy as np
import dask.dataframe as dd
import time

# Generating Dummy Data
np.random.seed(42)
num_samples = 10000  # Simulating 10,000 predictions

# Simulated actual values
actuals = np.random.randint(0, 2, num_samples)  # Binary classification (0 or 1)

# Simulated predictions from two different models
model_1_preds = np.random.uniform(0, 1, num_samples)  # Probability scores for Model 1
model_2_preds = np.random.uniform(0, 1, num_samples)  # Probability scores for Model 2

# Creating DataFrame
data = pd.DataFrame({
    "Actuals": actuals,
    "Model_1_Predictions": model_1_preds,
    "Model_2_Predictions": model_2_preds
})

# Saving dummy data to CSV
data.to_csv("model_predictions.csv", index=False)

# Load data using Dask for performance optimization
dd_data = dd.read_csv("model_predictions.csv")

# Define function to calculate QA metrics
def compute_metrics(df):
    """Computes key QA metrics comparing Model 1 and Model 2."""
    results = {}

    # Absolute Error
    df["Model_1_Error"] = abs(df["Actuals"] - df["Model_1_Predictions"])
    df["Model_2_Error"] = abs(df["Actuals"] - df["Model_2_Predictions"])

    # Mean Absolute Error (MAE)
    results["Model_1_MAE"] = df["Model_1_Error"].mean().compute()
    results["Model_2_MAE"] = df["Model_2_Error"].mean().compute()

    # Root Mean Squared Error (RMSE)
    results["Model_1_RMSE"] = np.sqrt(((df["Actuals"] - df["Model_1_Predictions"]) ** 2).mean().compute())
    results["Model_2_RMSE"] = np.sqrt(((df["Actuals"] - df["Model_2_Predictions"]) ** 2).mean().compute())

    # Model Performance Comparison
    results["Best_Model"] = "Model_1" if results["Model_1_MAE"] < results["Model_2_MAE"] else "Model_2"

    return results

# Measure execution time
start_time = time.time()

# Compute QA metrics
qa_results = compute_metrics(dd_data)

# Save results to a QA Report CSV file
qa_report_df = pd.DataFrame([qa_results])
qa_report_df.to_csv("QA_report.csv", index=False)

# Execution time
end_time = time.time()
print(f"QA Report generated in {round(end_time - start_time, 2)} seconds")

# Print results
print("QA Analysis Report:")
print(qa_report_df)

