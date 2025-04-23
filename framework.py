from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import expr

# Spark session
spark = SparkSession.builder.appName("DynamicDFFramework").getOrCreate()

# -------------------------------
# Transformation Functions
# -------------------------------
def select_columns(df: DataFrame, columns: list) -> DataFrame:
    return df.select(*columns)

def with_column(df: DataFrame, params: dict) -> DataFrame:
    for col_name, expr_str in params.items():
        df = df.withColumn(col_name, expr(expr_str))
    return df

def drop_columns(df: DataFrame, columns: list) -> DataFrame:
    return df.drop(*columns)

def filter_rows(df: DataFrame, condition: str) -> DataFrame:
    return df.filter(condition)

def join_dfs(df: DataFrame, params: dict) -> DataFrame:
    right_df = params["right_df"]
    on = params["on"]
    how = params.get("how", "inner")
    return df.join(right_df, on=on, how=how)

# -------------------------------
# Action Functions
# -------------------------------
def show_df(df: DataFrame, num: int = 20):
    df.show(num)

def collect_df(df: DataFrame):
    return df.collect()

# -------------------------------
# Mapping Dictionaries
# -------------------------------
TRANSFORMATIONS = {
    "select": select_columns,
    "withColumn": with_column,
    "drop": drop_columns,
    "filter": filter_rows,
    "join": join_dfs,
}

ACTIONS = {
    "show": show_df,
    "collect": collect_df,
}

# -------------------------------
# Apply Transformations
# -------------------------------
def apply_transformations(df: DataFrame, operations: list) -> DataFrame:
    for op in operations:
        operation = op["operation"]
        params = op["params"]
        df = TRANSFORMATIONS[operation](df, params)
    return df

def apply_actions(df: DataFrame, actions: list):
    for action in actions:
        name = action["action"]
        params = action.get("params", {})
        ACTIONS[name](df, **params)

# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    data = [("Alice", 34), ("Bob", 45)]
    df = spark.createDataFrame(data, ["name", "age"])

    config = {
        "transformations": [
            {"operation": "withColumn", "params": {"age_plus_10": "age + 10"}},
            {"operation": "select", "params": ["name", "age_plus_10"]},
        ],
        "actions": [
            {"action": "show", "params": {"num": 5}}
        ]
    }

    df_transformed = apply_transformations(df, config["transformations"])
    apply_actions(df_transformed, config["actions"])

