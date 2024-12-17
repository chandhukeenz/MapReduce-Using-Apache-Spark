from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType, LongType, IntegerType, BooleanType
)
import sys

if __name__ == "__main__":
    input_path = "historicalData.txt"
    for i, arg in enumerate(sys.argv):
        if arg == "--input" and i + 1 < len(sys.argv):
            input_path = sys.argv[i+1]

    if input_path is None:
        print("Please provide --input path to the historical data JSON file.")
        sys.exit(1)
    
    #Initialize Spark Session
    spark = SparkSession.builder \
        .appName("HistoricalArbitrageOpportunityCalculation") \
        .getOrCreate()

    #Define Schema for Input Data
    schema = StructType([
        StructField("ev", StringType(), True),
        StructField("pair", StringType(), True),
        StructField("lp", FloatType(), True),
        StructField("ls", FloatType(), True),
        StructField("bp", FloatType(), True),
        StructField("bs", FloatType(), True),
        StructField("ap", FloatType(), True),
        StructField("as", FloatType(), True),
        StructField("t", LongType(), True),
        StructField("x", IntegerType(), True),
        StructField("r", LongType(), True)
    ])

    #read json data and apply schema on it
    df = spark.read.schema(schema).json(input_path)

    # validate currency pairs to take into consideration for historical arbitrage opportunity
    def valid_pair(pair_str):
        if pair_str is None:
            return False
        parts = pair_str.split("-")
        if len(parts) == 2 and len(parts[0]) == 3 and len(parts[1]) == 3:
            return True
        return False

    #register user defined function -> valid_pair
    spark.udf.register("valid_pair_udf", valid_pair, BooleanType())
    df = df.filter(F.expr("valid_pair_udf(pair)"))

    #mapper: group data into 5 ms interval
    df = df.withColumn("bucket", (F.col("t") / F.lit(5)).cast("long"))

    #regroup the pairs with identical keys -> bucket, pair -> list of quotes {x,bp,ap}
    grouped = df.groupBy("bucket", "pair").agg(
        F.collect_list(F.struct("x", "bp", "ap")).alias("quotes")
    )

    #reducer: identify arbitrage opportunities
    def find_arbitrage_opportunities(quotes):
        best_per_exchange = {}
        for q in quotes:
            ex_id = q['x']
            bp = q['bp']
            ap = q['ap']
            if ex_id not in best_per_exchange or bp > best_per_exchange[ex_id]['bp']:
                best_per_exchange[ex_id] = {'bp': bp, 'ap': ap}

        if len(best_per_exchange) < 2:
            return 0

        exchanges = list(best_per_exchange.keys())
        for i in range(len(exchanges)):
            for j in range(i+1, len(exchanges)):
                ex1 = best_per_exchange[exchanges[i]]
                ex2 = best_per_exchange[exchanges[j]]
                if (ex2['bp'] - ex1['ap'] > 0.01) or (ex1['bp'] - ex2['ap'] > 0.01):
                    return 1 
        return 0

    #register user defined function -> find_arbitrage_opportunities
    spark.udf.register("find_arbitrage_udf", find_arbitrage_opportunities, IntegerType())

    #determine arbitrage opportunities within each group
    arbitrage_df = grouped.withColumn("opportunity", F.expr("find_arbitrage_udf(quotes)"))

    #aggregate the total number of arbitrage opportunities per currency pair
    result = arbitrage_df.groupBy("pair").agg(F.sum("opportunity").alias("total_opportunities"))

    result.show(truncate=False)

    spark.stop()