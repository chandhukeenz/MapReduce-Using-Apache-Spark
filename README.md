# ArbitrageGainer

**ArbitrageGainer** is a PySpark-based application designed to identify historical arbitrage opportunities in cryptocurrency trading by analyzing price discrepancies across multiple exchanges.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Running the Application](#running-the-application)
- [MapReduce Algorithm](#mapreduce-algorithm)
- [Sample Data](#sample-data)
- [License](#license)

## Features

- **Historical Arbitrage Analysis:** Processes historical cryptocurrency data to identify and count arbitrage opportunities.
- **Parallel Processing:** Utilizes Apache Spark's distributed computing capabilities for efficient data analysis.
- **Customizable Input:** Allows users to specify the path to their historical data file.

## Prerequisites

Before setting up **ArbitrageGainer**, ensure you have the following installed:

- **Apache Spark:** Version 3.0 or higher
- **Python:** Version 3.7 or higher
- **Java Development Kit (JDK):** Version 8 or higher
- **PySpark:** Python API for Spark

## Setup Instructions

### 1. Install Apache Spark

**Apache Spark** is essential for running the parallel data processing tasks in this application.

#### Download Spark:

1. Visit the [Apache Spark Downloads](https://spark.apache.org/downloads.html) page.
2. Choose the latest stable release (e.g., Spark 3.4.0) with **Pre-built for Apache Hadoop**.
3. Download and extract the package:

   ```bash
   wget https://downloads.apache.org/spark/spark-3.4.0/spark-3.4.0-bin-hadoop3.tgz
   tar -xvzf spark-3.4.0-bin-hadoop3.tgz
   mv spark-3.4.0-bin-hadoop3 /opt/spark
   ```

#### Set Environment Variables:

Add Spark to your `PATH` by editing your `~/.bashrc` or `~/.zshrc`:

```bash
export SPARK_HOME=/opt/spark
export PATH=$PATH:$SPARK_HOME/bin
```

Apply the changes:

```bash
source ~/.bashrc
```

#### Verify Spark Installation:

```bash
spark-shell
```

You should see the Spark shell prompt. Exit the shell:

```bash
:quit
```

### 2. Install Python Dependencies

Ensure you have **Python 3.7+** installed. Install PySpark using `pip`:

```bash
pip install pyspark==3.4.0
```

### 3. Prepare Historical Data

Ensure that your historical data file (`HistoricalData.txt`) is in JSON format and placed in the project's root directory or specify its path when running the script.

**Sample Record:**

```json
[
    {"ev":"XQ","pair":"CHZ-USD","lp":0,"ls":0,"bp":0.0771,"bs":41650.4,"ap":0.0773,"as":142883.4,"t":1690409119847,"x":1,"r":1690409119856},
    {"ev":"XQ","pair":"CHZ-USD","lp":0,"ls":0,"bp":0.0771,"bs":41650.4,"ap":0.0773,"as":135498.5,"t":1690409119848,"x":1,"r":1690409119856},
    {"ev":"XQ","pair":"KNC-USD","lp":0,"ls":0,"bp":0.72035,"bs":314,"ap":0.7216,"as":314,"t":1690409119855,"x":2,"r":1690409119855},
    ...
]
```

## Running the Application

Execute the `main.py` script using `spark-submit`, specifying the path to your historical data file if it's not named `historicalData.txt`.

```bash
spark-submit main.py --input /path/to/historicalData.txt
```

**Default Usage:**

If your data file is named `historicalData.txt` and located in the same directory as `main.py`, simply run:

```bash
spark-submit main.py
```

## MapReduce Algorithm

**ArbitrageGainer** follows a MapReduce-like algorithm to efficiently process and analyze historical cryptocurrency data:

1. **Produce (Key, Value) Pairs:**
   - **Key:** Combination of `bucket` (5ms interval) and `pair` (e.g., "CHZ-USD").
   - **Value:** Quote details `{x, bp, ap}` where `x` is the exchange ID, `bp` is the bid price, and `ap` is the ask price.

2. **Filter Out Incomplete Data:**
   - **Action:** Validates that each currency pair consists of exactly two 3-letter currency codes.
   - **Implementation:** Uses a User-Defined Function (UDF) `valid_pair_udf` to filter out invalid pairs.

3. **Mapper:**
   - **Action:** Transforms and groups data by `(bucket, pair)`, collecting all relevant quotes into a list.
   - **Implementation:** Uses `groupBy` and `collect_list` to aggregate quotes for each key.

4. **Shuffle & Sort:**
   - **Action:** Spark automatically handles the shuffle and sort phase during the `groupBy` operation, ensuring that all quotes for each `(bucket, pair)` key are processed together.

5. **Reducer:**
   - **Action:** Processes each group of quotes to identify arbitrage opportunities where price differences exceed $0.01.
   - **Implementation:** Uses another UDF `find_arbitrage_udf` to flag opportunities, then aggregates the total number of opportunities per currency pair.

### How Spark Leverages Parallelism

- **Distributed Data Processing:** Spark divides the data into partitions distributed across multiple nodes, allowing simultaneous processing.
  
- **Parallel Transformations:** Operations like `filter`, `groupBy`, and UDF applications are executed in parallel on different partitions.
  
- **Optimized Execution:** Spark's Catalyst optimizer efficiently plans the execution pipeline, minimizing data movement and maximizing resource utilization.
  
- **In-Memory Computing:** Intermediate results are kept in memory, reducing I/O overhead and speeding up iterative computations.

## Sample Data

Below is a snippet from `HistoricalData.txt` to illustrate the expected data format:

```json
[
    {"ev":"XQ","pair":"CHZ-USD","lp":0,"ls":0,"bp":0.0771,"bs":41650.4,"ap":0.0773,"as":142883.4,"t":1690409119847,"x":1,"r":1690409119856},
    {"ev":"XQ","pair":"CHZ-USD","lp":0,"ls":0,"bp":0.0771,"bs":41650.4,"ap":0.0773,"as":135498.5,"t":1690409119848,"x":1,"r":1690409119856},
    {"ev":"XQ","pair":"KNC-USD","lp":0,"ls":0,"bp":0.72035,"bs":314,"ap":0.7216,"as":314,"t":1690409119855,"x":2,"r":1690409119855},
    ...
]
```

Ensure your data follows this structure for accurate processing.
