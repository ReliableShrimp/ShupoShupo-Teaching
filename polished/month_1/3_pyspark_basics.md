# PySpark — From SQL to ML Pipelines

We'll learn `PySpark` already in the first month, because the person writing these notes doesn't really like `pandas`... yeah. But that doesn't mean you shouldn't learn `pandas`! Try it yourself, manage some data with it — but once the GBs get big enough, the problem stops being "buy a better machine" and becomes "manage those bubbly GBs waiting for you," sadly. To manage them, we'll need something built exactly for this: Spark. (There are other tools out there too, but we'll lean on Spark for most of this.)

## Table of Contents

1. PySpark Basics: SQL-in-Python (the Easy Way)
    1. SparkSession setup
    2. createDataFrame
    3. createOrReplaceTempView
    4. Full example with spark.sql
2. PySpark Basics: The DataFrame API (the Programmatic Way)
    1. Quick intro
    2. Learning to code
        1. .select()
        2. .filter()
        3. Handling nulls
    3. Window functions
        1. partitionBy + rank()
        2. lag() and lead()
        3. max(), min(), avg() over a window
3. PySpark for ML Engineering
    1. Quick intro
    2. Learning to code
        1. Reading data at scale
        2. VectorAssembler & StringIndexer
        3. Linear & logistic regression
        4. Pipelines (Transformers vs. Estimators)
        5. Loading a saved PipelineModel
        6. Reading from SQL databases via JDBC
        7. Final project: hallucination-detection pipeline

---

## 1. PySpark Basics: SQL-in-Python (the Easy Way)

Since we already learned `SQL` in the last note, we'll now do the same stuff, but in Python — using `PySpark`. We have two options here: the first is the easy way, the second is a bit more of a nightmare, but you'll probably end up using both eventually (month... 9, maybe? — prediction from 29/06/2026).

The easy way is to write `SQL` inside Python, thanks to `pyspark.sql`.

For example, let's say we have a table:

| Name     | Age | Role               | City          | Country        | Experience_Years |
| :------- | :-- | :----------------- | :------------ | :------------- | :---------------- |
| Alice    | 28  | Data Engineer      | New York      | United States  | 5                 |
| Bob      | 35  | Data Scientist     | San Francisco | United States  | 8                 |
| Charlie  | 42  | Manager            | London        | United Kingdom | 15                |
| Diana    | 31  | Software Engineer  | Berlin        | Germany        | 6                 |
| Evan     | 24  | Analyst            | Toronto       | Canada         | 2                 |
| Fiona    | 29  | DevOps Engineer    | Dublin        | Ireland         | 4                 |
| Giovanni | 33  | SRE                | Milan         | Italy           | 7                 |
| Haruto   | 26  | ML Engineer        | Tokyo         | Japan           | 3                 |
| Ilinca   | 38  | Solution Architect | Timisoara     | Romania         | 11                |
| Jin      | 45  | Director of IT     | Seoul         | South Korea     | 18                |

Now let's recreate it in Python:

```python
data_list = [
    ("Alice", 28, "Data Engineer", "New York", "United States", 5),
    ("Bob", 35, "Data Scientist", "San Francisco", "United States", 8),
    ("Charlie", 42, "Manager", "London", "United Kingdom", 15),
    ("Diana", 31, "Software Engineer", "Berlin", "Germany", 6),
    ("Evan", 24, "Analyst", "Toronto", "Canada", 2),
    ("Fiona", 29, "DevOps Engineer", "Dublin", "Ireland", 4),
    ("Giovanni", 33, "SRE", "Milan", "Italy", 7),
    ("Haruto", 26, "ML Engineer", "Tokyo", "Japan", 3),
    ("Ilinca", 38, "Solution Architect", "Timisoara", "Romania", 11),
    ("Jin", 45, "Director of IT", "Seoul", "South Korea", 18)
]

columns = ["Name", "Age", "Role", "City", "Country", "Experience_Years"]
```

### 1. SparkSession setup

Let's start with the easy way — first, some magic setup code:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("TryingPySpark").master("local[*]").getOrCreate()
```

Word-by-word breakdown:
- `SparkSession` — the newest, all-in-one entry point. You *could* instead write `SQLContext`, `SparkContext`, or `HiveContext` — older, more narrowly-scoped versions, each built for just one task rather than all of them together like `SparkSession`. `SQLContext` lets you write SQL the PySpark way, but nothing more; `HiveContext` lets you write SQL the way you already know it, but nothing more. So instead of picking one, just use `SparkSession` — it covers all of them.
- `.builder` — think of it as an architect: it gathers information like `appName` and `master`, and builds something useful out of them.
- `.appName("<name>")` — sets the name of the application. If you're running this on a cluster, this is the name shown on the Spark Web UI dashboard.
- `.master("local[*]")` — highly recommended when writing code on your local machine, since it uses all available cores instead of just one (which would be painfully slow).
- `.getOrCreate()` — a magic command that checks if you already have a session; if you do, it reuses it, otherwise it creates a new one.

### 2. createDataFrame

Now let's look at our table again, and write:

```python
df = spark.createDataFrame(data=data_list, schema=columns)
```

Word-by-word breakdown:
- `.createDataFrame()` — takes a raw Python object (like a list of tuples) and chops it up, spreading the rows across all the available CPU cores on your machine (or cluster) so they can be processed in parallel.
- `data=<your_list>` — the raw Python object you want it to chop up.
- `schema=<the_columns_name>` — the variable holding your column names. For example, since our column-name variable is called `columns`, we write `schema=columns`.

### 3. createOrReplaceTempView

Now let's write the next part:

```python
df.createOrReplaceTempView("people")
```

Word-by-word breakdown:
- `.createOrReplace` — creates a table view for you, or, if one already exists under the same name, overwrites it, so we don't get a random error.
- `TempView` — "Temporary View." Temporary, because it doesn't get saved to disk or anything like that — it only exists while your Python script is running. Once the script stops, it vanishes, since it lives purely in your computer's temporary RAM, not on your hard drive.
- `("people")` — just the name you give your SQL table, so it's recognized when you write `FROM <the_name_you_gave_it>`.

### 4. Full example with spark.sql

And in the end, you make a variable and write `spark.sql("""..."""")`.

Full code:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("AgainJustATry").master("local[*]").getOrCreate()

data_list = [
    ("Alice", 28, "Data Engineer", "New York", "United States", 5),
    ("Bob", 35, "Data Scientist", "San Francisco", "United States", 8),
    ("Charlie", 42, "Manager", "London", "United Kingdom", 15),
    ("Diana", 31, "Software Engineer", "Berlin", "Germany", 6),
    ("Evan", 24, "Analyst", "Toronto", "Canada", 2),
    ("Fiona", 29, "DevOps Engineer", "Dublin", "Ireland", 4),
    ("Giovanni", 33, "SRE", "Milan", "Italy", 7),
    ("Haruto", 26, "ML Engineer", "Tokyo", "Japan", 3),
    ("Ilinca", 38, "Data Engineer", "Timisoara", "Romania", 11),  # <- changed to "Data Engineer" here on purpose, so the filter below has 2 matches instead of 1
    ("Jin", 45, "Director of IT", "Seoul", "South Korea", 18)
]

columns = ["Name", "Age", "Role", "City", "Country", "Experience_Years"]

df = spark.createDataFrame(data=data_list, schema=columns)

df.createOrReplaceTempView("people")

only_data_engineers_df = spark.sql("""
SELECT Name, Age, Role
FROM people
WHERE Role = 'Data Engineer'
""")

order_by_age_df = spark.sql("""
SELECT Name, Age, City, Country
FROM people
ORDER BY Age DESC
""")

only_data_engineers_df.show()
order_by_age_df.show()
# We use "_df" as a suffix to make it obvious that a variable is a DataFrame.

"""
output of only_data_engineers_df:

+------+---+-------------+
|  Name|Age|         Role|
+------+---+-------------+
| Alice| 28|Data Engineer|
|Ilinca| 38|Data Engineer|
+------+---+-------------+

output of order_by_age_df:

+--------+---+-------------+--------------+
|    Name|Age|         City|       Country|
+--------+---+-------------+--------------+
|     Jin| 45|        Seoul|   South Korea|
| Charlie| 42|       London|United Kingdom|
|  Ilinca| 38|    Timisoara|       Romania|
|     Bob| 35|San Francisco| United States|
|Giovanni| 33|        Milan|         Italy|
|   Diana| 31|       Berlin|       Germany|
|   Fiona| 29|       Dublin|       Ireland|
|   Alice| 28|     New York| United States|
|  Haruto| 26|        Tokyo|         Japan|
|    Evan| 24|      Toronto|        Canada|
+--------+---+-------------+--------------+
"""
```

But sadly, after reading all of this, you now have to suffer, because we're about to learn the **DataFrame API** — the painful way.

---

## 2. PySpark Basics: The DataFrame API (a.k.a. the Programmatic Way)

### 1. Quick intro

Sadly, the painful part has arrived, and you're probably wondering: why do we need this other version at all, when we could just write SQL in Python and be happy forever? The SQL-in-Python approach falls short in a few real areas where the Programmatic way shines. For example:

1. **SQL can't handle dynamic code (loops and variables).** Say your company has millions of users from many different countries — let's say 84 countries — and your boss wants you to split the people out by country. In the SQL version, you'd have to do this manually, so get the popcorn ready for the next 84 `WHERE Country = '<Exact_Country>'` lines you're about to type by hand. With the Programmatic way, you'd just loop:
```python
countries = ["United States", "Germany", "Canada", "Japan"]

for c in countries:
    df.filter(F.col("Country") == c).write.csv(f"{c}_data.csv")
```
   (Not all 84 countries listed here, but you get the idea.) And boom, you're done in under 5 minutes. Expect a marriage proposal from your boss, a raise, and much more!

2. **Compile-time errors vs. runtime disasters.** Imagine you make a small typo in your cute SQL-in-Python string, something like `'SELECC * FROM users'`. Python sees your SQL query as just a plain, dumb string of text. The script starts running, wastes 10 minutes spinning up your cluster, and then — *boom* — crashes in the middle of production because of a typo. Our glorious DataFrame API, on the other hand, would never let that slide: it immediately calls in Guido van Rossum himself and slaps you with a missing-variable or syntax error before the code ever touches the server. It crushes your hopes in the first second, instead of letting you hope for 10 minutes first.

3. **Building "data pipelines" (modularity).** Sadly, you won't just write 10 lines of code to get something done — sometimes you'll need 11. (Kidding — you'll often need hundreds, sometimes way more.) So instead of writing one giant doom-and-spaghetti script, you can pass DataFrames through standard Python functions like a pipeline:

```python
def clean_missing_ages(input_df):
    return input_df.filter(F.col("Age").isNotNull())

def calculate_experience(input_df):
    return input_df.withColumn("Seniority", F.col("Experience_Years") > 10)

final_df = df.transform(clean_missing_ages).transform(calculate_experience)
```

Don't worry about it for now — you'll cry over it later, so save your tears for the next topics.

**Quick summary:** We use SQL when we want to quickly explore data or write a quick report. We use the DataFrame API when we're building enterprise, automated production pipelines that need to be tested, scaled, and automated.

### 2. Learning to code

First, let's spoil ourselves with a table:

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("BraceToSuffer").master("local[*]").getOrCreate()

data_list = [
    ("Leo", 25, "Junior Data Engineer", "Data", 55000, 5),
    ("Maya", 34, "Senior Data Scientist", "Data", 115000, 3),
    ("Liam", 41, "Engineering Manager", "Tech", 140000, 2),
    ("Noah", 22, "Intern", "Tech", 35000, 0),
    ("Ava", 29, "Analyst", "Marketing", 62000, 4),
    ("Lucas", 31, "DevOps Engineer", "Tech", 95000, 5),
    ("Sofia", 28, "Data Engineer", "Data", 82000, None),
    ("Emma", 45, "VP of Technology", "Tech", 190000, 1),
    ("Oliver", None, "Marketing Specialist", "Marketing", 58000, 3)
]

columns = ["Name", "Age", "Role", "Department", "Salary", "Remote_Days"]

df = spark.createDataFrame(data=data_list, schema=columns)
```

Now let's learn the basic commands everyone should know.

**`.select()`** — you probably already understand this without me explaining it. It's the programmatic version of `SELECT`. For example:

```python
# Let's say we want just the name, age, and role.
# (Yes, that's an Oxford comma up there, so I can officially announce I'm a C2 in
# English. No TOEFL needed.)

name_only_df = df.select(F.col("Name"), F.col("Age"), F.col("Role"))
# We imported functions from pyspark.sql as F, so we don't have to write
# functions.sum, functions.max, and so on every time. F.col is also a bit of a
# safety net — instead of writing df.select(df["Age"]), we write F.col("Age"),
# which is a little safer.

name_only_df.show()
# Shows the table.

"""
Output of the variable:

+------+----+--------------------+
|  Name| Age|                Role|
+------+----+--------------------+
|   Leo|  25|Junior Data Engineer|
|  Maya|  34|Senior Data Scien...|
|  Liam|  41| Engineering Manager|
|  Noah|  22|              Intern|
|   Ava|  29|             Analyst|
| Lucas|  31|     DevOps Engineer|
| Sofia|  28|       Data Engineer|
|  Emma|  45|    VP of Technology|
|Oliver|NULL|Marketing Specialist|
+------+----+--------------------+
"""
```

That's how `.select()` works — no `FROM` needed, since `df` is already tied to the data we passed in via `data=...`. That's the output we get, but there's a lot more we can do.

**`.filter()`** — this is basically the `WHERE` of the programmatic way. We can filter by exact age, exact role, and so on. `.filter()` supports logical conditions (`AND`, `OR`, `NOT`), just written a bit differently:
- `AND` = `&`
- `OR` = `|`
- `NOT` = `~`

Some examples:

```python
tech_dept_df = df.filter(F.col("Department") == "Tech")

tech_dept_df.show()

name_age40l_sal100k_df = df.filter((F.col("Age") < 40) & (F.col("Salary") > 100_000)) \
    .select(F.col("Name"), F.col("Age"), F.col("Salary")) \
    .show()
# The "\" tells Python the statement continues onto the next line. Without it,
# Python assumes the statement ends at the end of the current line — think of it
# as "wait! don't treat this as finished yet, I'm continuing below."
# Remember: whenever we use "\", the next line starts with "." to stay chained to df.

name_age25l_remote5eq_df = df.filter((F.col("Age") < 25) | (F.col("Remote_Days") == 5)) \
    .select(F.col("Name"), F.col("Age"), F.col("Remote_Days")) \
    .show()

"""
Output of tech_dept_df:

+-----+---+-------------------+----------+------+-----------+
| Name|Age|               Role|Department|Salary|Remote_Days|
+-----+---+-------------------+----------+------+-----------+
| Liam| 41|Engineering Manager|      Tech|140000|          2|
| Noah| 22|             Intern|      Tech| 35000|          0|
|Lucas| 31|    DevOps Engineer|      Tech| 95000|          5|
| Emma| 45|   VP of Technology|      Tech|190000|          1|
+-----+---+-------------------+----------+------+-----------+

Output of name_age40l_sal100k_df:

+----+---+------+
|Name|Age|Salary|
+----+---+------+
|Maya| 34|115000|
+----+---+------+

Output of name_age25l_remote5eq_df:

+-----+---+-----------+
| Name|Age|Remote_Days|
+-----+---+-----------+
|  Leo| 25|          5|
| Noah| 22|          0|
|Lucas| 31|          5|
+-----+---+-----------+
"""
```

That's how `.filter()` works, and it's genuinely useful, since we're going to go search for people with a good salary and marry them. (My bad — the charisma and beauty criteria have to be high too, for something like that.)

Another important concept is handling nulls, since we're going to run into them — and it's time to bury Python's last null:

```python
write_just_the_null_df = df.filter(F.col("Age").isNull())
# .isNull() means we're searching for everything that's null (a.k.a. nothing).

ignore_just_the_null_df = df.filter(F.col("Remote_Days").isNotNull())
ignore_just_the_null_df.show()
# .isNotNull() means we're searching for everything that's NOT null.

replace_null_df = df.fillna(value=0, subset=["Remote_Days"])
replace_null_df.show()
# .fillna() replaces all NULL values in the Remote_Days column with a default of 0.
# value=<number> is what you want to replace the null with.
# subset=[<column>] is the column (or columns) you want that replacement applied to.

df = df.dropna(subset=["Age", "Remote_Days"])
# This is really useful, because imagine we have a row with a NULL in it — we
# simply drop the whole row, since silently placing a 0 instead of NULL could
# make the model learn the wrong thing.

"""
Output of write_just_the_null_df:

+------+----+--------------------+----------+------+-----------+
|  Name| Age|                Role|Department|Salary|Remote_Days|
+------+----+--------------------+----------+------+-----------+
|Oliver|NULL|Marketing Specialist| Marketing| 58000|          3|
+------+----+--------------------+----------+------+-----------+

Output of ignore_just_the_null_df:

+------+----+--------------------+----------+------+-----------+
|  Name| Age|                Role|Department|Salary|Remote_Days|
+------+----+--------------------+----------+------+-----------+
|   Leo|  25|Junior Data Engineer|      Data| 55000|          5|
|  Maya|  34|Senior Data Scien...|      Data|115000|          3|
|  Liam|  41| Engineering Manager|      Tech|140000|          2|
|  Noah|  22|              Intern|      Tech| 35000|          0|
|   Ava|  29|             Analyst| Marketing| 62000|          4|
| Lucas|  31|     DevOps Engineer|      Tech| 95000|          5|
|  Emma|  45|    VP of Technology|      Tech|190000|          1|
|Oliver|NULL|Marketing Specialist| Marketing| 58000|          3|
+------+----+--------------------+----------+------+-----------+

Output of replace_null_df:

+------+----+--------------------+----------+------+-----------+
|  Name| Age|                Role|Department|Salary|Remote_Days|
+------+----+--------------------+----------+------+-----------+
|   Leo|  25|Junior Data Engineer|      Data| 55000|          5|
|  Maya|  34|Senior Data Scien...|      Data|115000|          3|
|  Liam|  41| Engineering Manager|      Tech|140000|          2|
|  Noah|  22|              Intern|      Tech| 35000|          0|
|   Ava|  29|             Analyst| Marketing| 62000|          4|
| Lucas|  31|     DevOps Engineer|      Tech| 95000|          5|
| Sofia|  28|       Data Engineer|      Data| 82000|          0|
|  Emma|  45|    VP of Technology|      Tech|190000|          1|
|Oliver|NULL|Marketing Specialist| Marketing| 58000|          3|
+------+----+--------------------+----------+------+-----------+
"""
```

Next topic: annoying, so get the napkins ready (I'll get mine ready too). Anyway, we're *not* going to learn `groupBy()` here, because `groupBy()` is a **destroyer** — it squashes rows down. If you have 100 employees and group by department, you get 3 rows back. The individual employees vanish.

So instead we'll learn windows! (Which don't destroy any of the rows.)

### 3. Window functions

What is a window? A window is an opening in the wall, or ro— ah, got you, jokes aside.

Let's walk through a small example. Imagine we have a table:

| Student | Class | Score | Submission_Date | Attended_Reviews |
| :-----: | :---: | :---: | :--------------: | :----------------: |
|  Alice  |   A   |  91   |    2026-06-15    |          4          |
|   Bob   |   A   |  75   |    2026-06-16    |          2          |
| Charlie |   A   |  88   |    2026-06-17    |          5          |
|  Frank  |   A   |  94   |    2026-06-18    |          3          |
|  Grace  |   A   |  82   |    2026-06-19    |          1          |
|  Liam   |   A   |  77   |    2026-06-20    |          2          |
|   Mia   |   A   |  96   |    2026-06-21    |          5          |
|  Noah   |   A   |  84   |    2026-06-22    |          4          |
|  David  |   B   |  95   |    2026-06-15    |          5          |
|   Eve   |   B   |  70   |    2026-06-16    |          0          |
|  Henry  |   B   |  85   |    2026-06-17    |          3          |
|   Ivy   |   B   |  91   |    2026-06-18    |          4          |
|  Jack   |   B   |  64   |    2026-06-19    |          2          |
|  Karen  |   B   |  79   |    2026-06-20    |          2          |
|   Leo   |   B   |  88   |    2026-06-21    |          5          |
|  Mona   |   B   |  73   |    2026-06-22    |          1          |
|  Ruby   |   C   |  67   |    2026-06-16    |          1          |
|   Sam   |   C   |  86   |    2026-06-17    |          3          |
|  Tina   |   C   |  98   |    2026-06-18    |          5          |
|   Uma   |   C   |  80   |    2026-06-19    |          2          |
| Victor  |   C   |  74   |    2026-06-20    |          1          |
|  Wendy  |   C   |  89   |    2026-06-21    |          4          |
| Xavier  |   C   |  83   |    2026-06-22    |          2          |
|  Yara   |   C   |  94   |    2026-06-23    |          5          |

Now, say Alice wants to know how good she was compared to the rest of Class A. To help Alice out, we'll use a window!

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("ManyTopicsIncoming").master("local[*]").getOrCreate()

data_list = [
    ("Alice", "A", 91, "2026-06-15", 4),
    ("Bob", "A", 75, "2026-06-16", 2),
    ("Charlie", "A", 88, "2026-06-17", 5),
    ("Frank", "A", 94, "2026-06-18", 3),
    ("Grace", "A", 82, "2026-06-19", 1),
    ("Liam", "A", 77, "2026-06-20", 2),
    ("Mia", "A", 96, "2026-06-21", 5),
    ("Noah", "A", 84, "2026-06-22", 4),
    ("David", "B", 95, "2026-06-15", 5),
    ("Eve", "B", 70, "2026-06-16", 0),
    ("Henry", "B", 85, "2026-06-17", 3),
    ("Ivy", "B", 91, "2026-06-18", 4),
    ("Jack", "B", 64, "2026-06-19", 2),
    ("Karen", "B", 79, "2026-06-20", 2),
    ("Leo", "B", 88, "2026-06-21", 5),
    ("Mona", "B", 73, "2026-06-22", 1),
    ("Ruby", "C", 67, "2026-06-16", 1),
    ("Sam", "C", 86, "2026-06-17", 3),
    ("Tina", "C", 98, "2026-06-18", 5),
    ("Uma", "C", 80, "2026-06-19", 2),
    ("Victor", "C", 74, "2026-06-20", 1),
    ("Wendy", "C", 89, "2026-06-21", 4),
    ("Xavier", "C", 83, "2026-06-22", 2),
    ("Yara", "C", 94, "2026-06-23", 5),
]

columns = [
    "Student",
    "Class",
    "Score",
    "Submission_Date",
    "Attended_Reviews"
]

df = spark.createDataFrame(data=data_list, schema=columns)

window = Window.partitionBy(F.col("Class")).orderBy(F.col("Score").desc())
# Breaking it down:
# .partitionBy — you already know the idea from SQL, but let's repeat it: since we
#   base ourselves on the "Class" column, PySpark looks at everyone in that class and
#   ranks them one by one. If we instead wrote .partitionBy(F.col("Student")), everyone
#   would get rank = 1, since every student's name is unique — each one becomes its
#   own group of one.
# .orderBy(F.col("Score").desc()) — orders from the biggest score down to the lowest.
#   Combined with .partitionBy(F.col("Class")), it ranks biggest-to-smallest *within*
#   each class independently — so if Class A has 5 people with high scores and Class B
#   has 5 people with high scores too, each class gets ranked on its own.

ranking_students_df = df.select(F.col("Student"), F.col("Class"), F.col("Score"), F.rank().over(window).alias("Class scores"))
# .rank() — ranks based on the last attribute we told it to look at; since we set the
#   window to order by Score, it ranks by Score.
# .alias(<name>) — renames the resulting column (works the same way for .lag, .lead, etc.)

ranking_students_df.show(len(data_list))


window = Window.orderBy(F.col("Score").desc(), F.col("Student"))
# We changed the window here, since we no longer want to look at each class independently.

previous_and_next_score_df = df.select(F.col("Student"), F.col("Class"), F.col("Score"), F.lag(F.col("Score")).over(window).alias("Previous score"), F.lead("Score").over(window).alias("Next score"))
# .lag(<column>) — shows the previous row's value for that column. (Note: the very
#   first row will always be NULL, since it has no previous row.)
# .lead(<column>) — shows the next row's value for that column. (The last row will
#   always be NULL, since there's no next row.)

previous_and_next_score_df.show(len(data_list))


window = Window.partitionBy(F.col("Class"))

max_min_avg_score_for_each_class = df.select(F.col("Student"), F.col("Class"), F.col("Score"), F.col("Submission_Date"), F.max(F.col("Score")).over(window).alias("Best score"), F.min(F.col("Score")).over(window).alias("Worst score"), F.avg(F.col("Score")).over(window).cast("decimal(10,1)").alias("Average score"))
# F.max looks at the class, finds the best score, and prints it for every row in that
#   class. For Class A, that's 94 (Frank's score).
# F.min does the opposite — for Class B, that's 64 (Jack's score).
# F.avg sums every score in the class and divides by the number of students. For Class
#   C we have 8 students: 67+86+98+80+74+89+83+94 = 671, and 671/8 = 83.9 (rounded a bit).
# We add .cast("decimal(10,1)") so Spark doesn't print out 15 ugly decimal digits.

max_min_avg_score_for_each_class.show(len(data_list))

"""
Output of ranking_students_df:

+-------+-----+-----+------------+
|Student|Class|Score|Class scores|
+-------+-----+-----+------------+
|    Mia|    A|   96|           1|
|  Frank|    A|   94|           2|
|  Alice|    A|   91|           3|
|Charlie|    A|   88|           4|
|   Noah|    A|   84|           5|
|  Grace|    A|   82|           6|
|   Liam|    A|   77|           7|
|    Bob|    A|   75|           8|
|  David|    B|   95|           1|
|    Ivy|    B|   91|           2|
|    Leo|    B|   88|           3|
|  Henry|    B|   85|           4|
|  Karen|    B|   79|           5|
|   Mona|    B|   73|           6|
|    Eve|    B|   70|           7|
|   Jack|    B|   64|           8|
|   Tina|    C|   98|           1|
|   Yara|    C|   94|           2|
|  Wendy|    C|   89|           3|
|    Sam|    C|   86|           4|
| Xavier|    C|   83|           5|
|    Uma|    C|   80|           6|
| Victor|    C|   74|           7|
|   Ruby|    C|   67|           8|
+-------+-----+-----+------------+

Output of previous_and_next_score_df:

+-------+-----+-----+--------------+----------+
|Student|Class|Score|Previous score|Next score|
+-------+-----+-----+--------------+----------+
|   Tina|    C|   98|          NULL|        96|
|    Mia|    A|   96|            98|        95|
|  David|    B|   95|            96|        94|
|  Frank|    A|   94|            95|        94|
|   Yara|    C|   94|            94|        91|
|  Alice|    A|   91|            94|        91|
|    Ivy|    B|   91|            91|        89|
|  Wendy|    C|   89|            91|        88|
|Charlie|    A|   88|            89|        88|
|    Leo|    B|   88|            88|        86|
|    Sam|    C|   86|            88|        85|
|  Henry|    B|   85|            86|        84|
|   Noah|    A|   84|            85|        83|
| Xavier|    C|   83|            84|        82|
|  Grace|    A|   82|            83|        80|
|    Uma|    C|   80|            82|        79|
|  Karen|    B|   79|            80|        77|
|   Liam|    A|   77|            79|        75|
|    Bob|    A|   75|            77|        74|
| Victor|    C|   74|            75|        73|
|   Mona|    B|   73|            74|        70|
|    Eve|    B|   70|            73|        67|
|   Ruby|    C|   67|            70|        64|
|   Jack|    B|   64|            67|      NULL|
+-------+-----+-----+--------------+----------+

Output of max_min_avg_score_for_each_class:

+-------+-----+-----+---------------+----------+-----------+-------------+
|Student|Class|Score|Submission_Date|Best score|Worst score|Average score|
+-------+-----+-----+---------------+----------+-----------+-------------+
|  Alice|    A|   91|     2026-06-15|        96|         75|         85.9|
|    Bob|    A|   75|     2026-06-16|        96|         75|         85.9|
|Charlie|    A|   88|     2026-06-17|        96|         75|         85.9|
|  Frank|    A|   94|     2026-06-18|        96|         75|         85.9|
|  Grace|    A|   82|     2026-06-19|        96|         75|         85.9|
|   Liam|    A|   77|     2026-06-20|        96|         75|         85.9|
|    Mia|    A|   96|     2026-06-21|        96|         75|         85.9|
|   Noah|    A|   84|     2026-06-22|        96|         75|         85.9|
|  David|    B|   95|     2026-06-15|        95|         64|         80.6|
|    Eve|    B|   70|     2026-06-16|        95|         64|         80.6|
|  Henry|    B|   85|     2026-06-17|        95|         64|         80.6|
|    Ivy|    B|   91|     2026-06-18|        95|         64|         80.6|
|   Jack|    B|   64|     2026-06-19|        95|         64|         80.6|
|  Karen|    B|   79|     2026-06-20|        95|         64|         80.6|
|    Leo|    B|   88|     2026-06-21|        95|         64|         80.6|
|   Mona|    B|   73|     2026-06-22|        95|         64|         80.6|
|   Ruby|    C|   67|     2026-06-16|        98|         67|         83.9|
|    Sam|    C|   86|     2026-06-17|        98|         67|         83.9|
|   Tina|    C|   98|     2026-06-18|        98|         67|         83.9|
|    Uma|    C|   80|     2026-06-19|        98|         67|         83.9|
| Victor|    C|   74|     2026-06-20|        98|         67|         83.9|
|  Wendy|    C|   89|     2026-06-21|        98|         67|         83.9|
| Xavier|    C|   83|     2026-06-22|        98|         67|         83.9|
|   Yara|    C|   94|     2026-06-23|        98|         67|         83.9|
+-------+-----+-----+---------------+----------+-----------+-------------+

That was the best, worst, and average score per class, since we used partitionBy on Class.
"""
```

And that's where our pain truly begins. I hope you've still got some tears left to spare, because things are only going to get worse. We started with a steep learning curve; now we're about to face a vertical cliff — a wall you'll have to climb with your bare hands. Your greatest strategy from this point on?

**[Fake it until you make it.](https://en.wikipedia.org/wiki/Fake_it_till_you_make_it)**

That's the Machine Learning cycle:

**Learn → Cry → Learn → Get motivated → Try to solve a problem → Fail → Cry → Repeat the cycle till something changes.**

Eventually you'll realize everyone goes through this cycle. The difference is that the ones who succeed are just luckier, or they actually use Google (or AI) a little better.

So... it's time for **PySpark for Machine Learning**. We've learned the fundamentals — now we're jumping straight into the deep end. Welcome to the DLC on **Nightmare difficulty**, because playing on Easy is no longer an option. Good luck — though you won't need it, because I'll get cooked explaining it, and you'll get cooked just by reading it.

---

## 3. PySpark for ML Engineering

### 1. Quick intro

Before we even start: in PySpark, `model.fit(X, y)` just isn't going to cut it anymore. PySpark wants us to suffer and learn new ideas, because Spark stores data **distributed across many machines** — having `X` and `y` as two separate objects would constantly require keeping them in sync across partitions, and that's just not worth it for Spark.

Imagine this: you're a manager, and one of your employees is called Bob. You'd usually tell Bob: "Come here, I'll give you $X$ and $Y$, and you'll match line 1 of $X$ to line 1 of $Y$." That's the `pandas` and `scikit-learn` idea. Say we have this $X$ and $Y$:

$$X = \begin{bmatrix} 91 & 4 \\\\ 75 & 2 \\\\ 88 & 5 \end{bmatrix}, \quad y = \begin{bmatrix} 1 \\\\ 0 \\\\ 1 \end{bmatrix}$$

Using `pandas` and `scikit-learn`, and then `model.fit(X, y)`, we get: Row 0 of $X$, `[91, 4]`, looks over at Row 0 of $y$, `[1]`, matches them up instantly, and trains the model.

But once the list gets past 100k+ rows (maybe millions), Bob insists on putting every single row on his own desk before he starts working. Once there are 100,000 rows (or millions), his desk — the computer's RAM — runs out of space.

Spark, on the other hand, was built intentionally to prevent exactly this. It uses more workers and hands out the info differently, for example:

```
+---------+----------------+-------+
|  Score  |Attended_Reviews| Label |  <-- Handed to Worker 1
+---------+----------------+-------+
|   91    |       4        |   1   |  (Alice is completely self-contained)
|   75    |       2        |   0   |  (Bob is completely self-contained)
+---------+----------------+-------+

+---------+----------------+-------+
|  Score  |Attended_Reviews| Label |  <-- Handed to Worker 2
+---------+----------------+-------+
|   88    |       5        |   1   |  (Charlie is completely self-contained)
+---------+----------------+-------+
```

Now let's reveal Spark's secret, and why everyone loves it (not platonic). Imagine — StoryMode:

```
Your boss said:
"Shrimp! Immediately put the 100M plates next to the forks! Each fork is already
labeled with a name, so don't mess up!"

You immediately thought of pandas, scikit-learn — you'd use them, solve the problem
in 2 minutes, and your boss would finally make you CEO and hand you all his money.
So that's what you did. Instead, your screen froze solid and threw an
Out Of Memory error. So now you're on your phone googling what to do after getting
an OOM error. After sorting it out, you start wondering what could actually help
you here, and then you see it: the honey, darling, love of your life — Spark,
waiting for you.

Now you start using Spark, and think, why like Spark over pandas? Because instead
of making the only worker (you) do everything, Spark traumatizes four other workers
too, and gives you just 20M plates, handing 20M to each of the other four. What
about the forks? Maybe it splits you 40M forks and one extra plate from one of your
coworkers. So instead of doing it all yourself, you start yelling for the other
workers, so they can pass you the exact fork that matches the name on the plates
you own, and you hand them what they need (this is actually called "shuffling").
After exchanging plates and forks back and forth a bunch of times, you finally have
your 20M plates and 20M forks — the computer didn't throw a tantrum, so we won. Now
you — employee #1 — have 20M forks and plates, employee #2 has the same amount...
employee #5 has the same amount too. The boss is happy and says:

"Because you had to exchange thousands of forks with the other workers (a shuffle),
everyone spent time walking around instead of placing plates. The work still
finished successfully, but the back-and-forth between workers made it slower than
if everything had already been in the right place. The whole process took 30
minutes instead of 30 seconds. (It's slower for smaller data — remember that.)
Could've gone faster — your pay's getting cut by 20%."
```

Life is sad, so that's how it works — but no worries, let's keep learning.

### 2. Learning to code

Sadly, the ad break's over, and now we have to code. But honestly, the first topic isn't even that hard — we just need to change a bit of what we're used to doing. In a real work environment, we won't get handed a plain Python list slammed into VS Code or Neovim — the data will live in a file holding all the info. That file might have 500K lines, or even 500M+; we don't know. But let me explain why using the old method here is wrong:

```python
# imagine company_info.txt is a file with 10M lines.
with open("company_info.txt", "r") as f:
    data_list = [line.split(",") for line in f]
    # We opened the file and tried to load 10M lines into a Python list at once...
    # BOOM! Your laptop turns blue, freezes, and crashes. You just used 100% of your RAM.

# This part is never reached.
df = spark.createDataFrame(data_list)
```

You crashed before Spark even got a chance to help you, because you tried to force a massive file through the tiny straw of your single laptop's memory. But Spark is a savior... so here's what it does instead:

**The production way: safe, fast, and scalable**

```python
spark = (
    SparkSession.builder
    .appName("MasterSpark")
    .master("spark://company-cluster:7077")
    .getOrCreate()
)

# Instead of using only our own computer, we connect to the company's Spark cluster.
# The Driver coordinates the work, while multiple Workers execute it in parallel.

df = spark.read.csv(
    "s3://my-academy-bucket/data/global_students.csv",
    header=True,
    inferSchema=True
)

# Verify it worked by printing the schema and a peek at the data
df.printSchema()
# df.printSchema() shows the structure of the data
df.show(5)
# df.show() prints the first 5 rows
```

What's all of this? Great question, I asked myself the same thing. Let's break it down piece by piece:

- **`.master("spark://company-cluster:7077")`** — connects to the company's Spark cluster instead of running everything on your local computer. This gives Spark access to many workers, CPUs, and a lot more RAM.
- **`spark.read.csv(<path>)`** — tells Spark to read a CSV file and build a PySpark DataFrame from it. If Spark is connected to a cluster, it automatically splits the file into pieces and lets multiple workers read those pieces in parallel.
- **`"s3://my-academy-bucket/data/global_students.csv"`** — the file's location, here stored in an Amazon S3 bucket instead of on your local machine (AWS knowledge we'll cover in... some month).
- **`header=True`** — the first row holds column names, so it doesn't get treated as an actual data row.
- **`inferSchema=True`** — Spark inspects the data and automatically determines each column's type (integer, double, string, date, etc.) instead of treating everything as text. Yeah, smart, I know.

The important thing to remember: **you don't manually split the CSV file.** You just say "read this file," and Spark figures out the best way to distribute the work across the cluster.

But life without problems is like a beach without sun, so let's go back to the good old times: Henry VI was born on December 6, 1421, at Windsor Castle. He w— wrong story, yet I remember drinking tea and talking to him.

Anyway, remember why Spark doesn't use `model.fit(x, y)`? Because it wants DataFrames, like:

```
+---------+----------------+-------+
| Score   |AttendedReviews | Label |
+---------+----------------+-------+
|   91    |       4        |   1   |
|   75    |       2        |   0   |
|   88    |       5        |   1   |
+---------+----------------+-------+
```

But here's the one problem: can an ML algorithm actually train directly on this DataFrame?

...

...

Absolutely not. :)

Because it wants vectors (yup, the linear algebra we love so much).

```
+----------------+-------+
| Features       | Label |
+----------------+-------+
| [91,4]         |   1   |
| [75,2]         |   0   |
| [88,5]         |   1   |
+----------------+-------+
```

What happened? Score and AttendedReviews got united into a single vector — but it's still easy to tell which is which: index 0 is Score, index 1 is AttendedReviews. We just combined them so the ML algorithm can actually learn from the data.

In code, it looks like this:

```python
# +---------+----------------+-------+
# | Score   |AttendedReviews | Label |
# +---------+----------------+-------+
# |   91    |       4        |   1   |
# |   75    |       2        |   0   |    <---- the table we're using
# |   88    |       5        |   1   |
# +---------+----------------+-------+

from pyspark.ml.feature import VectorAssembler
# We use pyspark.ml.feature since we want to add ML concepts, and VectorAssembler
# literally does "take several columns and pack them into one feature vector."

df = spark.read.csv(
    "s3://my-academy-bucket/data/global_students.csv",
    header=True,
    inferSchema=True
)
# Imagine we've already set up spark, and that the DataFrame from global_students.csv
# has plenty of columns, including "Score" and "AttendedReviews" (and, for this
# example, a "Label" column too).

assembler = VectorAssembler(inputCols=["Score", "AttendedReviews"], outputCol="Features")
# inputCols=[<columns>] — the columns we want to combine
# outputCol=<name> — the name of the new combined column

students_df = assembler.transform(df)
# Our new DataFrame — now an ML algorithm can actually work with it!
students_df.show(3)

"""
Output of students_df:

+----------------+-------+
| Features       | Label |
+----------------+-------+
| [91,4]         |   1   |
| [75,2]         |   0   |      <---- the two columns became one vector
| [88,5]         |   1   |
+----------------+-------+
"""
```

But before moving on: let's say $y$ won't always be as tidy as `np.array([[0], [1], [0]])` — sometimes it'll be `np.array(["Spam", "Safe", "Spam"])`. Scary, but it happens. That's why you'll use `StringIndexer`:

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer

spark = SparkSession.builder.appName("Index-ingStuff").master("local[*]").getOrCreate()

raw_data = [
    ("Congratulations! You won a lottery!", "spam"),
    ("Hey, are we still meeting for lunch today?", "safe"),
    ("Urgent: Claim your free cruise ticket now!", "spam")
]
# Yeah, you won't get a list this small at a real job, but it's fine for a trial run.

columns = ["EmailText", "Status"]

df = spark.createDataFrame(data=raw_data, schema=columns)
# .createDataFrame() again, since we only have a few lines here, no worries.

indexer = StringIndexer(inputCol="Status", outputCol="label")
# StringIndexer converts simple binary logic (Yes/No, Spam/NotSpam, Healthy/Sick)
# into numbers an ML algorithm can actually read (always 0/1, since it marks the
# most frequent value as 0 and the next as 1). So we'll see spam = 0 and safe = 1,
# because spam appears twice here, while safe only appears once.

indexed_df = indexer.fit(df).transform(df)
# .fit() lets it actually learn the rule — fit() studies the rule, it doesn't change
# anything by itself!
# .transform() applies that learned rule and actually converts the data into 0s and 1s.

indexed_df.select("Status", "label", "EmailText").show()

"""
+------+-----+--------------------+
|Status|label|           EmailText|
+------+-----+--------------------+
|  spam|  0.0|Congratulations! ...|
|  safe|  1.0|Hey, are we still...|    <---- automatically classified spam as 0,
|  spam|  0.0|Urgent: Claim you...|          safe as 1.
+------+-----+--------------------+
"""
```

We'll also learn to use linear regression, logistic regression, and more in `pyspark`... because apparently `scikit-learn` just wasn't enough. Even so, I can almost guarantee: you'll **never** use `pyspark.ml` most of the time — not unless your company is building a time machine, or a database that tracks every person on Earth.

```python
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("TryinSparkForReg").master("local[*]").getOrCreate()

columns = [
    "DaysActive",
    "SupportTickets",
    "ProjectsCompleted",
    "MonthlyHours"
]

raw_data = [
    [365, 1, 12, 45.5],
    [30, 8, 1, 2.0],
    [120, 4, 5, 12.5],
    [250, 2, 9, 32.0],
    [15, 9, 0, 1.0],
    [500, 0, 18, 60.0],
    [90, 6, 3, 8.5],
    [180, 3, 7, 22.0],
    [45, 7, 2, 4.0],
    [400, 1, 15, 54.5],
    [220, 2, 8, 29.5],
    [70, 5, 2, 7.5],
    [330, 1, 13, 42.0],
    [140, 4, 5, 15.0],
    [20, 10, 0, 0.5],
    [600, 0, 20, 72.0],
    [275, 2, 10, 35.0],
    [110, 5, 4, 10.0],
    [50, 7, 1, 3.5],
    [390, 1, 14, 50.0],
    [160, 3, 6, 18.0],
    [85, 6, 2, 6.5],
    [460, 0, 17, 58.5],
    [310, 2, 11, 39.5],
    [135, 4, 5, 14.0],
    [25, 9, 0, 1.5],
    [520, 0, 19, 65.0],
    [200, 3, 8, 25.5],
    [75, 6, 2, 7.0],
    [350, 1, 12, 47.0],
]

df = spark.createDataFrame(data=raw_data, schema=columns)

assembler = VectorAssembler(inputCols=["DaysActive", "SupportTickets", "ProjectsCompleted"], outputCol="feature")
# Prepares our DataFrame to be used by an ML algorithm.

ready_to_train_df = assembler.transform(df)
# The DataFrame, now with the VectorAssembler's addition.

train_df, test_df = ready_to_train_df.randomSplit([0.8, 0.2], seed=9)
# This part matters a lot, so let's slow down. First, we randomly split the rows —
# in our case, 80% go into train_df and 20% go into test_df. A different example
# might look like:
# bucket_a_df, bucket_b_df, bucket_c_df = ready_to_train_df.randomSplit([0.6, 0.3, 0.1], seed=1)
# That splits the rows into 3 buckets: 60% into bucket_a_df, 30% into bucket_b_df,
# and 10% into bucket_c_df.
#
# But why do we even do this? Because when we call .fit(), a machine learning model
# might look at thousands of rows and take a lazy shortcut: instead of learning the
# actual mathematical pattern, it simply memorizes the correct answers by heart.
# That's called overfitting. A model that's memorized the training data performs
# flawlessly during training, but fails miserably in the real world.
#
# To catch a model that's memorizing, we hide 20% of the data in test_df. The model
# never sees these rows during its "study phase." Once it's trained, we force it to
# predict outcomes for test_df instead.
#
# If the model memorized the training set, it won't know what to do with the new
# test rows, and its accuracy there will plummet. High training accuracy + low test
# accuracy = it cheated!

lr_reg = LinearRegression(featuresCol="feature", labelCol="MonthlyHours")
# This does nothing on its own — it just prepares the blueprint, so when it hears
# the order, it can act immediately.

model = lr_reg.fit(train_df)
# Here's the order. It does everything we learned about linear regression. Under
# the hood, a few things happen:
# 1) The hypothesis — same idea we already know: y_pred = X(feature vector) @ w + b
# 2) Loss — the engine compares y_pred to our y_true (here, that's MonthlyHours):
#    error = y_pred - y_true
# 3) Distributed Gradient Descent — Spark distributes the gradient descent work and
#    tries to minimize the model's loss, slowly shifting the line into position.
# 4) The birth of our model — after thousands of loops of "pregnancy," gradient
#    descent gives birth to our majestic model! (Quick warning: there's always a
#    chance it just memorized the answers, a.k.a. overfitting, instead of learning
#    the true pattern — exactly why we kept that 20% test_df hidden away, to check
#    for that in the next step.) But hey, at least we've got our LinearRegressionModel
#    safe and sound, right?

print(f"That's the model weight: {model.coefficients}")
# Prints our majestic weight (w).
print(f"That's the model bias: {model.intercept}")
# Prints our majestic bias (b).

prediction = model.transform(test_df)
prediction.select(F.col("MonthlyHours"), F.col("prediction"), F.col("feature")) \
    .show(truncate=False)
# truncate=False shows the full table, without cutting anything off if it's too wide.

# We'll evaluate the RMSE here (a.k.a. Root Mean Squared Error) -> (the average):
evaluator_rmse = RegressionEvaluator(labelCol="MonthlyHours", predictionCol="prediction", metricName="rmse")
# Here we test our sweetheart and see whether it was actually cooking, or just
# memorizing everything by heart — throwing out straight answers that deserve
# kudos, or lying with zero real confidence.

rmse = evaluator_rmse.evaluate(prediction)
# This brings the error back down to earth and tells us, on average, exactly how
# many real hours our model is missing the target by.

# Evaluate as MSE (variance of error) -> the full-blown error:
evaluator_mse = RegressionEvaluator(labelCol="MonthlyHours", predictionCol="prediction", metricName="mse")
mse = evaluator_mse.evaluate(prediction)
# We check the Mean Squared Error too, so if it's wrong, we see it twice as clearly.

print(f"The Mean Squared Error (MSE) is: {mse:.3f} hours2")  # <---- hours2 = hours squared (low budget notation, I know)
print(f"The Root Mean Squared Error (RMSE) is: {rmse:.3f} hours")
print(f"On average, our model's predictions are off by exactly {rmse:.2f} hours")

"""
That's the model weight: [0.12400231137796426,0.10569249631078974,0.34727271149548283]

That's the model bias: -3.338007427680129

Output of the prediction preview:
+------------+------------------+----------------+
|MonthlyHours|prediction        |feature         |
+------------+------------------+----------------+
|12.5        |13.701403480396156|[120.0,4.0,5.0] |
|60.0        |64.9140570682207  |[500.0,0.0,18.0]|
|22.0        |21.73039508975419 |[180.0,3.0,7.0] |
|72.0        |78.0088336290081  |[600.0,0.0,20.0]|   <---- not too far off...
|25.5        |24.557714028808956|[200.0,3.0,8.0] |
+------------+------------------+----------------+

The Mean Squared Error (MSE) is: 12.532 hours2
The Root Mean Squared Error (RMSE) is: 3.540 hours
On average, our model's predictions are off by exactly 3.54 hours
"""
```

Now let's do a small project — it might feel like a lot, but it's the same idea as above, just brought together. (Fake it till you make it, we're almost there.)

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import functions as F
from pyspark.ml.classification import LogisticRegression

spark = SparkSession.builder.appName("SmallProject").master("local[*]").getOrCreate()

data_list = [
    (365, 45.5, "Active"),
    (30, 2.0, "Canceled"),
    (120, 12.5, "Canceled"),
    (250, 32.0, "Active"),
    (15, 1.0, "Canceled"),
    (500, 60.0, "Active"),
    (90, 8.0, "Canceled"),
    (180, 20.0, "Active"),
    (45, 3.5, "Canceled"),
    (400, 55.0, "Active"),
    (60, 5.0, "Canceled"),
    (300, 40.0, "Active"),
    (10, 0.5, "Canceled"),
    (220, 28.0, "Active"),
    (75, 6.0, "Canceled"),
    (340, 48.0, "Active"),
    (25, 1.5, "Canceled"),
    (270, 35.0, "Active"),
    (5, 0.2, "Canceled"),
    (410, 58.0, "Active"),
]
columns = ["DaysActive", "MonthlyHours", "Subscription"]

df = spark.createDataFrame(data=data_list, schema=columns)

indexer = StringIndexer(inputCol="Subscription", outputCol="label")
indexed_df = indexer.fit(df).transform(df)
# Again, .fit() learns the rule, and .transform() applies the new column — in our
# case, "label", which ends up 0, 1, 1, 0...

indexed_df.select(
    F.col("DaysActive"), F.col("Subscription"), F.col("label")
).show()
# Active and Canceled each show up 10 times here — an exact tie. When there's a tie,
# StringIndexer breaks it alphabetically by default, so Active (A comes before C)
# gets 0, and Canceled gets 1.

assembler = VectorAssembler(
    inputCols=["DaysActive", "MonthlyHours"], outputCol="features")
# VectorAssembler again, turning DaysActive and MonthlyHours into a single 2-column vector.

ready_ml_train = assembler.transform(indexed_df)
# Appends the new "features" column.

# And now the sad plot twist: after all that logistic regression theory, we just
# have to write boilerplate.

train_df, test_df = ready_ml_train.randomSplit([0.8, 0.2], seed=1)
# Same train_df / test_df idea as before.

lr = LogisticRegression(featuresCol="features", labelCol="label")
# This doesn't change anything yet, it just sets up the blueprint — like telling it:
# "when we start, look inside the `features` column for the input vector," and also,
# "look inside the `label` column to find the true answer key (y) to check against."
# Right now, `lr` is just an untrained blueprint sitting in your driver's memory,
# waiting for orders.

model = lr.fit(train_df)
# The order. This tells the model to start learning (remember, we only gave it 80%
# of the data). This is where our linear algebra skills would come in — sadly we
# won't use them by hand, but we understand what's happening under the hood: it
# uses the classic logistic regression hypothesis, z = w1*DaysActive + w2*MonthlyHours + b,
# passes z through the sigmoid function to get a probability, compares that
# probability to the true labels (our 0, 1, 1, 0...) using Binary Cross-Entropy, and
# runs Distributed Gradient Descent to minimize the loss. Once the optimization loop
# finishes, Spark locks the finalized weights and bias into a new static object
# called `model` (a LogisticRegressionModel), ready to make predictions.

prediction = model.transform(test_df)
# Checking for fraud (a.k.a. overfitting).

print(model.coefficients)
# The `w`, called coefficients — the weight it used to reach its answer.

print(model.intercept)
# The `b`, called intercept — the bias it used to reach its answer.

prediction.select("features", "label", "probability", "prediction").show(truncate=False)
# truncate=False prevents Spark from cutting the row short just because it's long. E.g.:
#
# without truncate=False:
# +----+------------------+
# |name|text              |
# +----+------------------+
# |Bob |hello this is lo..|
# +----+------------------+
#
# with truncate=False:
# +----+---------------------------------+
# |name|text                             |
# +----+---------------------------------+
# |Bob |hello this is long, but he let it|
# +----+---------------------------------+

"""
Output of the DaysActive/Subscription/label preview:

+----------+------------+-----+
|DaysActive|Subscription|label|
+----------+------------+-----+
|       365|      Active|  0.0|
|        30|    Canceled|  1.0|
|       120|    Canceled|  1.0|
|       250|      Active|  0.0|
|        15|    Canceled|  1.0|
|       500|      Active|  0.0|
|        90|    Canceled|  1.0|
|       180|      Active|  0.0|
|        45|    Canceled|  1.0|
|       400|      Active|  0.0|
|        60|    Canceled|  1.0|
|       300|      Active|  0.0|
|        10|    Canceled|  1.0|
|       220|      Active|  0.0|
|        75|    Canceled|  1.0|
|       340|      Active|  0.0|
|        25|    Canceled|  1.0|
|       270|      Active|  0.0|
|         5|    Canceled|  1.0|
|       410|      Active|  0.0|
+----------+------------+-----+

Model weight: [-0.29267235252353213,-1.670299554073144]
Model bias:   71.06792287221867

Output of prediction:
+------------+-----+---------------------------+----------+
|features    |label|probability                |prediction|
+------------+-----+---------------------------+----------+
|[15.0,1.0]  |1.0  |[5.855782687144748E-29,1.0]|1.0       |
|[220.0,28.0]|0.0  |[1.0,0.0]                  |0.0       |
+------------+-----+---------------------------+----------+
"""
```

And the final boss — because life without a few harder days isn't much of a life. But this one is genuinely useful (in my opinion), because nobody wants to write `.fit().transform()` all day long. That's why, to save us (or kill us — depends how fast the concept clicks) comes **Pipelines**! Haven't you heard? Don't you know? Weren't you told from the very beginning that pipelines are important helpers? No? Well, me neither — now I'll learn it and pass it on. (Time-skip of... some length, since I didn't count.)

Anyway, to understand a hard concept, let's use the famous StoryMode:

```
Once Shrimp's pay got cut by 30%, he was sad. That's why he started giving it his
absolute best every day, constantly helping his boss. After a while, the boss
noticed how hard Shrimp was working, and raised his pay back to normal. But he also
handed Shrimp a devastating new task.

He said: "Great job! Now, every night at 1:21 AM, under the full moon and the
aurora veil, our production server receives a list of 10,000 brand-new users. I
want your model to predict who's going to cancel. Otherwise... Shrimp's pay won't
just get cut — it'll vanish. Just like the Florentine Diamond! (I didn't steal it,
so don't send the detectives, please.)"

The raw incoming table looked like this (just a preview of the 10k rows):

+------------+------------+------------+
|DaysActive  |MonthlyHours|Subscription|
+------------+------------+------------+
|14          |0.5         |Active      |
|280         |31.2        |Canceled    |
+------------+------------+------------+

Shrimp saw an opportunity to slack off, but life doesn't work that way. So he
endured a brutal routine for a month: waking up at 1:21 AM every night to write
manual transformation code for the midnight data drop.

As if writing indexer.transform() and assembler.transform() by hand every single
night wasn't strain enough, one night the boss added a brand-new feature to the
incoming stream: SupportTickets.

Because Shrimp was doing everything manually, his entire code crashed. The matrix
dimensions broke. He had to stay up until 4:00 AM changing column arrays,
recreating the VectorAssembler, tracking down where the columns went, and
rewriting the whole production script.

Then, one sunny Friday, under a rare parhelion, a circumhorizontal arc, and a sun
halo, he remembered the timeless words of Benjamin Franklin: "Use pipelines... or
say 'it works on my machine' while confidently denying every accusation."

So Shrimp tried it. He built an enterprise Pipeline. It acted like an automated
factory conveyor belt, locking the StringIndexer, the updated VectorAssembler (now
smoothly holding all 3 columns), and the trained LogisticRegressionModel into a
single saved file on disk.

The next night at 1:21 AM, Shrimp didn't rewrite a thing. His production code
shrank to exactly two lines:

# 1. Load the entire conveyor-belt structure from disk
loaded_pipeline = PipelineModel.load("models/shrimp_enterprise_pipeline")

# 2. Drop the raw, messy data onto the belt. It handles indexing and vectors automatically!
predictions = loaded_pipeline.transform(new_raw_data)

No more manual column adjustments, no more 2:00 AM data-engineering panics. The
conveyor belt processed the 10,000 rows flawlessly.

That's how Shrimp got a massive raise, endless kudos, and a glorious promotion: he
became an official Plumber with an electrician certificate.
```

*(I used AI for grammar here — perfectly normal, since I'll use it for grammar across the whole thing anyway when I put the polished version on GitHub.)*

But let's get properly deep with pipelines, since we'll actually need them.

First, we need to understand the two types of pipeline stages:

1. **Transformers (the workers)** — a tool that takes a `DataFrame`, does some math or string manipulation, and appends a new column to it. Transformers don't need to study anything first; they just transform blindly. For example, `VectorAssembler` is a pure Transformer — it doesn't need to "learn" anything to pack columns into a vector, it just blindly combines them.
2. **Estimators (the students)** — a tool that must scan your data first, to learn a rule, a vocabulary, or a set of weights, before it can do its job. The code tell: you must call `.fit()` on them first — calling `.fit()` on an Estimator gives birth to a Transformer (a Model). For example, `StringIndexer` is an Estimator: it can't index your strings until it reads the column, counts frequencies, and builds its alphabetical tie-breaker dictionary. `LogisticRegression` and `LinearRegression` are Estimators too — they can't predict anything until they run gradient descent on your training data to find the weights.

Clearly you didn't understand a thing, so let's move forward — by redoing the small logistic regression project from before, this time with a real Pipeline:

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import functions as F
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("SmallProject").master("local[*]").getOrCreate()

data_list = [
    (365, 45.5, "Active"),
    (30, 2.0, "Canceled"),
    (120, 12.5, "Canceled"),
    (250, 32.0, "Active"),
    (15, 1.0, "Canceled"),
    (500, 60.0, "Active"),
    (90, 8.0, "Canceled"),
    (180, 20.0, "Active"),
    (45, 3.5, "Canceled"),
    (400, 55.0, "Active"),
    (60, 5.0, "Canceled"),
    (300, 40.0, "Active"),
    (10, 0.5, "Canceled"),
    (220, 28.0, "Active"),
    (75, 6.0, "Canceled"),
    (340, 48.0, "Active"),
    (25, 1.5, "Canceled"),
    (270, 35.0, "Active"),
    (5, 0.2, "Canceled"),
    (410, 58.0, "Active"),
]
columns = ["DaysActive", "MonthlyHours", "Subscription"]

df = spark.createDataFrame(data=data_list, schema=columns)

train_df, test_df = df.randomSplit([0.8, 0.2], seed=1)
# We'll need this either way.

indexer = StringIndexer(inputCol="Subscription", outputCol="label")
# indexed_df = indexer.fit(df).transform(df)  <- we don't need this line anymore
assembler = VectorAssembler(
    inputCols=["DaysActive", "MonthlyHours"], outputCol="features")
# ready_ml_train = assembler.transform(indexed_df)  <- we don't need this line anymore
lr = LogisticRegression(featuresCol="features", labelCol="label")

# Now, our pipeline:
pipeline = Pipeline(stages=[indexer, assembler, lr])
# `pipeline` is just an Estimator — an empty shell for now, since it still needs
# .fit() to actually learn anything.
# The order of `stages` matters! Raw DataFrame --> [Stage 0: Indexer] -->
# [Stage 1: Assembler] --> [Stage 2: Logistic Regression] --> Prediction.
# Why does it matter? If you tried stages=[lr, indexer, assembler], the script would
# break at initialization, because `lr` looks for a "features" column and a "label"
# column that don't exist yet — those only get created by the indexer and assembler
# stages ahead of it.
# ("Initialization" — a fancy ML word. Word-by-word: before it learns anything, a
# model's "brain" is empty. But mathematically it can't really be void — it needs
# *some* starting values for w, b, etc. Those starting values are what we call
# "initialization.")

pipeline_model = pipeline.fit(train_df)
# Running pipeline.fit(train_df) isn't just one command — Spark loops through your
# `stages=[indexer, assembler, lr]` list under the hood, managing all the .fit() and
# .transform() calls automatically so you don't have to.
# Remember: we use .fit() to make the model learn.

prediction = pipeline_model.transform(test_df)
# .transform() runs the trained model on new data, to see how it does in a
# real-world situation — applying the same string conversions, vector assembly, etc.

lr_model = pipeline_model.stages[2]
# Grabs stage index 2 of our pipeline. Since we wrote
# stages=[indexer, assembler, lr], and counting starts at 0, our third object (index 2)
# is the LogisticRegression stage.
# We do this just to peek at the weight and bias it learned.

print(f"Weight of the model is: {lr_model.coefficients}")
print(f"Bias of the model is: {lr_model.intercept}")

prediction.select("features", "label", "probability", "prediction").show(truncate=False)

pipeline_model.write().overwrite().save("/home/shuposhuposhrimpo/doomed_models")
# Saves your model to whichever directory you choose — I chose my majestic
# "/home/shuposhuposhrimpo/doomed_models".

"""
Weight of the model is: [-0.29267235252353213,-1.670299554073144]
Bias of the model is: 71.06792287221867

Output of prediction:
+------------+-----+---------------------------+----------+
|features    |label|probability                |prediction|
+------------+-----+---------------------------+----------+
|[15.0,1.0]  |1.0  |[5.855782687144748E-29,1.0]|1.0       |
|[220.0,28.0]|0.0  |[1.0,0.0]                  |0.0       |
+------------+-----+---------------------------+----------+
"""
```

Now imagine another 10k lines of tax bills arrive — how do we evade them? (Don't actually evade them, or we'll get a [punishment](https://www.investopedia.com/terms/t/taxevasion.asp).) Jokes aside — another 10k lines arrived. What do we do? Just this:

```python
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

spark = SparkSession.builder.appName("ExistingModel").master("local[*]").getOrCreate()

scary_new_10k_lines = spark.read.csv("path_to_new_10k_data.csv", header=True, inferSchema=True)
# header=True should almost always be set, or your column names become part of the
# 10k lines... turning them into 10k-and-1 lines.
# inferSchema will automatically detect what's a string, int, float, and so on —
# without it, every column defaults to string.

loaded_pipeline = PipelineModel.load("/home/shuposhuposhrimpo/doomed_models")
predictions = loaded_pipeline.transform(scary_new_10k_lines)

# 5. Save your predictions to a database or disk for the business team
predictions.select("DaysActive", "MonthlyHours", "probability", "prediction") \
    .write \
    .mode("overwrite") \
    .parquet("outputs/shrimp_predictions")  # <---- another folder, where the output
    # ("DaysActive", "MonthlyHours", "probability", "prediction") gets saved

# That's how we reuse an already-trained model.
```

Before starting the last project, let me teach you (and myself) how to pull an SQL table from PostgreSQL, MySQL, and so on, in Python, so you can actually work with them:

```python
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

spark = SparkSession.builder.appName("ExistingModel").master("local[*]").getOrCreate()

# PostgreSQL way
df = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/<your_database>") \
    .option("dbtable", "<the_table_you_want_to_take>") \
    .option("user", "<your_username>") \
    .option("password", "<your_password>") \
    .option("driver", "org.postgresql.Driver") \
    .load()
# That's how you pull a SQL table from psql (now you can play with it as much as you want).
# Run this in the terminal:
# spark-submit --packages org.postgresql:postgresql:42.7.2 your_file.py
# (instead of "python3" — or "python4" by the time you're reading this)

# MySQL way
df = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:mysql://localhost:3306/<your_database>") \
    .option("dbtable", "<your_table>") \
    .option("user", "<your_username>") \
    .option("password", "<your_password>") \
    .option("driver", "com.mysql.cj.jdbc.Driver") \
    .load()

# spark-submit --packages mysql:mysql-connector-java:8.0.33 your_file.py
```

Scary part incoming! Get the napkins ready, the towels, even your cat — because the tears will overflow and the house will get [flooded](https://dictionary.cambridge.org/dictionary/english/flood). *(Thank me later.)*

Scary project incoming, too. But this is where we bury the last PySpark DataFrame — so stand up, hero, we've barely started.

```python
# We have around 50k rows here (not much at all), but here's a preview (20 rows):
# +-------+--------------+----------------------+--------------+-------------+
# |eval_id|interaction_id|contains_hallucination|toxicity_score|user_feedback|
# +-------+--------------+----------------------+--------------+-------------+
# |      2|         53017|                  true|          0.75|            2|
# |   2111|         75751|                  true|          0.50|            3|
# |   2113|        167134|                  true|          0.69|            3|
# |   2114|        190177|                  true|          0.58|            3|
# |   2115|        203665|                  true|          0.97|            1|
# |      5|          8159|                 false|          0.40|            2|
# |   2110|         39865|                 false|          0.43|            3|
# |   2121|         80056|                 false|          0.39|            3|
# |   2126|          6546|                 false|          0.32|            4|
# |   2139|         56954|                 false|          0.43|            2|
# |   2202|        112381|                 false|          0.48|            3|
# |   2203|        210353|                 false|          0.48|            3|
# |   2208|          8294|                 false|          0.31|            3|
# |   2214|        186371|                 false|          0.38|            4|
# |   2219|        187452|                 false|          0.33|            4|
# |   2220|        128637|                 false|          0.44|            4|
# |   2223|        179819|                 false|          0.41|            2|
# |   2118|        100138|                  true|          0.57|            3|
# |   2119|        128734|                  true|          0.88|            1|
# |   2122|         64222|                  true|          0.65|            1|
# +-------+--------------+----------------------+--------------+-------------+

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression

spark = (
    SparkSession.builder.appName("LastPySparkProject").master("local[*]").getOrCreate()
)  # <--- for a last joke: run 5M lines on an executor and call it "cluster computing"

# This time we won't use a Python list, since it'd be ridiculous for a company to
# hand you 1M lines in one giant Python list — instead, we'll pull directly from an
# SQL table.
df = (
    spark.read.format("jdbc")
    .option("url", "jdbc:postgresql://localhost:5432/ai_sql_lab")
    .option("dbtable", "ai_evaluations")
    .option("user", "<my_username>")
    .option("password", "<my_password>")
    .option("driver", "org.postgresql.Driver")
    .load()
)

cols = [
    "eval_id",
    "interaction_id",
    "contains_hallucination",
    "toxicity_score",
    "user_feedback",
]

# Start empty (no rules built yet)
condition = None

for c in cols:
    # Check if the current column is NULL (e.g., A IS NULL)
    expr = F.col(c).isNull()

    # Build a big OR chain, one column at a time: first loop -> condition = A IS NULL,
    # next loops -> condition = (A IS NULL) OR (B IS NULL) OR ...
    condition = expr if condition is None else condition | expr

df.filter(condition)
# (This particular line's result isn't kept anywhere — it's just here to show what
# .filter(condition) does, before we actually use it below.)

# Quick checkup, to see what was null (scroll to the output to see which rows were
# affected, and how we fixed things before training):
null_parts = df.filter(condition).select(cols).show(truncate=False)

fixed_df = df.dropna(subset=[
    "eval_id",
    "interaction_id",
    "contains_hallucination",
    "toxicity_score",
    "user_feedback"
])
# Drops any row where one of these is NULL. Filling with 0 instead would be risky —
# imagine seeing toxicity_score = 0.11 next to a user_feedback of 0. Confusing,
# since a toxicity of 0.9 clearly maps to a 0 or 1 verdict, but what does 0.11 even
# mean if the surrounding context was actually missing?

fixed_df = fixed_df.withColumn(
    "contains_hallucination", F.col("contains_hallucination").cast("int")
)
# An extra step, because StringIndexer can't take raw boolean logic as input — you
# have to convert it to a plain string or int first. This line turns false into 0
# and true into 1.
# Now we have:
# +-------+--------------+----------------------+--------------+-------------+
# |eval_id|interaction_id|contains_hallucination|toxicity_score|user_feedback|
# +-------+--------------+----------------------+--------------+-------------+
# |      2|         53017|                     1|          0.75|            2|
# |   2111|         75751|                     1|          0.50|            3|
# +-------+--------------+----------------------+--------------+-------------+
# (first two rows shown — true became 1, false became 0, both as ints)

train_df, test_df = fixed_df.randomSplit([0.8, 0.2], seed=9)
# Again, 80% of the rows go to train_df, the rest to test_df.

indexer = StringIndexer(inputCol="contains_hallucination", outputCol="label")
# Heads up: since we already cast contains_hallucination to int above, StringIndexer
# isn't doing much heavy lifting here — but we keep it anyway, so it still maps the
# most common value down to label 0.
assembler = VectorAssembler(
    inputCols=["toxicity_score", "user_feedback"], outputCol="feature"
)
lr = LogisticRegression(featuresCol="feature", labelCol="label")

pipeline = Pipeline(stages=[indexer, assembler, lr])
pipeline_model = pipeline.fit(train_df)
# Our beautiful pipeline.

prediction = pipeline_model.transform(test_df)
# Let's see how confident it is — though it'll be near 100% sure on most of these,
# since the underlying pattern is pretty easy to pick up.

lr_model = pipeline_model.stages[2]
# Grabs stage index 2 of our pipeline — since stages=[indexer, assembler, lr], and
# we count from 0, our third object is the LogisticRegression stage.

print(f"Weight of the model is: {lr_model.coefficients}")
print(f"Bias of the model is: {lr_model.intercept}")
prediction.select("feature", "label", "probability", "prediction").show(
    20, truncate=False
)
# Checking the results.

pipeline_model.write().overwrite().save("/home/shuposhuposhrimpo/doomed_models1")
# Saving the model to our new folder.
# spark-submit --packages org.postgresql:postgresql:42.7.2 your_file.py  <----- use
# this to run the project from your terminal

"""
Output of null_parts:

+-------+--------------+----------------------+--------------+----------------+
|eval_id|interaction_id|contains_hallucination|toxicity_score|user_feedback   |
+-------+--------------+----------------------+--------------+----------------+
|32262  |227067        |false                 |NULL          |1               |
|12268  |19203         |false                 |NULL          |1               |
|13928  |225059        |false                 |NULL          |1               |
|18185  |116302        |false                 |NULL          |1               |
|21245  |165569        |true                  |0.92          |NULL            |
|18376  |182688        |false                 |0.34          |NULL            |
|18599  |78804         |false                 |0.32          |NULL            |
|4639   |46613         |false                 |0.11          |NULL            |
+-------+--------------+----------------------+--------------+----------------+
# Poor 18376, he sees demons while looking at kittens.

Output of prediction:

+----------+-----+-----------------------------+----------+
|feature   |label|probability                  |prediction|
+----------+-----+-----------------------------+----------+
|[0.4,2.0] |0.0  |[1.0,0.0]                    |0.0       |
|[0.05,5.0]|0.0  |[1.0,0.0]                    |0.0       |
|[0.0,4.0] |0.0  |[1.0,0.0]                    |0.0       |
|[0.63,3.0]|1.0  |[1.7050707280202885E-150,1.0]|1.0       |
|[0.1,5.0] |0.0  |[1.0,0.0]                    |0.0       |
|[0.85,1.0]|1.0  |[0.0,1.0]                    |1.0       |
|[0.05,5.0]|0.0  |[1.0,0.0]                    |0.0       |
|[0.17,5.0]|0.0  |[1.0,0.0]                    |0.0       |
|[0.35,5.0]|0.0  |[1.0,0.0]                    |0.0       |
|[0.34,4.0]|0.0  |[1.0,0.0]                    |0.0       |
|[0.06,5.0]|0.0  |[1.0,0.0]                    |0.0       |
|[0.04,5.0]|0.0  |[1.0,0.0]                    |0.0       |
|[0.01,4.0]|0.0  |[1.0,0.0]                    |0.0       |
|[0.08,4.0]|0.0  |[1.0,0.0]                    |0.0       |
|[0.23,3.0]|0.0  |[1.0,0.0]                    |0.0       |
|[0.93,1.0]|1.0  |[0.0,1.0]                    |1.0       |
|[0.02,4.0]|0.0  |[1.0,0.0]                    |0.0       |
|[0.61,2.0]|1.0  |[8.492057237810915E-129,1.0] |1.0       |
|[0.01,5.0]|0.0  |[1.0,0.0]                    |0.0       |
|[0.37,5.0]|0.0  |[1.0,0.0]                    |0.0       |
+----------+-----+-----------------------------+----------+

Weight of the model is: [2559.553094161206,-1.2312502389120585]
Bias of the model is: -1263.9705412483017

They're huge because the underlying logic is extremely predictable — e.g. if
toxicity_score >= 0.5, the label is basically always 1 (contains_hallucination);
if toxicity_score < 0.5, it's basically always 0 (safe). So the model ends up wildly
overconfident given how clean the pattern is.
"""
```

The PySpark journey ends here for now... sadly. But no worries, you'll see it again in our next notes!
