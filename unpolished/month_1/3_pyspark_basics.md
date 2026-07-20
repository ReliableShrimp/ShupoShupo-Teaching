We will learn `PySpark` already at the second month, because the creator of this notes doesn't really likes `pandas`... yeah. 
But that doesn't mean that you shouldn't learn `pandas`! You can try it by yourself and try to manage data with it, yet when the GB get larger... the problem becomes not buying a better machine, but managing those bubbly GB that are waiting for you - sadly. And to manage them, we will need something made for it - Spark (There are others applications, but we will use spark for half of the stuff)

## 1) PySpark basics (SQL)

Since we already learned `Sql` by the last note. We will learn to do the same stuff, but on python - using `PySpark`. 
We have two options to do it, the first is the easiest way, the second will be a tad more nightmare-ish, but you may use both (In the month... 9, maybe - prediction from 29/06/2026)

The easiest way is to write `SQL` on python, thanks to `pyspark.sql`

for example... let's say we have a table:

| Name     | Age | Role               | City          | Country        | Experience_Years |
| :------- | :-- | :----------------- | :------------ | :------------- | :--------------- |
| Alice    | 28  | Data Engineer      | New York      | United States  | 5                |
| Bob      | 35  | Data Scientist     | San Francisco | United States  | 8                |
| Charlie  | 42  | Manager            | London        | United Kingdom | 15               |
| Diana    | 31  | Software Engineer  | Berlin        | Germany        | 6                |
| Evan     | 24  | Analyst            | Toronto       | Canada         | 2                |
| Fiona    | 29  | DevOps Engineer    | Dublin        | Ireland        | 4                |
| Giovanni | 33  | SRE                | Milan         | Italy          | 7                |
| Haruto   | 26  | ML Engineer        | Tokyo         | Japan          | 3                |
| Ilinca   | 38  | Solution Architect | Timisoara     | Romania        | 11               |
| Jin      | 45  | Director of IT     | Seoul         | South Korea    | 18               |


Now we will recreate it on Python:

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


We are going to start from the easy way... firstly we will have to write some magical codes... 

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("TryingPySpark").master("local[*]").getOrCreate()
```

Word-By-Word breakdown:
- `SparkSession` - This is the newest context you can put, but you could write instead of SparkSession many other stuff as: SQLContext, SparkContext, HiveContext <--- Older versions, they are made just for a task, not all of them together as SparkSession. SInce SQLContext can make you write SQL the PySpark way... but nothing more; HiveContext can make you write SQL the good way you know it, but nothing more... soo, instead of choosing, use SparkSession, that uses all of them.
- `.builder` - Think about it as an architect, he garther the information as the appName and master and start making something good out of them 
- `.appName("<name>")` - sets the name of the application, if you are running this on your cluster the name you gave to the application will be shown on the Spark Web UI dashboard.
- `.master("local[ * ]")`  - Highly recommended if you are writing the code on your local machine, because it will use all the cores, instead of maybe using one and becoming really slow.
- `.getOrCreate()` - A magical action command that checks if you have already a session; if you do, it will simply take the existing one, otherwise it will create a new one.


Now we will stare at our table again, and then write:

```python
df = spark.createDataFrame(data=data_list, schema=columns)
```

Word-By-Word breakdown:
- `.createDataFrame()` - It takes a raw Python object (Like a list of tulpes) and chops it up, spreading the rows across all the available CPU cores on your machine (or cluster) so they can be processed in parallel.
- `data=<your_list>` - You have to write the current raw Python object there, since it has to understand the exact object you want it to chop and so on.
- `schema=<the_columns_name>` - You will write the name of your variable that hold the name of the columns here, for example: 
```python
columns = ["Name", "Age", "Role", "City", "Country", "Experience_Years"]
```
since the variable that contains the info for our columns is called columns, we will write in the end `schema=columns`


Then we will write the next:

```python
df.createOrReplaceTempView("people")
```

Word-By-Word breakdown:
- `.createOrReplace` - It will create a table view for you or if it already exist (with the same name) it will overwrite the existing one so we get no random error.
- `TempView` - It stands for Temporary View. "Temporary" because it doesn't save itself in the hard drive or something as this, it only exists while your python script is running, after the python script stops it will vanish since it lives purely in your computer's temporary RAM, not saved on your hard drive.
- `("people")` - That just a name you give to your SQL table, so it get recognized by SQL when you write "`FROM <the_name_you_gave_it>`"


And in the end you will make a variable and write `spark.sql("""
`....`
`""")`

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
    ("Ilinca", 38, "Data Engineer", "Timisoara", "Romania", 11),
    ("Jin", 45, "Director of IT", "Seoul", "South Korea", 18)
]

columns = ["Name", "Age", "Role", "City", "Country", "Experience_Years"]

df = spark.createDataFrame(data=data_list, schema=columns)

df.createOrReplaceTempView("people")

only_data_engineers_df = spark.sql("""
SELECT Name, Age, Role
FROM people
WHERE Role = 'Data Engineer';  
""")

order_by_age_df = spark.sql("""
SELECT Name, Age, City, Country
FROM people
ORDER BY Age DESC;
""")

only_data_engineers_df.show() 
order_by_age_df.show()
# We use "_df" to explicitly understand that this variable is a DataFrame.

"""
output of only_data_engineers:
+------+---+-------------+                                                      
|  Name|Age|         Role|
+------+---+-------------+
| Alice| 28|Data Engineer|
|Ilinca| 38|Data Engineer|
+------+---+-------------+

output of order_by_age:
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
""
```

But sadly, after reading all of this, you have to suffer, because we will learn **DataFrame API**... the painful way.

------------------------------------------------------------------------
## 2. PySpark basics (DataFrame API - A.K.A Programmatic Way)

#### 1) Quick INTRO

Sadly the painful part came... and now you will probably think... why in this world do we need this other version when we can just write SQL in python and be happy forever. Sadly the SQL python way is bad in some aspects where the Programmatic way shines, for example:

1) **SQL Cannot Handle Dynamic Code (Loops and Variables)** -> Let's say that your company has millions of users from many different countries (let's say 84 countries)... and the boss wants you to split the people by their countries... on the SQL version you will have to split them manually... so be prepared and prepare popcorn for the next 84 `WHERE Country = '<Exact_Country>'` you will print manually.  While using Programmatic Way you will just do a loop like this:
   ```python 
 countries = ["United States", "Germany", "Canada", "Japan"]
  
 for c in countries:    
    df.filter(F.col("Country") == c).write.csv(f"{c}_data.csv")
   ```
   (Not all 84 countries listed here, but you understand the idea). And boom! You are done in less than 5 minutes. Expect to get a marriage proposal from your boss, a raise, and much more!

1) **Compile-Time Errors vs. Runtime Disasters** -> Imagine that you make a small typo while writing your cute SQL in python, something as: `'SELECC * FROM users'...` Python will view your SQL query as just a plain, dumb string text. The script will start running, waste 10 minutes, spin up your cluster, and then _boom_, crash in the middle of production because of a typo. While our glorious DataFrame API would never, he will immediately call Guido van Rossum, and sentence you with a cute missing variables or syntax error before the code even runs on the server. So he will break your hopes in the first second, without making you hope for 10 minutes.

2) **Building "Data Pipelines" (Modularity)** -> Sadly you will not write 10 lines of code to do something, sometimes you will need 11 - I am joking, you will need more than hundreds of thousands or more than thousands of thousands. So instead of writing a doom + spaghetti code, you will be able to pass DataFrames into standard Python functions like a pipeline: 

```python
def clean_missing_ages(input_df):                                                    return input_df.filter(F.col("Age").isNotNull())

def calculate_experience(input_df):                                                  return input_df.withColumn("Seniority", F.col("Experience_Years") > 10)

final_df = df.transform(clean_missing_ages).transform(calculate_experience)
```

Don't worry about it for now! Because you will cry later, so save your tears for the next topics.

**Quick summary**:
	**We use SQL when we want to quickly explore data or write a quick report. We use the DataFrame API when we are building enterprise, automated production pipelines that need to be tested, scaled, and automated.**


#### 2) Learning to code

Firstly we spoil ourselves with a table:

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appname("BraceToSuffer").master("local[*]").getOrCreate()

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

Now we will learn basic commands that anybody shall know.

- **`.select()`** - I am sure you understand what it does even without me telling you... It is identical to `SELECT` from SQL. But we will write it in a programmatic way. For example:
  ```python
  # Let us say we want to get just the name, age, and role <- Yes, I use a Oxford      comma there, so I can offically announce that I am a C2 in English. No TOEFL       Needed
  
  name_only_df = df.select(F.col("Name"), F.col("Age"), F.col("Role"))
  # We imported functions from pyspark.sql as F, so we wouldn't have to write         function.sum, function.max, and so on. F.col is a safety net, so instead of        writing df.select(df["Age"]), we will write F.col("Age"), since it is even more    safe.
  
  name_only_df.show()
  # It will show the table
  
  """
  Output of the variable will be:
  
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

  Sooo, that how select work, we don't need any `FROM`, since `df` is already connected to that one table we wrote `data=....` So we get this output, yet we can do many more stuff 

- `.filter()` - This basically the `WHERE` of the programmatic way. We can find the exact age we want, the exact role, and so on. But filter has some logical conditions (`AND`, `OR`, `NOT`), which are written a tad differently: 
     AND = &
     OR = |
     NOT = ~

	  Some examples:
	```python
	
	tech_dept_df = df.filter(F.col("Department") == "Tech") 
	
	tech_dept_df.show()
	
	name_age30l_sal100k_df = df.filter((F.col("Age") < 40) & (F.col("Salary") > 100_000)\
	.select(F.col("Name"), F.col("Age"), F("Salary"))
	.show()
	# The `\` tells Python that the statement continues onto the next line. Without it, Python assumes the statement ends at the end of the current line. Think about it as... wait! Don't read this as finished yet - I'm continuing on the next line. 
	# Remember that we always do \ and then in the next line we start with . and smth, because we are still connected to df
	
	name_age25l_remote5eq_df = df.filter((F.col("Age") < 25) | (F.col("Remote_Days") == 5))\
	.select(F.col("Name"), F.col("Age"), F.col("Remote_Days"))\
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
	
	Output of name_age30l_sal100k_df:
	
	+----+---+------+
	|Name|Age|Salary|
	+----+---+------+
	|Maya| 34|115000|
	+----+---+------+
	
	output of name_age25l_remote5eq_df:
	
	+-----+---+-----------+
	| Name|Age|Remote_Days|
	+-----+---+-----------+
	|  Leo| 25|          5|
	| Noah| 22|          0|
	|Lucas| 31|          5|
	+-----+---+-----------+
	"""
	```
	
	That is how `.filter` works, and it is really useful, since we are going to search for people with a good salary and marry them (My bad - The charisma and beauty criteria has to be high for such a stuff)
	Another important concept is to manage the null, since we are going to meet them... and be ready to bury Python's last null... 
	```python
	
	write_just_the_null_df = df.filter(F.col("Age").isNull())
	# .isNull() means that we are searching for everything that is null (A.K.A nothing) 
	
	  

    ignore_just_the_null_df = df.filter(F.col("Remote_Days").isNotNull())
    ignore_just_the_null_df.show()
    # .isNotNull() means that we are searching for everything that is not null.
    
    
    replace_null_df = df.fillna(value=0, subset=["Remote_Days"])
    replace_null_df.show()
    # .fillna() Replace all NULL values in the Remote_Days column with a default         of 0
    # value = <number> is the number you want to change the null with
    # subset = <column> is the... well... hard to say... but... the collumn you          want to change the value of null
    
    df = df.dropna(subset=["DaysActive", "MonthlyHours", "Subscription"])
    # This part is really useful, because let us imagine we have a row that hasa         NULL, we will simply drop the whole row, because placing 0 instead of NULL         can make the model get wrong answers.

    
	"""
	output of write_just_the_null_df:
		 
	+------+----+--------------------+----------+------+-----------+
	|  Name| Age|                Role|Department|Salary|Remote_Days|
	+------+----+--------------------+----------+------+-----------+
	|Oliver|NULL|Marketing Specialist| Marketing| 58000|          3|
	+------+----+--------------------+----------+------+-----------+
   
   output of ignore_just_the_null_df:
   
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
	
	output of replace_null_df:
		
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
	
	Next topic will be... annoying, so prepare the napkins... (I will prepare the napkins too...)
	anyway... we will not learn `groupBy`, because `groupBy()` is a **destroyer**. It squashes rows down. If you have 100 employees and group by department, you get 3 rows back. The individual employees vanish.
	
	So we will learn the windows! (Which will not destroy any of the columns)

-  `window`
	What is Window? Window is  an opening in the wall or ro- Ah! Got you, jokes aside.
	Let us give a small example... 

	Imagine we have a table:

| Student | Class | Score | Submission_Date | Attended_Reviews |
| :-----: | :---: | :---: | :-------------: | :--------------: |
|  Alice  |   A   |  91   |   2026-06-15    |        4         |
|   Bob   |   A   |  75   |   2026-06-16    |        2         |
| Charlie |   A   |  88   |   2026-06-17    |        5         |
|  Frank  |   A   |  94   |   2026-06-18    |        3         |
|  Grace  |   A   |  82   |   2026-06-19    |        1         |
|  Liam   |   A   |  77   |   2026-06-20    |        2         |
|   Mia   |   A   |  96   |   2026-06-21    |        5         |
|  Noah   |   A   |  84   |   2026-06-22    |        4         |
|  David  |   B   |  95   |   2026-06-15    |        5         |
|   Eve   |   B   |  70   |   2026-06-16    |        0         |
|  Henry  |   B   |  85   |   2026-06-17    |        3         |
|   Ivy   |   B   |  91   |   2026-06-18    |        4         |
|  Jack   |   B   |  64   |   2026-06-19    |        2         |
|  Karen  |   B   |  79   |   2026-06-20    |        2         |
|   Leo   |   B   |  88   |   2026-06-21    |        5         |
|  Mona   |   B   |  73   |   2026-06-22    |        1         |
|  Ruby   |   C   |  67   |   2026-06-16    |        1         |
|   Sam   |   C   |  86   |   2026-06-17    |        3         |
|  Tina   |   C   |  98   |   2026-06-18    |        5         |
|   Uma   |   C   |  80   |   2026-06-19    |        2         |
| Victor  |   C   |  74   |   2026-06-20    |        1         |
|  Wendy  |   C   |  89   |   2026-06-21    |        4         |
| Xavier  |   C   |  83   |   2026-06-22    |        2         |
|  Yara   |   C   |  94   |   2026-06-23    |        5         |

Now let us say that Alice wants to know... how good was she from all the class A.
Since we want to help Alice we will use the window!

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

  
  

window = Window.partitionBy(F.col("Class")).orderBy(F.col("Score").desc())
# Breaking down:
# .partitionBy - You know what it means by learning SQL, but i'll repeat mysef... You will rank, sum, average, and so on, based on a class, since we choose to base ourseves on the "Class" category, pyspark will look at everyone from that class and rank them one by one. If we would write `.partionBy(F.col("Student"))` everybody would get rank = 1, because everyone has a different name from each other.
# .orderBy(F.col("Score").desc()) - It will order them starting from the bigest score and going to the lowest, but since we used `.partitionBy(F.col("Class"))` it will write from biggest to smallest number depending by class, because if class A has 5 guys that scored high and class B has 5 guys that scored big too, the classes will get ranked indipendently


ranking_students_df = df.select(F.col("Student"), F.col("Class"), F.col("Score"), F.rank().over(window).alias("Class scores"))
# .rank() - it will rank something depending by the last atribute we wanted to check, since we checked score as last atribute, it will rank the score. 
# .alias(<name>) - it will rename the function as .rank, .lag, and so on by a name you give to them 

ranking_students_df.show(len(data_list))

  
  

window = Window.orderBy(F.col("Score").desc(), F.col("Student"))
# We changed a bit the window, since we dont need to see the previous of each class independently.


previous_and_next_score_df = df.select(F.col("Student"), F.col("Class"), F.col("Score"), F.lag(F.col("Score")).over(window).alias("Previous score"), F.lead("Score").over(window).alias("Next score"))
# .lag(<collumn_name>) - It shows the previous result of the exact column (NOTE! Remember that the first collumn will always be null, since it had no previous collumn)
# ,lead(<collumn_name>) - It shows the next result of the column you selected. (The last column will always be null, since there is no next collumn)

previous_and_next_score_df.show(len(data_list))


window = Window.partitionBy(F.col("Class"))


max_min_avg_score_for_each_class = df.select(F.col("Student"), F.col("Class"), F.col("Score"), F.col("Submission_Date"), F.max(F.col("Score")).over(window).alias("Best score"), F.min(F.col("Score")).over(window).alias("Worst score"), F.avg(F.col("Score")).over(window).cast("decimal(10,1)").alias("Average score"))
# F.max will look at the class, search for the best score, and print it out. For Class A, it will show 94 (Frank's score) for everyone.
# F.min does the opposite. It scans the class, finds the worst score, and prints it. For Class B, it will show 64 (Jack's score) for everyone.
# F.avg calculates the sum of all scores in that class divided by the number of students. For Class C we have 8 students, so it will be: 67 + 86 + 98 + 80 + 74 + 89 + 83 + 94 = 671, and now since we have 8 students: 671/8 = 83.9 (rounded a tad)
# We add .cast("decimal(10,1)") so Spark doesn't print out 15 ugly and abysmal decimal numbers, but round them.

max_score_for_each_class.show(len(data_list))



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

That was the best, worst, and average score per class, since we use partitionBy on Class

"""
```

And that is where our pain truly begins.
I hope you still have some tears left to spare, because things are only going to get worse. We started with a steep learning curve; now we're about to face a vertical cliff - a wall you'll have to climb with your bare hands.
Your greatest strategy from this point on?

**[Fake it until you make it.](https://en.wikipedia.org/wiki/Fake_it_till_you_make_it)**

That's the Machine Learning cycle:

**Learn → Cry → Learn → Get motivated → Try to solve a problem → Fail → Cry -> Repeat the cycle till something changes**

Eventually, you'll realize that everyone goes through this cycle. The difference is that those who succeed are just luckier or they actually use google or AI in a better way.
So... it's time for **PySpark for Machine Learning**.
We've learned the fundamentals. Now we're immediately jumping into the deep end. Welcome to the DLC on **Nightmare difficulty**, because playing on Easy is no longer an option.
Good luck - You will not need it anyway, because I will get cooked while explaining it, and you just by reading it.


------------------------------------------------------------------------

## 3. PySpark basics for ML engineering

#### 1) Quick INTRO

Before we even start, I have to say that in pyspark `model.fit(X, y)` will not slide anymore. Because PySpark want us to suffer and to learn new ideas (Spark stores data **distributed across many machines**, so having `X` and `y` as two separate objects would constantly require keeping them synchronized across partitions. So it is not worthy for Spark)

Imagine this... 
You are a manager and one of your employee is called Bob. You usually say to bob:
"Come here, I will give you $X$ and $Y$, and you will make line 1 from $X$, match line 1 from $Y$." That the
`pandas` and `scikit-learn` idea. 
Let us say that we have this $X$ and $Y$:

$$X = \begin{bmatrix} 91 & 4 \\ 75 & 2 \\ 88 & 5 \end{bmatrix}, \quad y = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}$$

By using `pandas` and `scikit-learn`, and then `model(X, y)`, we will get:
Row 0 $X$ `[91, 4]`, look over at Row 0 of $y$ `[1]`, match them up instantly, and train the model.

But when the list have 100k + (maybe millions) lines... Bob insists on putting every single row on his own desk before he starts working. Once there are 100,000 rows (or millions), his desk - the computer's RAM - runs out of space.

While, Spark was done intentionally for this, to prevent this from happening, Spark uses more workers and hand the info in a different way, for example:

```Table
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


With the next example we will reveal Spark's secret, and why anyone loves it (Not platonic)

Let us imagine (StoryMode:)

``` StoryMode

Your boss said:
"Shrimp! Immediately put the 100M plates next to the forks! Each fork is already labeled with a name, so don't mess up"

You immediately thought about... `pandas`... `scikit-learn`... that you will use it, solve the problem in 2 minutes... and then your boss will look at you, finally make you the CEO, and pay you all his moneys.
So that what you did, but instead of this, you got your screen blocked (frozen) and the Out of Memmory error. So now you are watching on your phone what to do after gettin and OOM error. After solving the problem, you start thinking... what can help me, and then you saw it... the honey, darling, and love of your life - Spark, waiting for you.

Now you start using Spark, and start thinking, why to like spark but not pandas? Because instead of making the only worker (You), work, spark traumatize even others workers, and gives you just 20M plates, and to the other 4 workers another 20M. What about the forks? It may split to you 40M forks and one aditional plate from one of your workers, so instead of doing all by yourself, you start yelling for others workers, so they pass the exact fork that has the name of the exact plates you owe, and you give what they need (That is actually called 'Shuffling'). After changing the plates and forks a lot of times, you finally have your 20M plates and 20M forks - The computer didn't throw tantrums, so we won. And now you: emplyee #1 has 20M forks and plates, employee #2 has the same amount.... employee #5 has the same ammount. Now the boss is happy and says...:

"Because you had to exchange thousands of forks with the other workers (a shuffle), everyone spent time walking around instead of placing plates. The work still finished successfully, but communication between workers made it slower than if everything had already been in the correct place. The whole process took 30 minutes instead of 30 seconds (It is slower for smaller data - remember that). Could be done faster, you will get your pay cut by 20%"
```

Life is sad, so that how it works, but no worries, let us continue learning.

#### 2. Learning to code

Sadly the AD break ended, and now we have to code, but to say so, the first topic is not even that hard. You just need to change a bit what we used to do. Because in a work environment, we will not get a normal python list slammed in our VS code or Neovim, the Data will be saved in a file where all the info is hold. The file may have 500K lines or even 500M +, we don't know, but I'll explain why using the old method is wrong.

``` python

# imagine the compay_info.txt is a file with 10M lines.
with open("Company_info.txt", "w") as f:
    data_list = [line.split(",") for line in file]
    # we opened the file and tried to pass 10M lines... BOOM! Your laptop turns blue, freezes, and crashes. You just used 100% of your RAM.
    
    
# This part is never reached.    
df = spark.createDataFrame(data_list)

```

You crashed before Spark even got to help you, because you tried to force a massive file through the tiny straw of your single laptop's memory.
But Spark is a savior... so it made this:

THE PRODUCTION WAY: Safe, fast, and scalable

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
    

# Verify it worked by printing the schema and the data df.printSchema() df.show(5)
df.printSchema()
# df.printSchema() will show the structure of the data
df.show(5)
# df.show will print the first 5 columns.
)
```

What is all of this? Great question, I asked myself the same thing. Let's break it down piece by piece.

- **`.master("spark://company-cluster:7077")`** – Connects to the company's Spark cluster instead of running everything on your local computer. This gives Spark access to many workers, CPUs, and much more RAM.

- **`spark.read.csv(<path>)`** – Tells Spark to read a CSV file and create a PySpark DataFrame from it. If Spark is connected to a cluster, it will automatically split the file into pieces and let multiple workers read those pieces in parallel.

- **`"s3://my-academy-bucket/data/global_students.csv"`** – The location of the CSV file. Here it is stored in an Amazon S3 bucket instead of on your local computer - AWS knowledge that we will understand in idk what month.

- **`header=True`** – The first row contains the column names, so it doesn't take the name of the columns as new persons to add to the rows.

- **`inferSchema=True`** – Spark inspects the data and automatically determines each column's data type (Integer, Double, String, Date, etc.) instead of treating everything as text - Yeah, smart, I know.

The important thing to remember is that **you don't manually split the CSV file**. You simply say "read this file," and Spark figures out the best way to distribute the work across the cluster.


But life without problems is like saying beach without sun, soooo, let's go back in the good old times:
Henry the VI was born on December 6 of 1421, at Windsor Castle. He w- Wrong story, yet I remember drinking tea and talking to him. 
Anyway, remember why spark doesn't use model.fit(x, y)? Because it wants DataFrames, as:

```table
+---------+----------------+-------+
| Score   |AttendedReviews | Label |
+---------+----------------+-------+
|   91    |       4        |   1   |
|   75    |       2        |   0   |
|   88    |       5        |   1   |
+---------+----------------+-------+
```

But we the problem is one, can a ML algorithm actually train directly on this DataFrame?
...
...
...
Absolutely Not :)

Because he wants vectors (Yup, the linear algebra we love so much).

```table
+----------------+-------+
| Features       | Label |
+----------------+-------+
| [91,4]         |   1   |
| [75,2]         |   0   |
| [88,5]         |   1   |
+----------------+-------+
```

What happened? The Score and AttendedReviews got united in a vector, but is easy to understand which is which, column 0 is score and column 1 is AttendedReviews. We just united the to make the ML algorithm teach on it.

In coding it will look like:

```python

# +---------+----------------+-------+
# | Score   |AttendedReviews | Label |
# +---------+----------------+-------+
# |   91    |       4        |   1   |
# |   75    |       2        |   0   |    < ---- The table we are using
# |   88    |       5        |   1   |
# +---------+----------------+-------+

from pyspark.ml.feature import VectorAssembler
# We will use pyspark.ml.feature since we want to add ml concepts, and VectorAssembler is literally "Take several columns and pack them into one feature vector."

df = spark.read.csv(
    "s3://my-academy-bucket/data/global_students.csv",
    header=True,
    inferSchema=True
)
# Imagine we already imported spark and so on, and anyway imagine that the DataFrame we made from gloabal_students.csv has many collumns, and even "Score", "AttendedReviews".

assembler = VectorAssembler(inputCols=["Score", "AttendedReviews"], outputCol="Features")
# inputCols= [<columns>] - we write the columns that we want to unite
# outputCol= <name_of_the_new_column> - the name of the new united column

students_df = assembler.transform(df)
# That our new DataFrame, but a ML algorithm can work now!
students_df.show(3)

"""

output of students_df:

+----------------+-------+
| Features       | Label |
+----------------+-------+
| [91,4]         |   1   |
| [75,2]         |   0   |      < ---- Now the two columns became a vector
| [88,5]         |   1   |
+----------------+-------+

"""
```

but before proceeding let's say that $y$ will not be so sweet and be always `np.array([[0], [1], [0]])`, but it will be: `np.array(["Spam", "Safe", "Spam"])` - scary, but it happens.
That why you will use StringIndexer:
```python

from pyspark.sql import SparkSession

from pyspark.ml.feature import StringIndexer


spark = SparkSession.builder.appName("Index-ingStuff").master("local[*]").getOrCreate()

  

raw_data = [

("Congratulations! You won a lottery!", "spam"),
("Hey, are we still meeting for lunch today?", "safe"),
("Urgent: Claim your free cruise ticket now!", "spam")

]
 # Yeah, you will not get a list like this at work, but for trial it is okay.
 
columns = ["EmailText", "Status"]

  
  

df = spark.createDataFrame(data=raw_data, schema=columns) #< ---- We used .createDataFrame again, since we have just a few lines, so worries.

  

indexer = StringIndexer(inputCol="Status", outputCol="label")
# We use StringIndexer to chancge from simply binary logic (Yes/No, Spam/NotSpam, Healthy/Sick) into numbers that an ML algorithm can actually read (Always 0/1, because he marks the most freaquent one with 0 and the next with 1. So we will understand if he writes spam = 0 and safe = 1, because spam appears 2 times, meanwhile safe just once)
  

indexed_df = indexer.fit(df).transform(df)
# We use it fit so it actually learn the rules, cuz fit learns a model the rules, it doesn't change anything!
# transform() = Spark applies those learned rules to actually convert the data into the 0 and 1. 
# So yeah, a model saves yo here, cute.

indexed_df.select("Status", "Label", "EmailText").show()

"""

+------+-----+--------------------+                                             
|Status|Label|           EmailText|
+------+-----+--------------------+
|  spam|  0.0|Congratulations! ...|
|  safe|  1.0|Hey, are we still...|    < ---- The model automatically classified
|  spam|  0.0|Urgent: Claim you...|           the spams as 0 and safe as 1.
+------+-----+--------------------+

"""
```

We will learn how to use even linear regression, logistic regression, and etc on `pyspark`... because `scikit-learn` wasn't enough... even if I can almost say for sure... you will **never** use pyspark.ml most of the times, at least if your company is not making a time machine or a database where they keep all the people on earth.

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
# We prepare our DataFrame to be used by a ML algorithm

ready_to_train_df = assembler.transform(df)
# We make the DataFrame, but we add the VectorAssembler aditions

train_df, test_df = ready_to_train_df.randomSplit([0.8, 0.2], seed=9)
# This part is really important, because I will explain what is happening. First of all, we randomlly split the rows (in our case we did: 80% will go in train_df and 20% of the rows will go in test_df. A different example would be the next:)
# Bucket_A_df, Bucket_B_df, Bucket_C_df = ready_ml_train.randomSplit([0.6, 0.3, 0.1], seed=1) -> This means that we broke the rows in 3 diffrent buckets, bucket_A_df will have 60% of the rows, bucket_B_df will have 30% of the rows, and bucket_C_df will have 10% of the rows. But we have a really important question...
# Why do we even have to do it?
# Because when we use .fit(), a machine learning model might look at thousands of rows and decide to take a lazy shortcut: instead of learning the actual mathematical patterns, it simply memorizes the correct answers by heart. This is called Overfitting. If a model memorizes the training data, it will perform flawlessly during training, but it will fail miserably in the real world.
# To catch a model that is memorizing, we hide 20% of the data in test_df. The model never gets to see these rows during its study phase. Once the model is trained, we force it to predict outcomes for the test_df.
# If the model memorized the training set, it won't know what to do with the new test rows and its prediction accuracy will plummet. If its training accuracy is super high but its test accuracy is low, we know it cheated!
# Man.. I could name this part something cool... yet it is a teaching repo, not a made-for-fun repo... :(


lr_reg = LinearRegression(featuresCol="feature", labelCol="MonthlyHours")
# That does nothing, it just prepare the blueprint, so when it hears the order, it will immediately act

model = lr_reg.fit(train_df)
# Here is our order. It does all we learned of Linear regression. Under the hood many stuff happens, as for:
# 1) The hypotesis:
# Here it will use something we already know, as for: y_pred = X(feature) @ w + b
# 2) Loss:
# To see how much our model messed up, the engine compare the y_pred to our y_true (In our case that the MonthlyHours, as for: error = y_pred - y_true)
# 3) Distributed Gradient Descent:
# Here it will distribute the gradient descent and try to minimize the loss of the model, by slowly shifting the line to its perfect possition
# 4) The birth of our model:
# 5) After thousands of loops of pregnancy, the gradient descent give birth to our majestic model! (A quick warning: there is always a chance it just memorized the answers (overfitting) instead of learning the true patterns, which is exactly why we kept that 20% test_df hidden away to test it in the next step.) But hey! At least we got our LinearRegressionModel safe and sound, right?

  
print(f"That the model weight: {model.coefficients}")
# Prints our majestic weight (w)
print(f"That the model bias: {model.intercept}")
# Prints our majestic bias (b)

prediction = model.transform(test_df)
a_check = prediction.select(F.col("MonthlyHours"), F.col("prediction"), F.col("feature"))\
.show(truncate=False)
# truncate=False wll show the full table, withouth cutting it if too big
  

# We will evaluate the RMSE here (A.K.A Root Mean Square Error) -> (The average):
evaluator_rmse = RegressionEvaluator(labelCol="MonthlyHours", predictionCol="prediction", metricName="rmse")
# Here we test our sweetheart and see if he was actually cooking or just memorizing everything by heart! Is he throwing straight answers that deserve  kudos, or does he lie and have zero real confidence, zero social life, and so on.

rmse = evaluator_rmse.evaluate(prediction)
# This brings the error back down to earth and tells us, on average, exactly how many real hours our model is missing the target by. 


# Evaluate as MSE (Variance of error) -> The full blown error:
evaluator_mse = RegressionEvaluator(labelCol="MonthlyHours", predictionCol="prediction", metricName="mse")
mse = evaluator_mse.evaluate(prediction)
# We check even with the Mean Square Error, so if he gets it wrong we see it 2 times as clear...
  
print(f"The Mean Squared Error (MSE) is: {mse:.3f} hours2") # < ---- hours2 = hours to the second power (Yet low budget)
print(f"The Root Mean Squared Error (RMSE) is: {rmse:.3f} hours")
print(f"On average, our model's predictions are off by exactly {rmse:.2f} hours")


"""

That the model weight: [0.12400231137796426,0.10569249631078974,0.34727271149548283]

That the model bias: -3.338007427680129

output of a_check:
+------------+------------------+----------------+
|MonthlyHours|prediction        |feature         |
+------------+------------------+----------------+
|12.5        |13.701403480396156|[120.0,4.0,5.0] |
|60.0        |64.9140570682207  |[500.0,0.0,18.0]|
|22.0        |21.73039508975419 |[180.0,3.0,7.0] |
|72.0        |78.0088336290081  |[600.0,0.0,20.0]|   < ---- Not too far...
|25.5        |24.557714028808956|[200.0,3.0,8.0] |
+------------+------------------+----------------+

The Mean Squared Error (MSE) is: 12.532 hours2
The Root Mean Squared Error (RMSE) is: 3.540 hours
On average, our model's predictions are off by exactly 3.54 hours

"""
```


Now we will do a small project... that will maybe feel overwhelming... but it has the same idea as the the one above, sooooo, I made it like a quick overall again. (Fake it till you make it, we almost finished.)

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
# Again, we use fit and transform. Remember that the .fit is a way to learn the model, and the transform will add the aditionall collumn to it. (In our case it is the label, which will be equal to 0, 1, 1, 0...

little_look_at_indxdf = indexed_df.select(
F.col("DaysActive"), F.col("Subscription"), F.col("label")
).show()
# Since both appear 3 times; Active will be equal to 0 and Canceled to 1, because to break the tie Spark uses alphabetic order.

  

assembler = VectorAssembler(
inputCols=["DaysActive", "MonthlyHours"], outputCol="features")
# We used VectorAssembler (again) to make from DaysActive and MonthlyHours a vector with 2 columns

ready_ml_train = assembler.transform(indexed_df)
# It will append the new column we just made ("features")

  
# And now the sad plotwist comes... we learned so much logistic regression, and so on... while now we just have to write bolerplates:

train_df, test_df = ready_ml_train.randomSplit([0.8, 0.2], seed=1)
# Same good train_df and test_df

lr = LogisticRegression(featuresCol="features", labelCol="label")
# This part never changes something, it just set up the blueprint, because it is like telling it: "When we will start the process, look inside the column `features` for the input vector matrix" and we also say: "Look inside the `label` column to find the true answer keys ($y$) to evaluate against."
# At this moment, `lr` is just an untrained blueprint sitting in your master machine's memory, waiting for orders.

model = lr.fit(train_df)
# The order. This part tells to the model to start learning (Remember that we gave just 80% of the data to it). This where our lovely linear algebra skill come in! (Sadly we will not use them, but we understand what happens under the hood)
# under the hood they use the classic logistic regression hypotesis formula:      z = w1 * DaysActive + w2 * MonthlyHours + b 
# Then it will pass the z through the sigmoid function which will give us a probability (prob_1), after this, it will compare the prob_1 to the trues probabilities of y (Our beautiful 0, 1, 1, 0...) using the BCR (Binary-Cross-Entropy), then Distributed Gradient Descent
# Once the optimization loop finishes, Spark locks those finalized, perfect weights and biases into a new static object called `model` (a `LogisticRegressionModel`). The training data is thrown away from the engine's active focus. The `model` object now simply holds a frozen, highly accurate mathematical equation ready to make predictions on your test data!

prediction = model.transform(test_df)
# Checking the fraud

print(model.coefficients)
# The `w` is called coefficients. We want to check the weight it used to get such and answer

print(model.intercept)
# The `b` is called intercept. We want to check the bias it used to get this answer

prediction.select("features", "label", "probability", "prediction").show(truncate=False)
# truncate=False - prevents Spark from cutting the message just because it is too long, for example: 
# without truncat=False:

# +----+------------------+
# |name|text              |
# +----+------------------+
# |Bob |hello this is lo..|
# +----+------------------+

# with truncat=False:


# +----+---------------------------------+
# |name|text                             |
# +----+---------------------------------+
# |Bob |hello this is long, but he let it|
# +----+---------------------------------+

"""

output of little_look_at_indxdf:

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


That the models weight: [-0.29267235252353213,-1.670299554073144]
That the models weight: 71.06792287221867

output of prediction:
+------------+-----+---------------------------+----------+
|features    |label|probability                |prediction|
+------------+-----+---------------------------+----------+
|[15.0,1.0]  |1.0  |[5.855782687144748E-29,1.0]|1.0       |
|[220.0,28.0]|0.0  |[1.0,0.0]                  |0.0       |
+------------+-----+---------------------------+----------+

"""
```

And the final boss... because life without some harder days is not life. But! This one is really useful (In my opinion), because nobody wants to write all the day long .fit().transform()... and so on. That why to save us... (or to kill - depends how fast we grasp the concept) comes Pipelines!! Haven't you heard? Don't you know? Haven't  been told from the beginning? That the pipelines are some important helpers? No? Well, me too, now I'll learn and tell... (Timeskip of: idk, cuz I didn't count...)
Anyway... To understand hard concepts, I use the famous StoryMode:

```StoryMode
Once Shrimp's pay was cut by 30%... he was sad. That's why he started doing his best every day. He helped his boss constantly! After a period of time, the boss came by, saw how hard Shrimp was working, and raised his pay by 30% back to normal. But he also gave Shrimp a devastating task...

He said:
"Great job! Now, every night at 1:21 AM, under the full moon and the aurora veil, our production server will receive a list of 10,000 brand-new users. I want your model to predict who is going to cancel. Otherwise... Shrimp's pay will not be cut - it will vanish! Just like the Florentine Diamond! (I didn't steal it, so don't send the detectives, please)."

The raw incoming table looked like this (just a preview of the 10k rows):

Plaintext


+------------+------------+------------+
|DaysActive  |MonthlyHours|Subscription|
+------------+------------+------------+
|14          |0.5         |Active      |
|280         |31.2        |Canceled    |
+------------+------------+------------+


Shrimp saw opportunity... but he really wanted to slack off instead of doing manual labor. But life is life, so he endured a brutal routine for a month: waking up at 1:21 AM to write manual transformation code for the midnight data drop.

As if writing `indexer.transform()` and `assembler.transform()` every single night wasn't straining enough, one night the boss added a new feature to the incoming data stream: `SupportTickets`.

Because Shrimp was doing things manually, his entire code crashed. The matrix dimensions were broken. He had to stay up until 4:00 AM changing column arrays, recreating the `VectorAssembler`, tracking down where the columns went, and rewriting the production script.

But then, one sunny Friday, under a rare Parhelia, a Circumhorizontal arc, and a Sun halo... he remembered the timeless words of Benjamin Franklin:

"Use pipelines... or say 'it works on my machine' while confidently negating every accusation."

So Shrimp tried it. He built an enterprise Pipeline.

The pipeline acted like an automated factory conveyor belt. It locked the `StringIndexer`, the updated `VectorAssembler` (now smoothly holding all 3 columns), and the trained `LogisticRegressionModel` into a single saved file on disk.

The next night at 1:21 AM, Shrimp didn't rewrite a thing. His production code shrank to exactly two lines:

```
-----
```Python

# 1. Load the entire conveyor belt structure from disk
loaded_pipeline = PipelineModel.load("models/shrimp_enterprise_pipeline")

# 2. Drop the raw, messy data onto the belt. It handles indexing and vectors automatically!
predictions = loaded_pipeline.transform(new_raw_data)
```
-----
```StoryMode

No more manual column adjustments, no more 2:00 AM data engineering panics. The conveyor belt processed the 10,000 rows flawlessly.

That is how Shrimp got a massive raise, endless kudos, and a glorious promotion: he became an official Plumber with an electrician certificate.
```

I used AI for grammar, but that normal, since I'll use it for the whole grammar when putting the polished version on `GitHub`.
But we will get deep with pipelines, since we will need them.

First of all, we need to understand the two types of pipelines:
1) The 'Transformers' (The Workers):
     It is a tool that takes a `DataFrame`, does some math or some string manipulation and then append the new column to the DataFrame. The transformers don't need to study anything, they just transform blindly.
     For example:
     `VectorAssembler` is a pure Transformer. It doesn’t need to "learn" anything to pack columns into a vector; it just blindly combines them. 
2) The 'Estimators' (The Students):
     An Estimator is any tool that must scan your data first to learn a rule, a vocabulary, or a set of weights before it can do its job.
     - **The Code Sign:** You must call `.fit()` on them first. Calling `.fit()` on an Estimator gives birth to a **Transformer** (a Model).
     For example:
     `StringIndexer` is an Estimator. It cannot index your strings until it reads the column to count the frequencies and build its alphabetical tie-breaker dictionary.
     `LogisticRegression` or `LinearRegression` are Estimators. They can't predict anything until they run Gradient Descent on your training data to find the weights.


Clearly you didn't understand nothing, so we move forward. But we will redo the old small project we did about LogisticRegression:

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
# We will need that anyway.

indexer = StringIndexer(inputCol="Subscription", outputCol="label")
# indexed_df = indexer.fit(df).transform(df) < -- We don't need it anymore
assembler = VectorAssembler(
inputCols=["DaysActive", "MonthlyHours"], outputCol="features")
# ready_ml_train = assembler.transform(indexed_df) < -- We don't need it anymore 
lr = LogisticRegression(featuresCol="features", labelCol="label")

# Now we do our pipeline
pipeline = Pipeline(stages=[indexer, assembler, lr])
# pipeline is just an Estimator, it is an empty shell for now... since it has to learn by using fit on it.
# Remember that the stages have a sequence, as for:
# Raw DataFrame --> [ Stage 0: Indexer ] --> [ Stage 1: Assembler ] --> [ Stage 2: Logistic Regression ] --> Prediction
# Why it matters? because If you try `stages=[lr, indexer, assembler]`, the script breaks on initialization because the `lr` model looks for a column called `features` and a column called `label` which do not exist yet in the dataframe structure (Since they get created by the indexer and assembler in the first place.
# Initialization - A fancy ML word... translation to word-by-word: Before learning something, the model brain is empty. But from a mathematical look, it can't be 'void', because we have to gice it firstly some values, as: w, b, and etc... This starting values are called 'Initialization'

pipeline_model = pipeline.fit(train_df)
# When you run `pipeline.fit(train_df)`, you are not just running a single command. Spark looks inside your `stages=[indexer, assembler, lr]` list and runs a hidden loop, managing the `.fit()` and `.transform()` calls automatically so you don't have to.
# Remember that we use .fit() to make the model learn

prediction = pipeline_model.transform(test_df)
# Remember that we use .transform() to try the model right now, to see how it will do with a real life situation, and we to add some changes as string manipulations and etc...
  
lr_model = pipeline_model.stages[2]
# This will take the 2 stage of our pipeline, since we placed -> pipeline = Pipeline(stages=[indexer, assembler, lr < -- our second]) we know that our second object is the LogisticRegression (since we start counting from 0)
# We do it just to check the weight, bias...

print(f"Weight of the model is of: {lr_model.coefficients}")
print(f"Bias of the model is of: {lr_model.intercept}")

prediction.select("features", "label", "probability", "prediction").show(truncate=False)

pipeline_model.write().overwrite().save("/home/shuposhuposhrimpo/doomed_models")
# You will save your model in the directory (folder) you choose. I choosed my majestic "/home/shuposhuposhrimpo/doomed_models"
"""

Weight of the model is of: [-0.29267235252353213,-1.670299554073144]            
Bias of the model is of: 71.06792287221867

output of prediction: 

+------------+-----+---------------------------+----------+
|features    |label|probability                |prediction|
+------------+-----+---------------------------+----------+
|[15.0,1.0]  |1.0  |[5.855782687144748E-29,1.0]|1.0       |
|[220.0,28.0]|0.0  |[1.0,0.0]                  |0.0       |
+------------+-----+---------------------------+----------+

"""
```

Now imagine that new data came... another 10k lines of tax bills, so how we evade them? 
(Don't evade them, cuz we will get a [punishment](https://www.investopedia.com/terms/t/taxevasion.asp).) Jokes aside, another 10k lines arrived... what do we do? We just simply do the next:

```python
from pyspark.sql import SparkSession 
from pyspark.ml import PipelineModel

spark = SparkSession.builder.appName("ExistingModel").master("local[*]").getOrCreate()

scary_new_10k_lines = spark.read.csv("path_to_new_10k_data.csv", header=True, inferSchema=True)
# Remember that header=True shall almost always be placed, or your columns name will be part of the 10k line... becoming 10k and 1 line.
#inferSchema will automatically detect what is a string, what an int, what a float, and so on, without it, all your lines will automatically become strings.

loaded_pipeline = PipelineModel.load("/home/shuposhuposhrimpo/doomed_models")
predictions = loaded_pipeline.transform(new_raw_data)

# 5. Save your predictions to database or disk for the business team 
predictions.select("DaysActive", "MonthlyHours", "probability", "prediction") \ .write\
.mode("overwrite")\ 
.parquet("outputs/shrimp_predictions") # < ---- Another folder where the output ("DaysActive", "MonthlyHours", "probability", "prediction") will be added

# That how we use already existing models 

```

Before starting with the last project, I'll teach you (and teach myself) how to extract an SQL table from PostgreSQL, MySQL... on python, so you can work with them:

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
.option("password", "<your_pasword>") \
.option("driver", "org.postgresql.Driver")\
.load()
# That a way to take the an sql table from psql (Now you can play with it as much as you want
# run this in the terminal...: spark-submit --packages org.postgresql:postgresql:42.7.2 your_file.py  <--    instead of python3 (or 4 by the time you read)

# MySQL way
df = spark.read \    
.format("jdbc") \    
.option("url", "jdbc:mysql://localhost:3306/<your_database>") \    .option("dbtable", "<your_table>") \    
.option("user", "<your_username>") \    
.option("password", "<your_password>") \    
.option("driver", "com.mysql.cj.jdbc.Driver") \    
.load()

#spark-submit \--packages mysql:mysql-connector-java:8.0.33 \your_file.py
```

Scary part incoming! Prepare the napkins, the towels, and even your cat! Because tears will overflow and the house will get [flooded](https://dictionary.cambridge.org/dictionary/english/flood) < --- Thanks me later.

Scary project incoming... But so we will bury the last PySpark DataFrame. So, stand up hero, we just started.

```python
# We have... around 50k rows (Not much at all), but that just how it look (20 rows):
#+-------+--------------+----------------------+--------------+-------------+
#|eval_id|interaction_id|contains_hallucination|toxicity_score|user_feedback|
#+-------+--------------+----------------------+--------------+-------------+
#|      2|         53017|                  true|          0.75|            2|
#|   2111|         75751|                  true|          0.50|            3|
#|   2113|        167134|                  true|          0.69|            3|
#|   2114|        190177|                  true|          0.58|            3|
#|   2115|        203665|                  true|          0.97|            1|
#|      5|          8159|                 false|          0.40|            2|
#|   2110|         39865|                 false|          0.43|            3|
#|   2121|         80056|                 false|          0.39|            3|
#|   2126|          6546|                 false|          0.32|            4|
#|   2139|         56954|                 false|          0.43|            2|
#|   2202|        112381|                 false|          0.48|            3|
#|   2203|        210353|                 false|          0.48|            3|
#|   2208|          8294|                 false|          0.31|            3|
#|   2214|        186371|                 false|          0.38|            4|
#|   2219|        187452|                 false|          0.33|            4|
#|   2220|        128637|                 false|          0.44|            4|
#|   2223|        179819|                 false|          0.41|            2|
#|   2118|        100138|                  true|          0.57|            3|
#|   2119|        128734|                  true|          0.88|            1|
#|   2122|         64222|                  true|          0.65|            1|
#+-------+--------------+----------------------+--------------+-------------+

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression

spark = (
SparkSession.builder.appName("LastPySparkProject").master("local[*]").getOrCreate()) # < --- For a last joke: run 5M lines on a executor and call it 'cluster computing'

# This time we will not have a python list, because it would be ridiculus if a company gives you 1M lines in one bif python list, so we will extract an SQL table.
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
    # Check if current column is NULL (e.g., A IS NULL)
    expr = F.col(c).isNull()
    
    # Build a big OR chain step-by-step: First loop: condition = A IS NULL             Next loops: condition = (A IS NULL) OR (B IS NULL) OR ...
    condition = expr if condition is None else condition | expr

# This will keep rows where AT LEAST ONE column is NULL
df.filter(condition)

# Quick check up, to see which is null... (Scroll to the output, so you see which were null, and how we fixed it and started training)
null_parts = df.filter(condition).select(cols).show(truncate=False)

fixed_df = df.dropna(subset=[
    "eval_id",
    "interaction_id",
    "contains_hallucination",
    "toxicity_score",
    "user_feedback"
])

# This will drop the rows where NULL will be found. Because plaicng all to 0 can be risky, because we may see toxicity = 0.11 and user ranking? 0. And that confusing, since a toxicity of 0.9 is of 0 or 1, how can a toxicity of 0.11 be 0??

fixed_df = fixed_df.withColumn(
"contains_hallucination", F.col("contains_hallucination").cast("int")
)

# That an additional step I did, because remember one important concept... StringIndexer can't take BooleanLogic as input, you have to change from boolean to a simple string or int.
# This part will convert from false to 0 and true to 1.
# Now we have:
#+-------+--------------+----------------------+--------------+-------------+
#|eval_id|interaction_id|contains_hallucination|toxicity_score|user_feedback|
#+-------+--------------+----------------------+--------------+-------------+
#|      2|         53017|                  1   |          0.75|            2|
#|   2111|         75751|                  1   |          0.50|            3|
#+-------+--------------+----------------------+--------------+-------------+

# This is the first two rows of the DataFrame on the top. Now true became 1 and false 0 (in int)
# When we will use StringIndexer, there are two scenarios
# The 1 (true) appears more times than 0 (false), so StringIndexer will convert the 1 (true) into 0 and the 0 (false) into 1. 

train_df, test_df = fixed_df.randomSplit([0.8, 0.2], seed=9)
# again we give 80% of the rows to trainz-df and the rest to test_df

indexer = StringIndexer(inputCol="contains_hallucination", outputCol="label")
# I'll let StringIndexer to don't confuse you, but you have to understand that the fixed_df = fixed_df.withColumn("contains_hallucination", F.col("contains_hallucination").cast("int")) already did its job, so String indexer is prety much useless here. But we will keep it, so it changes the most popular BinaryLogic to 0.
assembler = VectorAssembler(
inputCols=["toxicity_score", "user_feedback"], outputCol="feature"
)
lr = LogisticRegression(featuresCol="feature", labelCol="label")

pipeline = Pipeline(stages=[indexer, assembler, lr])
pipeline_model = pipeline.fit(train_df)
# Our beautiful pipeline


prediction = pipeline_model.transform(test_df)
# Let's see how sure he is, even thought... he will be 100% sure for everything, since the logic is too easy.

lr_model = pipeline_model.stages[2]
# This will take the 2 stage of our pipeline, since we placed -> pipeline = Pipeline(stages=[indexer, assembler, lr]) we know that our second object is the LogisticRegression (since we start counting from 0)

print(f"Weight of the model is of: {lr_model.coefficients}")
print(f"Bias of the model is of: {lr_model.intercept}")
prediction.select("feature", "label", "probability", "prediction").show(
20, truncate=False
)
# Checking on his results...

pipeline_model.write().overwrite().save("/home/shuposhuposhrimpo/doomed_models1")
# Saving the model to our new folder.
# spark-submit --packages org.postgresql:postgresql:42.7.2 your_file.py <----------- Use this to start the project in your terminal

"""

output of null_parts
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


output of prediction:

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

Weight of the model is of: [2559.553094161206,-1.2312502389120585]
Bias of the model is of: -1263.9705412483017
They are huge because the logic is really predictable, for example... if toxicity >= 0.5, then it is a 100% = 1 as label, if it is 0.5 < toxicity, it is safe. So the model is way too overconfident with the data.
"""

```

The PySpark journey ended for now... sadly, but no worries, you will see it in our next codes!
