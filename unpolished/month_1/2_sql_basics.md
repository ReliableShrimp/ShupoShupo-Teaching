
This is our first real Data Engineering concept! And before you sit there thinking, "I don't need Data Engineering because I'm going to be an ML Engineer / AI Engineer / Pipeline Plumber / Neuro-Electrician Pilot," let me halt you right there. 
You WILL need `SQL`. Every single job offer nowadays is going to demand it. 

Picture this: 
(Character.ai style)
**the interviewer leans in, looking slightly confused, and asks:**
Interviewer: "Do you know `GNNs` to absolute perfection?"

**You smirk, completely unfazed.** 
You: "Of course. It's a fundamental pillar." 

**He gets a tad surprised and ask you another question**
Interviewer: "What about `SQL`?"

**You laugh, since you ate it yesterday**
You: "Nasty, don't you think?"
(Because in your mind?: GNN = "**G**uys, **N**o **N**otetaking" or maybe "**G**rownups **N**eed **N**aps.")
(And SQL = "**S**uper **Q**uadrillion **L**ines")

**He left the call, block you, report you, send the police at your house, get a prosecutor, and now you have a criminal report of S**uper **Q**uadrillion of **L**ines

Jokes aside, every company needs you to know more than just your narrow slice of the codebase. You can't be an ophthalmologist (an eye specialist) and have absolutely no clue how the stomach works. In production AI, Data Engineering is the stomach that feeds your model's brain. 

So, we will start with the architectural basics of data extraction and keep pushing forward. The concepts aren't complex, so take a breath, relax, and let's get into it.


## 1) Learning SQL concepts and commands:

#### 1. Introduction
Why do we need even `SQL`? What is even `SQL`?
`SQL` (A.K.A **S**tructured **Q**uery **L**anguage) - It is a language made to talk to the Database. 
(What is a Database? - Think about it as a hyper secure wardrobe, you can spear data inside it... but data will not be thrown randomly, the data will be stored in some tables (Think about them as files), they will look smth as:

| id  | name             | formula   | mass         |
| :-- | :--------------- | :-------- | :----------- |
| 1   | Water            | H2O       | 18           |
| 2   | Carbon Dioxide   | CO2       | 44           |
| 3   | Glucose          | C6H12O6   | 180          |
| 4   | BrokenWater      | NULL      | 9999         |
| 5   | FakeMethane      | CH888     | not_a_number |
| 6   | Oxygen           | O2        | 32           |
| 7   | WeirdCompound    | C6H@@12O6 | NULL         |
| 8   | HydrogenPeroxide | H2O2      | 34           |

(8 rows))

If a database just stores data in rows and columns, why can't we just use a folder full of standard CSV files, Excel spreadsheets, or JSON text files?

Because all things in this world have some pros and some cons. A fish can swim fast, but can it climb trees? No? That doesn't mean that they are bad, but like fish, files are made for smaller waters, while databases are made for oceans. And we even have some reasons to say so. Let us look at the reasons:

- **The Search Speed Catastrophe (Indexing)** - 
1) Let us say you have a CSV file with 100M lines, but you need to find the exact data for a user with the ID: `84_292_182`... And as expected, he is on row 84,292,182... so what is the problem? Let us get it! We will use Python to get it... and so Python opens the file (two scenarios are available... either it chokes and throws an Out Of Memory error, or it doesn't choke). Then, it starts from line one... goes to line two... to line three... (A.K.A. a Full Table Scan) and somewhere in the year 2083, it will finish. (It is a joke, but remember that this will be painfully slow). 
2) But with a database: Databases use a concept called Indexing (usually structured as B-Trees). It acts like a highly advanced book index or another method. Instead of reading every row, the database engine skips straight to the exact physical location of that user ID on the hard drive. A lookup that takes minutes in a raw file takes less than 3 milliseconds in a database.

- **The Fraud & Consistency Trap** - 
1) Imagine Account A sends Account B $100. Your script will automatically subtract $100 from Account A... But BOOM! The server lost power. The result? -$100 for user A, and for Account B, the $100 was never written. Cute, but sad :(
2)  Meanwhile, our glorious king of databases guarantees a property called Atomicity. This property treats a series of operations as "All-or-Nothing", so if something fails, it will immediately do a Rollback, undoing the first step completely as if it never happened.

- **Memory Choking** -
1) To open a file with a library like `Pandas`, or by just using `with open(...)`, the machine will use its own RAM... so when we have too many lines that cost many GBs... our machine can just choke on all the data - since it uses all the RAM and gives us an OOM error. (Unless you use PySpark or related libraries that let you split data to other machines by connecting to the master URL.)
2) The database was exactly made to store information there, so it stores data entirely on disk and optimizes the internal cache pools. It allows you to slice, add, take, and do other things with many terabytes of information and so on.

So that the reason why Databases exist. But now we will learn how to control them, with the help of ..` SQL`, later `PySpark` (Joking, because soon after it, you will use `PySpark` too)

#### 2. Commands of basic SQL

Now we start from the easy part, as for...

```SQL
SELECT column_name FROM table_name;   --Always remember to place ; to start it
```

What it means? Let's paste again the table from above:

| id  | name             | formula   | mass         |
| :-- | :--------------- | :-------- | :----------- |
| 1   | Water            | H2O       | 18           |
| 2   | Carbon Dioxide   | CO2       | 44           |
| 3   | Glucose          | C6H12O6   | 180          |
| 4   | BrokenWater      | NULL      | 9999         |
| 5   | FakeMethane      | CH888     | not_a_number |
| 6   | Oxygen           | O2        | 32           |
| 7   | WeirdCompound    | C6H@@12O6 | NULL         |
| 8   | HydrogenPeroxide | H2O2      | 34           |

Firstly we have to know that this is a table and it has its own name, in our case the name of this table is molecules (you usually choose the name when doing a table)

What does the code SELECT means? 
It means to Select one of the columns name so it will print the column, for example:

```SQL
SELECT id, name FROM molecules;

"""
Output:

 id |       name       
----+------------------
  1 | Water
  2 | Carbon Dioxide
  3 | Glucose
  4 | BrokenWater
  5 | FakeMethane
  6 | Oxygen
  7 | WeirdCompound
  8 | HydrogenPeroxide

"""
```

See? It printed just the exact stuff we asked for, what about if we want more stuff... the whole table? then we do:

```SQL
SELECT * FROM molecules;

"""
Output:

 id |       name       |  formula  |     mass     
----+------------------+-----------+--------------
  1 | Water            | H2O       | 18
  2 | Carbon Dioxide   | CO2       | 44
  3 | Glucose          | C6H12O6   | 180
  4 | BrokenWater      | H20000O   | 9999
  5 | FakeMethane      | CH888     | not_a_number
  6 | Oxygen           | O2        | 32
  7 | WeirdCompound    | C6H@@12O6 | 180.5
  8 | HydrogenPeroxide | H2O2      | 34
  
"""
```

 The `*` means **all columns**... But what if we only want to display the `formula` and `mass` columns for the first four molecules? Then we will do:
   
   ```SQL
-- Before we start If WHERE doesn't return anything, check the data type of the column.  
-- If the column stores text (str) instead of numbers (int), you may need quotes, e.g. '4'.

SELECT formula, mass FROM molecules WHERE id <= 4;
   
"""
Output:
   
formula | mass 
--------+------
H2O     | 18
CO2     | 44
C6H12O6 | 180
H20000O | 9999
   
   
"""
   ```

WHERE - It actually means something as 'Where this column is... or has'  something as this, since WHERE can be used in many ways, as for:

```SQL
WHERE mass >= 15; 
-- It will select the rows where this condition is satisfied (mass = bigger or equal to 15)
WHERE mass <= 15; -- It will select the rows where mass is smaller or equal to 15 
WHERE mass = 15  -- It will select the rows where mass is equal to 15
WHERE mass = 15 OR formula = 'CO2'  -- It will select the rows where mass is 15 or where the formula is equal to CO2
WHERE formula = 'H2O' AND mass = 18  -- It will select the rows where the formula is H2O and the mass is equal to 18
WHERE NOT formula = 'H2O' -- It will select the rows where the formula is not H2O and anyway, You can write it as WHERE formula != 'H2O' or 
-- WHERE formula <> 'H20', the result is the same, since mean formula is not equal to...
WHERE mass <= 180 AND NOT formula = 'CO2' -- It will select the rows where the mass is lower or equal to 180 and the formula is not equal to CO2
```

That are some basic logical condition of WHERE.

Now we will learn how to do some tables, because we will need when prototyping or for other tries... Sooo, we can cook by making a table with all the VOCALOIDSSSS, but sadly I will not do that, because we are professional and we are talking about learning. That why we will learn how to create and drop tables while using Vocaloids as example :

```SQL
-- The syntax will look like this:

CREATE TABLE table_name (column_name data_type, column_name data_type, column_name data_type);

-- Now i'll show the example:

CREATE TABLE Vocaloids (name TEXT, height NUMERIC, weight NUMERIC, age INTEGER)

```

But what is a TEXT, a NUMERIC, an INTEGER? (I'll explain later though):

| Column   | Data Type | Think of it as           | Example          |
| -------- | --------- | ------------------------ | ---------------- |
| `name`   | `TEXT`    | A **string**             | `'Hatsune Miku'` |
| `height` | `NUMERIC` | A **decimal number**     | `158.5`          |
| `weight` | `NUMERIC` | A **decimal number**     | `42.3`           |
| `age`    | `INTEGER` | A **whole number (int)** | `16`             |

Now we will teach how to update the empty table by adding values

```SQL

-- Now we will teach another command... but before that, remember that the order we made goes as (name, height, weight, age).

INSERT INTO Vocaloids -- <--- The table name, INSERT INTO table_name
VALUES ('Hatsune Miku', 158, 42.3, 18), ('Kagamine Rin', 152, 43.0, 14), ('Kasane Teto', 159.5, 47.0, 31)

-- Now we have our table:

--      name      | height | weight | age
------------------+--------+--------+-----
-- Hatsune Miku   |  158.0 |   42.3 |  18
-- Kagamine Rin   |  152.0 |   43.0 |  14
-- Kasane Teto    |  159.5 |   47.0 |  31

-- And now... a command that will make you cry... DROP TABLE IF EXISTS... it will delete the entire table from existance...

DROP TABLE IF EXISTS Vocaloids -- <--- Rip Miku, Rin, and Teto...

-- Another command is DELETE (It delets all the rows or just a specific row)

DELETE FROM Vocaloids;

--      name      | height | weight | age
------------------+--------+--------+-----

-- Now this is empty.

-- Or... 
DELETE FROM Vocaloids WHERE age < 18;

--      name      | height | weight | age
------------------+--------+--------+-----
-- Hatsune Miku   |  158.0 |   42.3 |  18
-- Kasane Teto    |  159.5 |   47.0 |  31

-- RIP Teto.
```

But after learning basics, now we will learn this new commands, as:
- LIKE
- BETWEEN
- IS NULL
- IN

We will use a different table for them (There is no meaning behind this choice, I did it so I don't use the same molecule table):
The table name is 'users'

| user_id | first_name | last_name  | age |
| ------: | ---------- | ---------- | --: |
|       1 | Shrimp     | ShupoShupo |  19 |
|       2 | Daniel     | Kovalenko  |  59 |
|       3 | Alex       | Rossi      |  34 |
|       4 | Elena      | Petrenko   |  71 |
|       5 | Mark       | Johnson    |  22 |
|       6 | Sophia     | Müller     |  26 |
|       7 | Luca       | Bianchi    |  26 |
|       8 | Anya       | Novak      |  67 |
|       9 | Kenji      | Tanaka     |  76 |
|      10 | Mia        | Smith      |  54 |

But before that, we will learn ORDER BY, because it will be really useful:

```SQL

SELECT * FROM users ORDER BY age;

"""
Output

 user_id | first_name | last_name  | age
---------+------------+------------+-----
       1 | Shrimp     | ShupoShupo |  19
       5 | Mark       | Johnson    |  22
       6 | Sophia     | Müller     |  26
       7 | Luca       | Bianchi    |  26
       3 | Alex       | Rossi      |  34
      10 | Mia        | Smith      |  54
       2 | Daniel     | Kovalenko  |  59
       8 | Anya       | Novak      |  67
       4 | Elena      | Petrenko   |  71
       9 | Kenji      | Tanaka     |  76

"""
-- If we write just ORDER By <column_name> we will get from smallest to biggest, same if we order by strings, as if we write ORDER BY first_name -> It will start with A and go to Z.
-- Now we try the way around, we try to use another idea:

SELECT * FROM users ORDER BY age DESC;

"""

 user_id | first_name | last_name  | age
---------+------------+------------+-----
       9 | Kenji      | Tanaka     |  76
       4 | Elena      | Petrenko   |  71
       8 | Anya       | Novak      |  67
       2 | Daniel     | Kovalenko  |  59
      10 | Mia        | Smith      |  54
       3 | Alex       | Rossi      |  34
       6 | Sophia     | Müller     |  26
       7 | Luca       | Bianchi    |  26
       5 | Mark       | Johnson    |  22
       1 | Shrimp     | ShupoShupo |  19

"""

-- The difference is that now we went by the logic: biggest to smallest and to a string it would apply as from Z to A.
-- The structure is usually SELECT <> FROM <> WHERE <> (If needed) ORDER BY <>;
```
firstly we start with `LIKE` (LIKE doesn't support integers):

```SQL
SELECT * FROM users WHERE first_name LIKE '%S' -- It will print all the rows in which the first_name ends with the sylable S
SELECT * FROM users WHERE last_name LIKE 'o%' -- It will print all the rows in which the last_name starts with the sylable o
SELECT * FROM users WHERE first_name LIKE '%a%' -- It will print all the rows that have at least a "a" in their name, doesn't matter where, just to have it
```

Since many times the numbers maybe be strings instead of integers (or vice versa) we will use some tricks.
Since there will be [hard times](https://www.youtube.com/watch?v=KpvHi6kqLHI&list=RDKpvHi6kqLHI&start_radio=1) sooo, we prepare you like a warrior to change the values to your necessity! Because there is no second chance (I am joking, but doing mistakes because of something like that is not elegant.) 
But before that we will check the type of our columns, because I suppose you don't want to change an int into an int.

So, you check the type of data of your columns by using:

```SQL
SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'users'; -- Don't chance the column_name to the actual column_name of your table, just let it ths way, so it prints everything down.

"""

 column_name |     data_type          
-------------+---------------------
 user_id     | integer
 first_name  | character varying  < --- That a fancy way of saying text
 last_name   | character varying
 age         | integer
 country     | character varying
 tier        | character varying

"""
```

Now after discovering the outputs we can start with manipulating the data_type:

```SQL

-- To manipulate the datatype, we have to use this -> :: <- this two little dots, because we will write different stuff (I will not show all, because you will not all of that)

-- Let us imagine firstly... We want to find everybody that has the number 3 in their age, we dont care if in the middle or etc, we only care for it to have that 3, but sadly LIKE doesn't support integers, so what we will do?  Let us say that we have the table above (In case you forgot):

--| user_id | first_name | last_name  | age |
--| ------: | ---------- | ---------- | --: |
--|       1 | Shrimp     | ShupoShupo |  19 |
--|       2 | Daniel     | Kovalenko  |  59 |
--|       3 | Alex       | Rossi      |  34 |
--|       4 | Elena      | Petrenko   |  71 |
--|       5 | Mark       | Johnson    |  22 |
--|       6 | Sophia     | Müller     |  26 |
--|       7 | Luca       | Bianchi    |  26 |
--|       8 | Anya       | Novak      |  67 |
--|       9 | Kenji      | Tanaka     |  76 |
--|      10 | Mia        | Smith      |  54 |

-- since LIKE doesn't support integers, we will use the :: method to make all the integers in a text (imagine as a string) and get LIKE to work:

SELECT * FROM users WHERE age::TEXT LIKE '%3%';

"""
Output:

|user_id | first_name | last_name | age | country | tier |  
+---------+------------+-----------+-----+---------+-----+
|      3 | Alex       | Maryna    |  34 | Ukraine | free |

"""

-- We just treated the age as a text, so we could use LIKE on it, because otherwise he would complain about not finding nothing, since he searches for a text, not a integer.
-- Let us say we want to use ORDER BY age... but sometimes the integers are just some hidden texts, so if we ORDER BY age when the age is a text, he will simply do: Hey! A is the first "number" till Z, and then the logic will go as... every number that starts with 1 will be first, then if we have 10000000 and 43... the second number will be 10000000, since it starts with a zero.
SELECT * FROM users ORDER BY age::INTEGER 
-- It will give the same output as the first column, but if it was a text, it would give a wrong answer as the first number is 10000 and the second is 2... and so on

-- Suppose we want to convert the age into a NUMERIC (decimal) value. We can use: age::NUMERIC -- This converts the INTEGER into a NUMERIC data type. But why do we even need this mess? isn't it useless... well.... It is useful when you need decimal arithmetic or when working with values that should be treated as decimal numbers, so it is not that "useless*. Be careful: if you try to convert text that is not a valid number, PostgreSQL will throw an error.
```

Now we can start again with
- LIKE < --- Done
- BETWEEN
- IS NULL
- IN

We will start with BETWEEN and IN, since they are pretty easy:

```SQL

-- What is BETWEEN? The concept is really easy, think about BETWEEN as something as: (x <= column_name <= y) That the math way to imagine it. 
-- So instead of doing manual WHERE age >= 20 AND age <= 40; -- We can just use BETWEEN: 
WHERE age BETWEEN 20 AND 40; 
-- This will look for all the rows that satisfy this condition (Only ages that are in between 20 and 40, even 20 and 40 are included!)
SELECT * FROM users WHERE age BETWEEN 20 AND 40 ORDER BY age, first_name;
-- It will order by the smallest age to the biggest (In a range of 20 to 40) and if the ages are identical, it will look at the name and print the one that is closest to the A (Since it goex from A to Z)

"""
Output:

 user_id | first_name | last_name | age
---------+------------+-----------+-----
       5 | Mark       | Johnson   |  22
       7 | Luca       | Bianchi   |  26
       6 | Sophia     | Müller    |  26
       3 | Alex       | Rossi     |  34

"""

-- We can use even logic as NOT, as for:

SELECT * FROM users WHERE NOT age BETWEEN 20 AND 40 ORDER BY age, first_name;
-- This works as: Keep every row whose age is not between 20 and 40 (Both included).

"""
Output:

 user_id | first_name | last_name  | age
---------+------------+------------+-----
       1 | Shrimp     | ShupoShupo |  19
      10 | Mia        | Smith      |  54
       2 | Daniel     | Kovalenko  |  59
       8 | Anya       | Novak      |  67
       4 | Elena      | Petrenko   |  71
       9 | Kenji      | Tanaka     |  76

"""
```

After understanding BETWEEN, we can start with IN - A saver. Why so? 

Imagine that your boss came in your room at 3:24 AM and said to you: "Hey! 3:24 AM is not for sleeping! They are for working, so wake up and find the age of the users with the id 3, 15, 84, 921, 2831, 4632, 5000 and tell which is the oldest."

So what will you do? Refuse him? Not an option, ye wanna work. So you will stand up and write

```SQL
SELECT * FROM users WHERE user_id = 3 OR WHERE user_id = 15 OR .....
```

I am sure we don't wanna write this at 3:24 AM, so you would use a small trick... you will use IN...

```SQL
SELECT * FROM users WHERE user_id IN (3, 15, 84, 1241, 2831, 4632, 5000) ORDER BY age DESC;

"""
Output:

user_id | first_name |  last_name   | age 
---------+------------+--------------+-----+
    2831 | Stefan     | Xylander     |  87 
    4632 | Matteo     | Zephyrus     |  82 
      15 | John       | Quicksilver  |  65 
       3 | Liam       | Nightshade   |  34 
    1241 | Andrii     | Vintervarg   |  29 
      84 | Sophie     | Featherstone |  18 
    5000 | Giulia     | Obsidian     |  17

"""
-- We got the clear answer, we can do the same answer with less effort. 
-- We can get even other logic as:

SELECT * FROM users WHERE  user_id BETWEEN 500 AND 1000 AND age BETWEEN 15 AND 30 OR age IN (35, 45, 55, 65, 75, 85, 95) ORDER BY age; 
-- It will select rows where user_id is between 500 and 1000, and their age must either be between 15 and 30, or be one of these exact ages: 35, 45, 55, 65, 75, 85, 95. Then it orders everything by age.

"""
Output:

user_id | first_name |  last_name   | age 
---------+------------+--------------+-----
       566 | Dmitry     | Stormweaver  |  19 
       921 | Elena      | Shadowend    |  35 
       732 | Lev        | Thornevale   |  45 
       835 | Nataliya   | Frostspire   |  55 
       632 | Bohdan     | Ironwood     |  65

"""
```

Now we will use IS NULL - An easy yet really important topic on SQL.
NULL doesn't mean that we have age = 0 or we have an empty strings as '', NULL means - "We don't know that", as if age is NULL, that means that we don't know the age of the user, not that it doesn't have an age.

(We will use this table:

| run_id | model_name     | accuracy | training_time | dataset_size |
| ------:|----------------|---------:|--------------:|-------------:|
| 1      | Random Forest  | 91.2     |            18 | 10000        |
| 2      | XGBoost        | 94.8     |            27 | 10000        |
| 3      | Logistic Reg.  | 87.5     |             5 | 10000        |
| 4      | Neural Network | 96.1     |          NULL | 50000        |
| 5      | SVM            | 90.0     |            34 | 25000        |
| 6      | LightGBM       | 95.3     |            19 | 50000        |
)
But how do we check it? Now I'll show you got to check:

```SQL
SELECT * FROM models WHERE training_time IS NULL;
-- This will print us rows from the users table where we don't know the user's age (where the `age` value is NULL (Missing, buddy boy)).

"""
Output:

 run_id | model_name      | accuracy | training_time | dataset_size
--------+-----------------+----------+---------------+--------------
      4 | Neural Network  |    96.1  |     NULL      |    50000

"""


SELECT * FROM models WHERE training_time IS NOT NULL;
-- This will print everything where the age is not NULL, if the age row is null, it will print everything beside that row.

"""
Output:

 run_id | model_name      | accuracy | training_time | dataset_size
--------+-----------------+----------+---------------+--------------
      1 | Random Forest   |    91.2  |      18       |    10000
      2 | XGBoost         |    94.8  |      27       |    10000
      3 | Logistic Reg.   |    87.5  |       5       |    10000
      5 | SVM             |    90.0  |      34       |    25000
      6 | LightGBM        |    95.3  |      19       |    50000

"""
```


Now... We will learn a easy yet messy topic, because you will have to memorize more than... 5 commands per time!!
And this is... `COUNT, SUM, AVG, MAX, MIN, AS` 
OMG There is 6 of them... Be careful to [don't pass out](https://www.healthline.com/health/how-to-prevent-fainting)!!

Now we will start with it:
```SQL

-- Let's start with the absolute basics: COUNT and AS. Cuz they are easy, and one example will totally make you understand the whole point. (I hope...)

1. COUNT

-- What is COUNT? It simply counts how many items are in a column. If your boss asks: "How many models have we trained in total?" (Since you work for 14 years under his company), you don't count them by hand. You use COUNT:

SELECT COUNT(model_name) FROM models;

""" 
Output:
count:
6
""" 

-- It tells us exactly 6 models are in that column (You trained just 6 models in 14 years...). Quick warning about NULL values (Yeah, our lovely NoExistance... again): if you do COUNT(training_time), it will output 5, because COUNT completely ignores NULL values! If you want to count every single row regardless of NULL, you write COUNT(*)

2. AS
   
-- In the future, by susing sum, avg, over... and so on, you will get hit with an ugly name (Especially over()!). But you want to be professional, you want to call your column "Hatsune Miku" rather than SUM or smth as this, so a professional way is using As:

SELECT COUNT(model_name) AS Hatsune_Miku FROM models;

"""
Output:

hatsune_miku:
6
""" 
-- Really cool, really professional, so no need to be afraid of any judgement.
-- No need to think that this is all, because I'll use them in the future too.
```

Now we start with `SUM, AVG, MAX, MIN`:

```SQL
--| run_id | model_name     | accuracy | training_time | dataset_size |
--| -----: | -------------- | -------: | ------------: | -----------: |
--|      1 | Random Forest  |     91.2 |            18 |        10000 |
--|      2 | XGBoost        |     94.8 |            27 |        10000 |
--|      3 | Logistic Reg.  |     87.5 |             5 |        10000 |
--|      4 | Neural Network |     96.1 |          NULL |        50000 |
--|      5 | SVM            |     90.0 |            34 |        25000 |
--|      6 | LightGBM       |     95.3 |            19 |        50000 |


1. SUM
-- SUM adds all the values in a column together.
SELECT SUM(dataset_size) AS grand_total_data FROM models;
--- it will do: 10000 + 10000 + 10000 + 50000 + 25000 + 50000

"""
Output:

 grand_total_data 
------------------
           155000
(1 row)
"""

2. AVG
-- AVG calculates the mathematical mean (average) of a column.
SELECT AVG(accuracy) AS average_accuracy FROM models;
-- Now we use a tad of math:
-- It will do: (91.2 + 94.8 + 87.5 + 96.1 + 90.0 + 95.3)/n (n = total numbers in the parentheses, since we have 6 numbers in the parentheses - n = 6)


"""
Output:

  average_accuracy  
--------------------
 92.4833333333333333
(1 row)
"""
-- Be carefull of the dreaded NULL, because if you have a null it will divide by the n - 1, since it doesn't take null, so out of your 6 numbers id one was null, it meant that it will not sum the 6 numbers and divide by 6, but sum 5 numbers and divide by 5 (Breaks the avg idea). So let us try to count the average of training_time. So we will do:
SELECT AVG(COALESCE(training_time, 0)) AS real_average FROM models;
""" 
Output: 

   real_average
--------------------
 17.1666666666666667 
(1 row)
"""
-- Now it will go by the avg, without breaking the idea.

3. MAX
-- MAX finds the highest value in a column.
SELECT MAX(training_time) AS slowest_time FROM models;
-- Now this is the maximum from statistics, but in easy words - It will choose the biggest number in that collumn.

"""
Output:
 slowest_time 
--------------
           34
(1 row)
"""

4. MIN
-- MIN finds the lowest value in a column.
SELECT MIN(training_time) AS fastest_time FROM models;
-- Same idea as maximum, just the smallest number from that column.

"""
Output:
 fastest_time 
--------------
            5
(1 row)
"""
```

Before going with the tasty... we start by GROUP BY...
Which is just a way to squish all of the columns together,  but it is fine for now, since we are teaching way too many, already... And GROUP BY will be an extreme case command, since I never really use it.

``` SQL
-- GROUP BY...  is a command I don't really like, but that not my choice, because hiring doesn't care about feelings. So I will explain them like Fenyman (I could never, he is just too good). 

-- First of all, let's make a table, because it will help us to understand the idea much clearer:

-- | id | employee | department | salary | 
 -- | :--- | :--- | :--- | :--- | 
 -- | 1 | Alice | Sales | 5000 | 
 -- | 2 | Bob | HR | 4500 | 
 -- | 3 | Charlie | Sales | 6000 | 
 -- | 4 | David | HR | 4800 | 
 -- | 5 | Eva | Sales | 5500 | 
 -- | 6 | Frank | HR | 5200 | 
 -- | 7 | Grace | Sales | 7000 | 
 -- | 8 | Henry | HR | 4600 | 
 -- | 9 | Ivy | Sales | 6200 | 
 -- | 10 | Jack | HR | 5000 |

-- Now let us imagine: 

-- The boss came all angry in your shower and start complaining... "Why nobody understands GROUP BY??" He said while putting some hair shampoo on his hair "Shrimp! Give me a clean list of our 10 departments and how much they spent this month"  (the table has 1 M rows with the transictions of all 10 departments)

-- By using other stuff as OVER() (You will learn it much later) you will get printed all 1M rows, not cute. So you will squash them... by using GROUP BY!

-- *The boss continued to wash his hair, but now he start mixing some shampoos with each other.*

SELECT department, SUM(salary) AS Total_Income_Of_Department FROM company GROUP BY department;

"""
Output:

+------------+-----------------------------+
| department | Total_Income_Of_Department  |
+------------+-----------------------------+
| Sales      |                       29700 |
| HR         |                       24100 |
+------------+-----------------------------+

This will show all '10' departments. (imagine that we have 10 departments...) 

"""
--What will happen if we use GROUP BY something different? 

SELECT employee, department, AVG(salary) FROM company GROUP BY employee;

-- Nothing will happen, because the names don't repeat so you see the difference:

"""
Output:

+----------+------------+-------------+
| employee | department | AVG(salary) |
+----------+------------+-------------+
| Alice    | Sales      |        5000 |
| Bob      | HR         |        4500 |
| Charlie  | Sales      |        6000 |
| David    | HR         |        4800 |
| Eva      | Sales      |        5500 |
| Frank    | HR         |        5200 |
| Grace    | Sales      |        7000 |
| Henry    | HR         |        4600 |
| Ivy      | Sales      |        6200 |
| Jack     | HR         |        5000 |
+----------+------------+-------------+

"""

-- I will give another example, with another table:

-- | order_id | customer_id | status     | order_amount |
-- | :------- | :---------- | :--------- | :----------- |
-- | 101      | Cust_A      | Shipped    | 150          |
-- | 102      | Cust_B      | Shipped    | 40           |
-- | 103      | Cust_A      | Cancelled  | 200          |
-- | 104      | Cust_C      | Shipped    | 300          |
-- | 105      | Cust_B      | Shipped    | 60           |
-- | 106      | Cust_A      | Shipped    | 80           |

-- Let us say that we want to find just the one with shipped status and how many shipped they have (per customer).

SELECT customer_id, COUNT(order_id) AS total_orders, SUM(order_amount) AS total_spent FROM orders WHERE status = 'Shipped' GROUP BY customer_id;

"""
Output:

+-------------+--------------+-------------+
| customer_id | total_orders | total_spent |
+-------------+--------------+-------------+
| Cust_A      |            2 |         230 |
| Cust_B      |            2 |         100 |
| Cust_C      |            1 |         300 |
+-------------+--------------+-------------+

"""
-- That means that Cust_A had 2 shipped (One got cancelled, because it was not 'Shipped'), Cust_B had 2, and Cust_C had 1 shipped.
```

Now we will learn other topics, because there are many waysssss of breaking yourself mentally and morally.  So let's learn some new ideas as DISTINCT and JOIN < --- Scary part.
We will start with DISTINCT, because it is less painful :

``` SQL

-- experiment_id | model_name      | dataset
---------------+-------------------+---------
--             1 | Random Forest   | Titanic
--             2 | XGBoost         | Titanic
--             3 | XGBoost         | MNIST
--             4 | Neural Network  | MNIST
--             5 | Random Forest   | CIFAR-10
--             6 | XGBoost         | Titanic

-- By looking at this we can notice that there is a copy as: the one with id 2 and id 6.

SELECT DISTINCT model_name, dataset FROM models;

""" 
Output:

 model_name      | dataset
-----------------+----------
 Random Forest   | Titanic
 XGBoost         | Titanic
 XGBoost         | MNIST
 Neural Network  | MNIST
 Random Forest   | CIFAR-10
 
"""
-- We select just model_name and dataset, because it will look if there is a model that has the same model_name and dataset, if yes -> take the duplicates away in the print, if no -> let both. Because if we would do: 

SELECT DISTINCT * FROM models;

"""
Output:

 experiment_id | model_name      | dataset
 --------------+-----------------+---------
             1 | Random Forest   | Titanic
             2 | XGBoost         | Titanic
             3 | XGBoost         | MNIST
             4 | Neural Network  | MNIST
             5 | Random Forest   | CIFAR-10
             6 | XGBoost         | Titanic
             
"""

-- It will print everything because SQL asks itself "Are there any rows where every selected column is identical?" He will notice the second and 6th, but then you placed *, so he looks even at experiment_id, which is different, so it will not take away the duplicate.

```

Since we understood DISTINCT, we will proceed with something else... with JOIN... 
Well, for this I'll use two different tables, because the command name already explain its function.

```SQL 
1)
-- This table will be called 'models':

-- model_id |   model_name   |  framework   | created_at 
------------+----------------+--------------+------------
--        1 | Random Forest  | scikit-learn | 2026-01-15
--        2 | XGBoost        | xgboost      | 2026-02-01
--        3 | Neural Network | PyTorch      | 2026-03-10
--        4 | LightGBM       | lightgbm     | 2026-03-15

2)
-- This table will be called 'runs':

 --run_id | model_id | accuracy | training_time_secs |  status   
----------+----------+----------+--------------------+-----------
--      1 |        2 |     94.8 |                120 | Completed
--      2 |        1 |     91.2 |                 45 | Completed
--      3 |        3 |     96.1 |               1800 | Completed
--      4 |        2 |     95.3 |                145 | Completed
--      5 |        4 |          |                 12 | Failed
--      6 |        3 |     89.4 |                600 | Completed

3) Trying the codes
   
SELECT models.model_id, models.framework, runs.accuracy, runs.training_time_secs, runs.status FROM runs JOIN models ON runs.model_id = models.model_id;

-- OR just give a small alias to runs and models (Result is identical):

SELECT m.model_id, m.framework, r.accuracy, r.training_time_secs, r.status FROM runs r JOIN models m ON r.model_id = m.model_id;

-- Let me break it to piece:

-- SELECT models.model_id, models.framework ... runs.status = This will take the parts from what table we want:                                                    Let's take as example models.model_id - this will take the column model_id from the table models                                                                 Let's take as example runs.accuracy - this will take the accuracy column from the table runs

-- FROM runs JOIN models = it doesn't really matter when you use just JOIN, so you could write FROM models JOIN runs, the outcome will be the same.

-- ON runs.model_id = models.model_id = We are matching the ids, so it doesn't give wrong information as, accuracy = 94.8 is from Random Forest, when it is from XGBoost. We want it to match them automatically so it matches them perfectly

"""
Output:

 model_id |  framework   | accuracy | training_time_secs |  status   
----------+--------------+----------+--------------------+-----------
        2 | xgboost      |     94.8 |                120 | Completed
        1 | scikit-learn |     91.2 |                 45 | Completed
        3 | PyTorch      |     96.1 |               1800 | Completed
        2 | xgboost      |     95.3 |                145 | Completed
        4 | lightgbm     |          |                 12 | Failed
        3 | PyTorch      |     89.4 |                600 | Completed
        
It gaved to us all the info we required while matching (thanks to ON runs.model_id = models.model_id) automatically the right model to the accuracy.

"""

```

That it for the JOIN (A.K.A as Inner JOIN), we will not learn LEFT JOIN or RIGHT JOIN, because even if useful, you are not a full job Data Engineer.

"If I will ever stumble upon it, I will come back and write about them" - Said on the date of 3 July 2026 at 14:49 PM.
(If you will see underneath the LEFT JOIN or RIGHT JOIN, that means that I stumbled on it. If no, that means you will probably not use them too)

Now we will do our last topic...

1) Window functions - hard
2) CASE WHEN - easy

Let's start with... Window functions, because it is the easiest in my eyes, but google says that it is the hardest from all of them, but as usual, we will not go as deep as a spelunker (We prefer ML topics and math over data... yet PySpark will be really interesting --- TIME SPOILER, It will be easy if you know SQL.)

So let's finish with this madness already! 

Window functions... would be easy to say stare at a window and think about functions, but that fine.

Window functions will be broke on this parts:
- RANK() with OVER(ORDER BY) at the same time
- LAG()
- LEAD()
- PARTITION BY

```SQL 
-- We will use the table below for this example (The table name is 'models'):

--| run_id | model_name     | accuracy | training_time | dataset_size |
--| -----: | -------------- | -------: | ------------: | -----------: |
--|      1 | Random Forest  |     91.2 |            18 |        10000 |
--|      2 | Neural Network |     94.8 |            27 |        10000 |
--|      3 | Random Forest  |     87.5 |             5 |        10000 |
--|      4 | Neural Network |     96.1 |          NULL |        50000 |
--|      5 | SVM            |     90.0 |            34 |        25000 |
--|      6 | Neural Network |     95.3 |            19 |        50000 |

-- I placed more copies, because it is easier to explain

-- We will start with RANK() and OVER(), but why not just OVER? Because without OVER it will be really hard to use 100% of the commands.

1) RANK and OVER

SELECT run_id, model_name, accuracy, RANK() OVER(ORDER BY accuracy DESC) AS accuracy_ranking FROM models;
-- RANK() - will rank everyone by something you will choose in OVER... think about writing OVER(ORDER BY accuracy DESC) it will rank based on accuracy, so the biggest accuracy score gets 1st place, the second biggest accuracy score gets 2nd place and so on... (That why I used accuracy DESC, because I just wrote ORDER BY accuracy, it would've given me 1 place to the smallest value, since it goes from smallest to biggest, but using DESC would give me from biggest to smallest)
-- Always use OVER(), otherwise SQL will chock and die. What is OVER?? 
-- OVER() permits to us to keep every single row, withouth squishing everything as GROUP BY does (THAT WHY I OBMITED IT, OKAY????) - Normally, when you ask SQL to calculate something like an average, a sum, or a total count, it forces you to smash your data together. It takes all your rows and crushes them down into a single line. Sooo, I saved you from wondering "Where did my 1 row went? where is 6? Gone. Crushed into the machine."
-- Why did I use ORDER BY accuracy in over? Now imagine...
--Your boss hands you a stack of medals and says: "Rank these students from 1st place to 6th place."
-- If your boss just says that and walks away (which is like writing RANK() OVER()), your first question will be: "Rank them based on WHAT?" Is it by height?By running speed? By their test scores? So if you write just RANK() OVER(), it will print to everybody 1 place. While if you do: RANK () OVER(ORDER BY accuracy) it actually tells to RANK to rank by accuracy, so you will get a normal output as:

"""
Output:

 run_id |   model_name   | accuracy | accuracy_ranking 
--------+----------------+----------+------------------
      3 | Neural Network |     96.1 |                1
      6 | Neural Network |     95.3 |                2
      2 | Neural Network |     94.8 |                3
      1 | Random Forest  |     91.2 |                4
      5 | SVM            |     90.0 |                5
      3 | Random Forest  |     87.5 |                6
(6 rows)
"""

-- I'll give another example, so we don't actually fake it by saying "Yeahhh, I understood" - while we clearly didn't.

SELECT model_name, training_time, dataset_size, RANK() OVER(ORDER BY dataset_size) AS dataset_s_rank, MAX(dataset_size) OVER() AS max_d_s, MIN(dataset_size) OVER() AS min_d_s FROM models WHERE training_time IS NOT NULL;

"""
Output:

model_name     | training_time | dataset_size |   d_s_rank   | max_d_s | min_d_s 
-------------  +---------------+--------------+--------------+---------+--------
Random Forest  |             5 |        10000 |            1 |   50000 |  10000
Random Forest  |            18 |        10000 |            1 |   50000 |  10000
Neural Network |            27 |        10000 |            1 |   50000 |  10000
SVM            |            34 |        25000 |            4 |   50000 |  10000
Neural Network |            19 |        50000 |            5 |   50000 |  10000
(5 rows)

"""
-- When we use OVER() next to any command as AVG, SUM, MAX, MIN, it will print the same value in all the rows, rather than squishing everything.
```

Now we will try to use LAG() and LEAD(), because they are pretty useful - depends by the situation.
In poor words:
- LAG() - prints the previous value of the column you chose
- LEAD() - prints the following value that has to come, of the column you chose

```SQL

-- run_id | model_name     | accuracy | training_time | dataset_size
---------+----------------+----------+---------------+--------------
--     1 | Random Forest  |     91.2 |            18 |        10000
--     2 | Neural Network |     94.8 |            27 |        10000
--     3 | Random Forest  |     87.5 |             5 |        10000
--     4 | Neural Network |     96.1 |          NULL |        50000
--     5 | SVM            |     90.0 |            34 |        25000
--     6 | Neural Network |     95.3 |            19 |        50000
--     7 | XGBoost        |     92.4 |            85 |        30000
--     8 | XGBoost        |     94.1 |            90 |        30000
--     9 | Random Forest  |     89.9 |            12 |        10000
--    10 | Neural Network |     97.2 |          2100 |        50000

-- We incresed a tad the lenght, so we have a better output.

-- Firstly I'll tell a universal law, just like the law of gravity. LAG() will always have an empty place on the first row, since it takes from the previous, but since we have a table of 10 rows... he can't take from row a non existing previous row. Same for LEAD(), but LEAD() will have in the end the empty line, since it takes the following value, because it can't take the 11th value from a table with 10 rows. 

1) LAG
-- We start with LAG < ---- I like the name, it reminds me of Lagtrain by Inabakumori

SELECT run_id, model_name, accuracy, LAG(accuracy) OVER(ORDER BY run_id) AS previous_accuracy FROM models;

-- It printed all the previous answers... but why is it useful? For now you will just check the differences manually, but the next trick is beautiful!
-- Anyway, I used ORDER BY run_id, because otherwise he wouldn't know from where to start, so we force him to start with the accurancy from id 1 till 10.

"""
Output:

run_id |   model_name   | accuracy | previous_accuracy 
--------+----------------+----------+-------------------
      1 | Random Forest  |     91.2 |              NULL
      2 | Neural Network |     94.8 |              91.2  <-- You can check diff-
      3 | Random Forest  |     87.5 |              94.8        erence
      4 | Neural Network |     96.1 |              87.5
      5 | SVM            |     90.0 |              96.1
      6 | Neural Network |     95.3 |              90.0
      7 | XGBoost        |     92.4 |              95.3
      8 | XGBoost        |     94.1 |              92.4
      9 | Random Forest  |     89.9 |              94.1
     10 | Neural Network |     97.2 |              89.9
(10 rows)

"""

-- Now we will use LAG() in a useful way... we will do some basic arithmetic. 

SELECT run_id, accuracy, accuracy - LAG(accuracy) OVER(ORDER BY run_id) AS accuracy_growth FROM models;
-- We just did a small mathematical process, as `accuracy - previous accuracy`,  and that reall useful because now you can see the difference in numbers (MAIN RULE:)
-- If `accuracy - previous accuracy = +pos` if the result is positive, it means that the accuracy rose by +pos% (Only if you go by 0 to 100)
-- If `accuracy - previous accuracy = -neg` if the result is negative, it means that the accuracy fell by -neg% (Only if you go by 0 to 100)

-- Imagine that all this list is composed by the same model, and you look how the accuracy grow by time.
"""
Output:

run_id | accuracy | accuracy_growth 
--------+----------+-----------------
      1 |     91.2 |            NULL  <-- number - NULL = NULL (Always)
      2 |     94.8 |             3.6  <-- rose by 3.6%
      3 |     87.5 |            -7.3  <-- fell by 7.3%
      4 |     96.1 |             8.6
      5 |     90.0 |            -6.1
      6 |     95.3 |             5.3
      7 |     92.4 |            -2.9
      8 |     94.1 |             1.7
      9 |     89.9 |            -4.2
     10 |     97.2 |             7.3
(10 rows)

"""

2) LEAD

-- Now we start with LEAD()... The twin of LAG(), you can use it the same way I used LAG(), but we try other ways, first example will be just a visual, so you understand how it work:

SELECT run_id, model_name, accuracy, LEAD(accuracy) OVER(ORDER BY run_id) AS next_accuracy FROM models;

"""
Output:

run_id |   model_name   | accuracy | next_accuracy
-------+----------------+----------+---------------
     1 | Random Forest  |     91.2 |          94.8  <-- ogle at Row 2
     2 | Neural Network |     94.8 |          87.5
     3 | Random Forest  |     87.5 |          96.1
     4 | Neural Network |     96.1 |          90.0
     5 | SVM            |     90.0 |          95.3
     6 | Neural Network |     95.3 |          92.4
     7 | XGBoost        |     92.4 |          94.1
     8 | XGBoost        |     94.1 |          89.9
     9 | Random Forest  |     89.9 |          97.2
    10 | Neural Network |     97.2 |          NULL  <-- Universal Law! No 11th                                                           row.
(10 rows)

"""

-- Why would we want to look at the _next_ row? Imagine your boss hands you this list and says: "I want to know how much accuracy we are going to lose or gain in the very next run so I can plan ahead when to place al my company funds on red."

-- We can do the same basic arithmetic, but flipped: `LEAD(accuracy) - accuracy`.

SELECT run_id, accuracy, LEAD(accuracy) OVER(ORDER BY run_id) - accuracy AS future_variance FROM models;

"""
Output:

run_id | accuracy | future_variance
-------+----------+-----------------
     1 |     91.2 |             3.6  <-- The next run will be 3.6% higher!
     2 |     94.8 |            -7.3  <-- Watch out, the next run drops by 7.3%!
     3 |     87.5 |             8.6  <-- All on black immediately!
     4 |     96.1 |            -6.1
     5 |     90.0 |             5.3
     6 |     95.3 |            -2.9
     7 |     92.4 |             1.7
     8 |     94.1 |            -4.2
     9 |     89.9 |             7.3
    10 |     97.2 |            NULL  <-- NULL - number = NULL (Always) <--                                                Universal law
(10 rows)

"""
```

Now we will learn PARTITION BY, which is really useful! And one of my fav, soooo, let's dive in.
What is PARTITION BY?

This table will help us:

| run_id | model_name     | accuracy | training_time | dataset_size |
| :----- | :------------- | :------- | :------------ | :----------- |
| 1      | Random Forest  | 91.2     | 18            | 10000        |
| 2      | Neural Network | 94.8     | 27            | 10000        |
| 3      | Random Forest  | 87.5     | 5             | 10000        |
| 4      | Neural Network | 96.1     | 26            | 50000        |
| 5      | SVM            | 90.0     | 34            | 25000        |
| 6      | Neural Network | 95.3     | 19            | 50000        |
| 7      | XGBoost        | 92.4     | 85            | 30000        |
| 8      | XGBoost        | 94.1     | 90            | 30000        |
| 9      | Random Forest  | 89.9     | 12            | 10000        |
| 10     | Neural Network | 97.2     | 2100          | 50000        |


```SQL

-- Partition by is not even that hard, think about it, you have to use RANK, and rank all the models, but this time... you have to rank within their own group only... as for: 
-- It’s like having a giant basket filled with 20 apples, 20 oranges, and 20 bananas. They all have a `weight` column. If you use a normal rank, you are ranking all 60 fruits together. But if you use `PARTITION BY fruit_type`, SQL resets the ranking counter for each group. You get a rank from 1 to 20 for the apples, a fresh 1 to 20 for the oranges, and a fresh 1 to 20 for the bananas.
-- Let us rank each model accuracy and see how they did.

SELECT model_name, accuracy, training_time, RANK() OVER(PARTITION BY model_name ORDER BY accuracy DESC) AS model_rank FROM models; 

"""
Output:

+----------------+----------+---------------+------------+
| model_name     | accuracy | training_time | model_rank |
+----------------+----------+---------------+------------+
| Neural Network |     97.2 |          2100 |          1 |
| Neural Network |     96.1 |            26 |          2 |
| Neural Network |     95.3 |            19 |          3 |
| Neural Network |     94.8 |            27 |          4 |
| Random Forest  |     91.2 |            18 |          1 |
| Random Forest  |     89.9 |            12 |          2 |
| Random Forest  |     87.5 |             5 |          3 |
| SVM            |     90.0 |            34 |          1 |
| XGBoost        |     94.1 |            90 |          1 |
| XGBoost        |     92.4 |            85 |          2 |
+----------------+----------+---------------+------------+
(10 rows)

"""
-- We did that so the highest accuracy Neural Network gets a 1, and the highest accuracy Random Forest also gets a 1. They don't interfere with each other's scoreboards at all.

-- We try the same with LAG and AVG.

SELECT model_name, training_time, accuracy,   AVG(accuracy) OVER(PARTITION BY model_name) AS avg_accuracy, accuracy - LAG(accuracy) OVER(PARTITION BY model_name ORDER BY accuracy DESC) AS accuracy_drop FROM models;
-- We add to LAG() ORDER BY, because we want for it t make sure to start from the biggest value and go down.

"""
Output:

+----------------+---------------+----------+--------------+---------------+
| model_name     | training_time | accuracy | avg_accuracy | accuracy_drop |
+----------------+---------------+----------+--------------+---------------+
| Neural Network |          2100 |     97.2 |       95.850 |          NULL |
| Neural Network |            26 |     96.1 |       95.850 |          -1.1 |
| Neural Network |            19 |     95.3 |       95.850 |          -0.8 |
| Neural Network |            27 |     94.8 |       95.850 |          -0.5 |
| Random Forest  |            18 |     91.2 |       89.533 |          NULL |
| Random Forest  |            12 |     89.9 |       89.533 |          -1.3 |
| Random Forest  |             5 |     87.5 |       89.533 |          -2.4 |
| SVM            |            34 |     90.0 |       90.000 |          NULL |
| XGBoost        |            90 |     94.1 |       93.250 |          NULL |
| XGBoost        |            85 |     92.4 |       93.250 |          -1.7 |
+----------------+---------------+----------+--------------+---------------+
(10 rows)

"""
-- The accuracy drop shows us by how much our model accuracy dropped, as for: 
-- Firstly it dropped by 1.1%, then by 0.8% and in the end by 0.5%. So we can see by how weaker our previous model was (There are NULLs, because LAG always add NULL at the start), the AVG showed the average_accuracy for each group of models.

```

That how all this mess work, now we will end with CASE WHEN.

it is simply the:
- If - CASE WHEN
- elif - WHEN
- else - ELSE

simple as that, now I'll give some examples:

```SQL

-- We will use the same table as before: 

-- run_id | model_name      | accuracy | training_time | dataset_size
--------+-----------------+----------+---------------+--------------
--      1 | Random Forest   |     91.2 |            18 |        10000
--      2 | Neural Network  |     94.8 |            27 |        10000
--      3 | Random Forest   |     87.5 |             5 |        10000
--      4 | Neural Network  |     96.1 |            26 |        50000
--      5 | SVM             |     90.0 |            34 |        25000
--      6 | Neural Network  |     95.3 |            19 |        50000
--      7 | XGBoost         |     92.4 |            85 |        30000
--      8 | XGBoost         |     94.1 |            90 |        30000
--      9 | Random Forest   |     89.9 |            12 |        10000
--     10 | Neural Network  |     97.2 |          2100 |        50000

-- We will teach you fastly~

SELECT model_name, accuracy, CASE WHEN accuracy >= 95 THEN 'Excellent' WHEN accuracy >= 90 THEN 'Good' ELSE 'Needs Improvement' END AS rating FROM model_runs; -- END AS will just give the name to the new collumn

"""
Output:

 model_name      | accuracy | rating
-----------------+----------+-------------------
 Random Forest   |     91.2 | Good
 Neural Network  |     94.8 | Good
 Random Forest   |     87.5 | Needs Improvement
 Neural Network  |     96.1 | Excellent
 SVM             |     90.0 | Good
 Neural Network  |     95.3 | Excellent
 XGBoost         |     92.4 | Good
 XGBoost         |     94.1 | Good
 Random Forest   |     89.9 | Needs Improvement
 Neural Network  |     97.2 | Excellent
 
 """
 
 -- We will give another example:
 
SELECT model_name, training_time, CASE WHEN training_time < 20 THEN 'Fast' WHEN training_time <= 60 THEN 'Medium' ELSE 'Slow' END AS speed FROM model_runs;
 
"""
Output:

 model_name      | training_time | speed
-----------------+---------------+--------
 Random Forest   |            18 | Fast
 Neural Network  |            27 | Medium
 Random Forest   |             5 | Fast
 Neural Network  |            26 | Medium
 SVM             |            34 | Medium
 Neural Network  |            19 | Fast
 XGBoost         |            85 | Slow
 XGBoost         |            90 | Slow
 Random Forest   |            12 | Fast
 Neural Network  |          2100 | Slow
 
""" 
```

We survived all this madness, now we will advance forward... with more madness incoming, cuz this were baby steps... sadly.
But chin up, the more we advance, the more we suffer, so be ready to suffer.
