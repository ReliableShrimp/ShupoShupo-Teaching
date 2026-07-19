# SQL — First Data Engineering Concepts

This is our first real Data Engineering concept! And before you sit there thinking, "I don't need Data Engineering because I'm going to be an ML Engineer / AI Engineer / Pipeline Plumber / Neuro-Electrician Pilot," let me stop you right there. You **WILL** need `SQL`. Every single job posting nowadays demands it.

Picture this:

*(Character.ai style)*

HIM: **The interviewer leans in, looking slightly confused, and asks:**
Interviewer: "Do you know `GNNs` to absolute perfection?"

YOU: **You smirk, completely unfazed.**
You: "Of course. It's a fundamental pillar."

HIM: **He gets a tad surprised and asks you another question.**
Interviewer: "What about `SQL`?"

YOU: **You laugh, since you ate it yesterday.**
You: "Nasty, don't you think?"

*(Because in your mind: GNN = "**G**uys, **N**o **N**otetaking" or maybe "**G**rownups **N**eed **N**aps." And SQL = "**S**uper **Q**uadrillion **L**ines")*

HIM: **He leaves the call, blocks you, reports you, sends the police to your house, gets a prosecutor — and now you've got a criminal record for Super Quadrillion Lines.**

Jokes aside, every company needs you to know more than just your narrow slice of the codebase. You can't be an ophthalmologist (an eye specialist) with absolutely no clue how the stomach works. In production AI, Data Engineering is the stomach that feeds your model's brain.

So we'll start with the architectural basics of data extraction and keep pushing forward. The concepts aren't complex — take a breath, relax, and let's get into it.

## 1. Introduction

Why do we even need `SQL`? What even is `SQL`?

`SQL` (a.k.a. **S**tructured **Q**uery **L**anguage) is a language made for talking to a database.

*(What's a database? Think of it as a hyper-secure wardrobe — you can shove data inside it, but the data won't be thrown in randomly. It gets stored in tables (think of them as files), which look something like this:)*

| id | name | formula | mass |
| :-- | :--------------- | :-------- | :----------- |
| 1 | Water | H2O | 18 |
| 2 | Carbon Dioxide | CO2 | 44 |
| 3 | Glucose | C6H12O6 | 180 |
| 4 | BrokenWater | NULL | 9999 |
| 5 | FakeMethane | CH888 | not_a_number |
| 6 | Oxygen | O2 | 32 |
| 7 | WeirdCompound | C6H@@12O6 | NULL |
| 8 | HydrogenPeroxide | H2O2 | 34 |

*(8 rows)*

If a database just stores data in rows and columns, why can't we just use a folder full of plain CSV files, Excel spreadsheets, or JSON text files?

Because everything in this world has pros and cons. A fish can swim fast, but can it climb trees? No — and that doesn't make it a bad fish. Like fish, files are made for smaller waters, while databases are made for oceans. We even have concrete reasons to back that up:

**The Search Speed Catastrophe (Indexing)**
1. Say you have a CSV file with 100M lines, and you need to find the exact data for a user with ID `84_292_182` — who, as expected, sits on row 84,292,182. What's the problem? Let's find out. You open the file in Python (two scenarios: either it chokes and throws an Out Of Memory error, or it doesn't). Then it starts from line one, goes to line two, to line three (a.k.a. a **Full Table Scan**), and somewhere around the year 2083, it finishes. *(Joke — but remember, this really is painfully slow.)*
2. A database instead uses **indexing** (usually structured as B-Trees). Think of it as a highly advanced book index. Instead of reading every row, the database engine skips straight to the exact physical location of that user ID on disk. A lookup that takes minutes in a raw file takes less than 3 milliseconds in a database.

**The Fraud & Consistency Trap**
1. Imagine Account A sends Account B $100. Your script subtracts $100 from Account A... but BOOM, the server loses power. The result? Account A is down $100, and Account B never actually received it. Cute, but sad. :(
2. Meanwhile, our glorious king of databases guarantees a property called **Atomicity**. This treats a series of operations as "all-or-nothing" — if anything fails partway through, the database immediately performs a **Rollback**, undoing everything as if it never happened.

**Memory Choking**
1. To open a file with a library like `Pandas`, or with plain `with open(...)`, the machine loads it into its own RAM — so once you're dealing with files that cost many GBs, your machine can choke on all that data and throw an OOM error. (Unless you use PySpark or similar libraries that split the data across other machines by connecting to a master URL.)
2. A database is built exactly for this: it stores data on disk and optimizes its internal cache pools, letting you slice, add, remove, and otherwise work with many terabytes of information without your laptop bursting into flames.

So that's why databases exist. Now let's learn to control them — with the help of `SQL`, and later `PySpark` (joking — you'll be using `PySpark` again soon enough).

## 2. Commands of Basic SQL

### SELECT and WHERE

Now we start with the easy part:

```sql
SELECT column_name FROM table_name;   -- Always remember to place a ; at the end of it
```

What does this mean? Let's bring back the table from above:

| id | name | formula | mass |
| :-- | :--------------- | :-------- | :----------- |
| 1 | Water | H2O | 18 |
| 2 | Carbon Dioxide | CO2 | 44 |
| 3 | Glucose | C6H12O6 | 180 |
| 4 | BrokenWater | NULL | 9999 |
| 5 | FakeMethane | CH888 | not_a_number |
| 6 | Oxygen | O2 | 32 |
| 7 | WeirdCompound | C6H@@12O6 | NULL |
| 8 | HydrogenPeroxide | H2O2 | 34 |

First, we need to know this is a table with its own name — in our case, `molecules` (you usually choose the name when creating a table).

What does `SELECT` mean? It means "select one or more columns by name," so it'll print just those columns:

```sql
SELECT id, name FROM molecules;

/*
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
*/
```

See? It printed exactly what we asked for. What if we want more — say, the whole table? Then we do:

```sql
SELECT * FROM molecules;

/*
Output:

 id |       name       |  formula  |     mass
----+------------------+-----------+--------------
  1 | Water            | H2O       | 18
  2 | Carbon Dioxide   | CO2       | 44
  3 | Glucose          | C6H12O6   | 180
  4 | BrokenWater      | NULL      | 9999
  5 | FakeMethane      | CH888     | not_a_number
  6 | Oxygen           | O2        | 32
  7 | WeirdCompound    | C6H@@12O6 | NULL
  8 | HydrogenPeroxide | H2O2      | 34
*/
```

The `*` means **all columns**. But what if we only want the `formula` and `mass` columns for the first four molecules? Then we do:

```sql
-- Before we start: if WHERE doesn't return anything, check the data type of the column.
-- If the column stores text (str) instead of numbers (int), you may need quotes, e.g. '4'.

SELECT formula, mass FROM molecules WHERE id <= 4;

/*
Output:

formula | mass
--------+------
H2O     | 18
CO2     | 44
C6H12O6 | 180
NULL    | 9999
*/
```

`WHERE` roughly means "where this column is (or has) such-and-such" — it can be used in a bunch of ways:

```sql
WHERE mass >= 15;
-- Selects rows where this condition holds (mass is bigger than or equal to 15)
WHERE mass <= 15;                     -- Selects rows where mass is smaller than or equal to 15
WHERE mass = 15;                      -- Selects rows where mass is exactly 15
WHERE mass = 15 OR formula = 'CO2';   -- Selects rows where mass is 15, OR the formula is CO2
WHERE formula = 'H2O' AND mass = 18;  -- Selects rows where the formula is H2O AND the mass is 18
WHERE NOT formula = 'H2O';            -- Selects rows where the formula is NOT H2O.
-- You can also write this as WHERE formula != 'H2O' or WHERE formula <> 'H2O' — same result,
-- since both just mean "formula is not equal to..."
WHERE mass <= 180 AND NOT formula = 'CO2';  -- mass is 180 or lower, AND the formula isn't CO2
```

Those are some basic `WHERE` conditions.

### Creating, filling, and clearing tables

Now let's learn how to create tables, since we'll need this for prototyping and other experiments. We could cook up a table with all the VOCALOIDSSSS, but sadly I won't do that, since we're professionals here, talking about learning. That's why we'll learn to create and drop tables using Vocaloids as our example:

```sql
-- The general syntax looks like this:

CREATE TABLE table_name (column_name data_type, column_name data_type, column_name data_type);

-- Now here's the actual example:

CREATE TABLE Vocaloids (name TEXT, height NUMERIC, weight NUMERIC, age INTEGER);
```

But what is `TEXT`, `NUMERIC`, or `INTEGER`? (I'll explain more later, but here's the gist:)

| Column | Data Type | Think of it as | Example |
| -------- | --------- | ------------------------ | ---------------- |
| `name` | `TEXT` | A **string** | `'Hatsune Miku'` |
| `height` | `NUMERIC` | A **decimal number** | `158.5` |
| `weight` | `NUMERIC` | A **decimal number** | `42.3` |
| `age` | `INTEGER` | A **whole number (int)** | `16` |

Now let's fill that empty table with values:

```sql
-- Remember, the column order we defined is (name, height, weight, age).

INSERT INTO Vocaloids  -- <--- the table name, INSERT INTO table_name
VALUES ('Hatsune Miku', 158, 42.3, 18), ('Kagamine Rin', 152, 43.0, 14), ('Kasane Teto', 159.5, 47.0, 31);

-- Now we have our table:

--      name      | height | weight | age
------------------+--------+--------+-----
-- Hatsune Miku   |  158.0 |   42.3 |  18
-- Kagamine Rin   |  152.0 |   43.0 |  14
-- Kasane Teto    |  159.5 |   47.0 |  31
```

Starting from that same 3-row table, `DELETE` lets us remove rows — either all of them, or just some:

```sql
-- DELETE with no WHERE clause wipes every row, but keeps the empty table structure around:

DELETE FROM Vocaloids;

--      name      | height | weight | age
------------------+--------+--------+-----
-- (0 rows — the table still exists, it's just empty now)

-- Or, starting fresh from our 3-row table again, we can target just some rows:

DELETE FROM Vocaloids WHERE age < 18;

--      name      | height | weight | age
------------------+--------+--------+-----
-- Hatsune Miku   |  158.0 |   42.3 |  18
-- Kasane Teto    |  159.5 |   47.0 |  31
-- RIP Rin — she's the only one under 18.
```

And now... a command that'll make you cry: `DROP TABLE IF EXISTS`. Unlike `DELETE`, which only removes rows, this deletes the entire table — structure and all — from existence:

```sql
DROP TABLE IF EXISTS Vocaloids;  -- <--- Rip Miku, Rin, and Teto... and the table itself.
```

But after learning the basics, now let's learn these new commands:
- `LIKE`
- `BETWEEN`
- `IS NULL`
- `IN`

We'll use a different table for these (no particular reason, just didn't want to reuse the molecules table again). The table is called `users`:

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

But before that, let's learn `ORDER BY`, since it'll come in handy:

```sql
SELECT * FROM users ORDER BY age;

/*
Output:

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
*/
-- Just ORDER BY <column_name> and you get smallest-to-biggest; ordering by a text column
-- (e.g. ORDER BY first_name) goes A to Z the same way.
-- Now let's flip it around:

SELECT * FROM users ORDER BY age DESC;

/*
Output:

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
*/

-- DESC flips it to biggest-to-smallest (or Z to A for text).
-- General structure: SELECT <> FROM <> WHERE <> (if needed) ORDER BY <>;
```

### LIKE, BETWEEN, IN, and IS NULL

First up, `LIKE` (note: `LIKE` doesn't work on integers):

```sql
SELECT * FROM users WHERE first_name LIKE '%S';   -- rows where first_name ends with "S"
SELECT * FROM users WHERE last_name LIKE 'o%';    -- rows where last_name starts with "o"
SELECT * FROM users WHERE first_name LIKE '%a%';  -- rows with at least one "a" anywhere in the name
```

Since numbers are sometimes stored as strings (or vice versa), we'll need a couple of tricks. Since there'll be [hard times](https://www.youtube.com/watch?v=KpvHi6kqLHI&list=RDKpvHi6kqLHI&start_radio=1) ahead, let's prepare you like a warrior — because there's no second chance. (I'm joking, but making mistakes like this isn't exactly elegant.)

First, let's check the type of our columns — you probably don't want to cast an int into an int for no reason:

```sql
SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'users';
-- Don't change column_name to an actual column name from your table — leave it exactly
-- like this, so it prints every column in the table.
```

*(For the next part, let's imagine our `users` table has since picked up two more columns — `country` and `tier` — the same way real tables grow over time. Here's what checking the column types now shows:)*

```
/*
Output:

 column_name |     data_type
-------------+---------------------
 user_id     | integer
 first_name  | character varying  < --- a fancy way of saying text
 last_name   | character varying
 age         | integer
 country     | character varying
 tier        | character varying
*/
```

Now that we know the column types, let's manipulate them:

```sql
-- To cast a data type, we use "::" (two little colons). There's a bunch of casts you can do —
-- I won't show them all, since you won't need most of them.

-- Say we want everyone with the number 3 anywhere in their age — we don't care where, just
-- that it's there. But LIKE doesn't support integers, so what do we do?

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

-- Since LIKE doesn't support integers, we'll cast the age to text first, so LIKE can work on it:

SELECT * FROM users WHERE age::TEXT LIKE '%3%';

/*
Output:

 user_id | first_name | last_name | age | country | tier
---------+------------+-----------+-----+---------+------
       3 | Alex       | Rossi     |  34 | Ukraine | free
*/

-- We treated age as text just so LIKE could search it — otherwise it would complain about
-- finding nothing, since it's searching for text, not an integer.

-- Now, be careful with ORDER BY on a text column that holds numbers. If `age` were stored as
-- text and you did ORDER BY age directly, sorting goes character-by-character, not by numeric
-- value — so '10000000' would sort BEFORE '43', since '1' comes before '4', even though 43 is
-- the far smaller number. That's exactly why you'd cast it first:

SELECT * FROM users ORDER BY age::INTEGER;
-- This gives you a proper numeric sort. If age were left as text and sorted directly, you'd
-- get that same "10000000 before 43" kind of nonsense.

-- We can also cast into NUMERIC (decimal) with age::NUMERIC. Why bother? It's handy whenever
-- you need decimal arithmetic, or you're working with values that should be treated as
-- decimals rather than whole numbers. Be careful, though: if you try to cast text that isn't a
-- valid number, PostgreSQL will throw an error.
```

Now, back to our list:
- `LIKE` ✅ done
- `BETWEEN`
- `IS NULL`
- `IN`

Let's start with `BETWEEN` and `IN`, since they're pretty easy:

```sql
-- BETWEEN is simple: think of it as (x <= column_name <= y).
-- So instead of writing WHERE age >= 20 AND age <= 40, we can just write:
WHERE age BETWEEN 20 AND 40;
-- This selects every row that satisfies the condition (both 20 and 40 are included!)

SELECT * FROM users WHERE age BETWEEN 20 AND 40 ORDER BY age, first_name;
-- Orders from smallest to biggest age (within 20–40), and if ages tie, breaks the tie
-- alphabetically by first name.

/*
Output:

 user_id | first_name | last_name | age
---------+------------+-----------+-----
       5 | Mark       | Johnson   |  22
       7 | Luca       | Bianchi   |  26
       6 | Sophia     | Müller    |  26
       3 | Alex       | Rossi     |  34
*/

-- We can combine it with NOT too:

SELECT * FROM users WHERE NOT age BETWEEN 20 AND 40 ORDER BY age, first_name;
-- Keeps every row whose age is NOT between 20 and 40 (both ends included in the exclusion).

/*
Output:

 user_id | first_name | last_name  | age
---------+------------+------------+-----
       1 | Shrimp     | ShupoShupo |  19
      10 | Mia        | Smith      |  54
       2 | Daniel     | Kovalenko  |  59
       8 | Anya       | Novak      |  67
       4 | Elena      | Petrenko   |  71
       9 | Kenji      | Tanaka     |  76
*/
```

After `BETWEEN`, let's cover `IN` — a lifesaver. Why? Imagine your boss barges into your room at 3:24 AM and says: "Hey! 3:24 AM isn't for sleeping, it's for working — go find the age of the users with IDs 3, 15, 84, 921, 2831, 4632, and 5000, and tell me who's oldest." *(Imagine this is a much bigger `users` table now — thousands of rows, not just our 10.)*

What do you do? Refuse him? Not an option, you want to keep your job. So you get up and start writing:

```sql
SELECT * FROM users WHERE user_id = 3 OR user_id = 15 OR ... ;
```

Nobody wants to write that at 3:24 AM, so you use a small trick — `IN`:

```sql
SELECT * FROM users WHERE user_id IN (3, 15, 84, 1241, 2831, 4632, 5000) ORDER BY age DESC;

/*
Output:

 user_id | first_name |  last_name   | age
---------+------------+--------------+-----
    2831 | Stefan     | Xylander     |  87
    4632 | Matteo     | Zephyrus     |  82
      15 | John       | Quicksilver  |  65
       3 | Liam       | Nightshade   |  34
    1241 | Andrii     | Vintervarg   |  29
      84 | Sophie     | Featherstone |  18
    5000 | Giulia     | Obsidian     |  17
*/
-- Same answer, way less effort. We can layer on more logic too:

SELECT * FROM users WHERE user_id BETWEEN 500 AND 1000 AND age BETWEEN 15 AND 30 OR age IN (35, 45, 55, 65, 75, 85, 95) ORDER BY age;
-- Selects rows where user_id is between 500 and 1000 AND age is between 15 and 30 — OR the age
-- is exactly one of 35, 45, 55, 65, 75, 85, 95. Then orders everything by age.

/*
Output:

 user_id | first_name |  last_name   | age
---------+------------+--------------+-----
     566 | Dmitry     | Stormweaver  |  19
     921 | Elena      | Shadowend    |  35
     732 | Lev        | Thornevale   |  45
     835 | Nataliya   | Frostspire   |  55
     632 | Bohdan     | Ironwood     |  65
*/
```

Now let's cover `IS NULL` — easy, but genuinely important. `NULL` doesn't mean age = 0, or an empty string `''`. `NULL` means "we don't know" — if `age` is `NULL`, it means we don't know that user's age, not that they don't have one.

*(We'll use this table for the next part:)*

| run_id | model_name     | accuracy | training_time | dataset_size |
| ------:|----------------|---------:|--------------:|-------------:|
| 1      | Random Forest  | 91.2     |            18 | 10000        |
| 2      | XGBoost        | 94.8     |            27 | 10000        |
| 3      | Logistic Reg.  | 87.5     |             5 | 10000        |
| 4      | Neural Network | 96.1     |          NULL | 50000        |
| 5      | SVM            | 90.0     |            34 | 25000        |
| 6      | LightGBM       | 95.3     |            19 | 50000        |

Here's how to check for it:

```sql
SELECT * FROM models WHERE training_time IS NULL;
-- Prints rows from the models table where we don't know the training time (where the
-- `training_time` value is NULL — missing, buddy).

/*
Output:

 run_id | model_name      | accuracy | training_time | dataset_size
--------+-----------------+----------+---------------+--------------
      4 | Neural Network  |    96.1  |     NULL      |    50000
*/

SELECT * FROM models WHERE training_time IS NOT NULL;
-- Prints everything where training_time is NOT NULL — i.e. every row except the missing one.

/*
Output:

 run_id | model_name      | accuracy | training_time | dataset_size
--------+-----------------+----------+---------------+--------------
      1 | Random Forest   |    91.2  |      18       |    10000
      2 | XGBoost         |    94.8  |      27       |    10000
      3 | Logistic Reg.   |    87.5  |       5       |    10000
      5 | SVM             |    90.0  |      34       |    25000
      6 | LightGBM        |    95.3  |      19       |    50000
*/
```

### COUNT, SUM, AVG, MAX, MIN, and AS

Now for an easy-but-slightly-messy topic, since you'll need to memorize more than 5 commands at once: `COUNT, SUM, AVG, MAX, MIN, AS`. OMG, there's 6 of them — be careful not to [pass out](https://www.healthline.com/health/how-to-prevent-fainting)!

Let's start with the absolute basics: `COUNT` and `AS`, since one example each will make the whole point click.

```sql
-- 1. COUNT

-- What is COUNT? It counts how many items are in a column. If your boss asks "how many models
-- have we trained in total?" (after 14 years at this company), you don't count them by hand —
-- you use COUNT:

SELECT COUNT(model_name) FROM models;

/*
Output:

 count
-------
     6
*/
-- Exactly 6 models in that column (just 6 in 14 years... rough). Quick warning about NULL
-- values: if you run COUNT(training_time), it outputs 5, because COUNT completely ignores
-- NULLs! If you want to count every single row regardless of NULLs, use COUNT(*).

-- 2. AS

-- Down the line, using SUM, AVG, OVER, and friends, you'll get stuck with an ugly default
-- column name (especially with OVER()!). But you're a professional, so you want to call your
-- column "Hatsune Miku" instead of "sum" or whatever — that's what AS is for:

SELECT COUNT(model_name) AS Hatsune_Miku FROM models;

/*
Output:

 hatsune_miku
--------------
            6
*/
-- Really cool, really professional, no judgment incoming. (And no, that's not everything AS
-- can do — we'll use it plenty more later.)
```

Now let's cover `SUM, AVG, MAX, MIN`:

```sql
--| run_id | model_name     | accuracy | training_time | dataset_size |
--| -----: | -------------- | -------: | ------------: | -----------: |
--|      1 | Random Forest  |     91.2 |            18 |        10000 |
--|      2 | XGBoost        |     94.8 |            27 |        10000 |
--|      3 | Logistic Reg.  |     87.5 |             5 |        10000 |
--|      4 | Neural Network |     96.1 |          NULL |        50000 |
--|      5 | SVM            |     90.0 |            34 |        25000 |
--|      6 | LightGBM       |     95.3 |            19 |        50000 |

-- 1. SUM — adds every value in a column together.
SELECT SUM(dataset_size) AS grand_total_data FROM models;
-- 10000 + 10000 + 10000 + 50000 + 25000 + 50000

/*
Output:

 grand_total_data
------------------
           155000
(1 row)
*/

-- 2. AVG — the mathematical mean of a column.
SELECT AVG(accuracy) AS average_accuracy FROM models;
-- (91.2 + 94.8 + 87.5 + 96.1 + 90.0 + 95.3) / n, where n = 6 (the count of numbers)

/*
Output:

  average_accuracy
--------------------
 92.4833333333333333
(1 row)
*/
-- Watch out for the dreaded NULL: if one of your 6 numbers is NULL, AVG divides by 5 instead
-- of 6, since it skips NULLs entirely (which quietly breaks the "average" you had in mind).
-- Let's compute the average of training_time properly instead:
SELECT AVG(COALESCE(training_time, 0)) AS real_average FROM models;

/*
Output:

   real_average
--------------------
 17.1666666666666667
(1 row)
*/
-- Now it treats the missing value as 0 rather than silently excluding it from the count.

-- 3. MAX — the highest value in a column.
SELECT MAX(training_time) AS slowest_time FROM models;

/*
Output:

 slowest_time
--------------
           34
(1 row)
*/

-- 4. MIN — the lowest value in a column.
SELECT MIN(training_time) AS fastest_time FROM models;

/*
Output:

 fastest_time
--------------
            5
(1 row)
*/
```

### GROUP BY

Before the fun stuff, let's cover `GROUP BY` — basically a way to squash rows together based on shared values in a column. I'll admit `GROUP BY` isn't something I reach for that often, so we'll keep this part light, but it's still worth knowing well, since hiring managers don't care about my personal preferences.

Let's explain it Feynman-style (I could never really pull that off — the man was just too good).

First, a table, since it'll make the idea much clearer:

```sql
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

-- Now imagine: the boss barges into your bathroom mid-shower, furious, and starts
-- complaining: "Why does nobody understand GROUP BY??" he shouts, lathering shampoo into his
-- hair. "Shrimp! Give me a clean list of our 10 departments and how much they spent this
-- month!" (The real table has 1,000,000 rows of transactions across all 10 departments.)

-- If you used something like OVER() instead (you'll learn it later), you'd get all 1M rows
-- printed out — not cute. So instead, you squash them with GROUP BY!

-- *The boss keeps washing his hair, now mixing two different shampoos together for no reason.*

SELECT department, SUM(salary) AS Total_Income_Of_Department FROM company GROUP BY department;

/*
Output:

+------------+-----------------------------+
| department | Total_Income_Of_Department  |
+------------+-----------------------------+
| Sales      |                       29700 |
| HR         |                       24100 |
+------------+-----------------------------+

This gives us all 10 departments (imagine we actually have 10 here, not just Sales and HR).
*/

-- What happens if we GROUP BY something else entirely?

SELECT employee, department, AVG(salary) FROM company GROUP BY employee, department;

-- Not much changes, because no two employees share a name — each one is already its own
-- group of one, so nothing gets "squashed" together:

/*
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
*/

-- One more example, with a different table:

-- | order_id | customer_id | status     | order_amount |
-- | :------- | :---------- | :--------- | :----------- |
-- | 101      | Cust_A      | Shipped    | 150          |
-- | 102      | Cust_B      | Shipped    | 40           |
-- | 103      | Cust_A      | Cancelled  | 200          |
-- | 104      | Cust_C      | Shipped    | 300          |
-- | 105      | Cust_B      | Shipped    | 60           |
-- | 106      | Cust_A      | Shipped    | 80           |

-- Let's find just the shipped orders, and how many each customer had.

SELECT customer_id, COUNT(order_id) AS total_orders, SUM(order_amount) AS total_spent FROM orders WHERE status = 'Shipped' GROUP BY customer_id;

/*
Output:

+-------------+--------------+-------------+
| customer_id | total_orders | total_spent |
+-------------+--------------+-------------+
| Cust_A      |            2 |         230 |
| Cust_B      |            2 |         100 |
| Cust_C      |            1 |         300 |
+-------------+--------------+-------------+
*/
-- Cust_A had 2 shipped orders (the 3rd got cancelled, so it doesn't count), Cust_B had 2,
-- and Cust_C had 1.
```

### DISTINCT and JOIN

Now for something new, since there are so many ways to break yourself mentally and morally. Let's learn `DISTINCT` and `JOIN` (the scary one). We'll start with `DISTINCT`, since it's less painful:

```sql
-- experiment_id | model_name      | dataset
---------------+-------------------+---------
--             1 | Random Forest   | Titanic
--             2 | XGBoost         | Titanic
--             3 | XGBoost         | MNIST
--             4 | Neural Network  | MNIST
--             5 | Random Forest   | CIFAR-10
--             6 | XGBoost         | Titanic

-- Notice there's a copy: rows 2 and 6 are identical in model_name and dataset.

SELECT DISTINCT model_name, dataset FROM models;

/*
Output:

 model_name      | dataset
-----------------+----------
 Random Forest   | Titanic
 XGBoost         | Titanic
 XGBoost         | MNIST
 Neural Network  | MNIST
 Random Forest   | CIFAR-10
*/
-- We selected only model_name and dataset, so SQL checks if any row shares the same
-- (model_name, dataset) pair — if so, the duplicate gets dropped from the output.
-- But if we do this instead:

SELECT DISTINCT * FROM models;

/*
Output:

 experiment_id | model_name      | dataset
 --------------+-----------------+---------
             1 | Random Forest   | Titanic
             2 | XGBoost         | Titanic
             3 | XGBoost         | MNIST
             4 | Neural Network  | MNIST
             5 | Random Forest   | CIFAR-10
             6 | XGBoost         | Titanic
*/
-- Now everything prints, because SQL asks itself "are there any rows where EVERY selected
-- column is identical?" It notices rows 2 and 6 share model_name/dataset, but since we used
-- *, it also checks experiment_id — which differs — so neither row counts as a duplicate.
```

Now that we understand `DISTINCT`, let's move to `JOIN`. For this, we need two different tables, since the command's name already gives away what it does:

```sql
-- 1) This table is called 'models':

-- model_id |   model_name   |  framework   | created_at
------------+----------------+--------------+------------
--        1 | Random Forest  | scikit-learn | 2026-01-15
--        2 | XGBoost        | xgboost      | 2026-02-01
--        3 | Neural Network | PyTorch      | 2026-03-10
--        4 | LightGBM       | lightgbm     | 2026-03-15

-- 2) This table is called 'runs':

 --run_id | model_id | accuracy | training_time_secs |  status
----------+----------+----------+--------------------+-----------
--      1 |        2 |     94.8 |                120 | Completed
--      2 |        1 |     91.2 |                 45 | Completed
--      3 |        3 |     96.1 |               1800 | Completed
--      4 |        2 |     95.3 |                145 | Completed
--      5 |        4 |          |                 12 | Failed
--      6 |        3 |     89.4 |                600 | Completed

-- 3) Trying the code:

SELECT models.model_id, models.framework, runs.accuracy, runs.training_time_secs, runs.status FROM runs JOIN models ON runs.model_id = models.model_id;

-- Or with short aliases for runs and models (identical result):

SELECT m.model_id, m.framework, r.accuracy, r.training_time_secs, r.status FROM runs r JOIN models m ON r.model_id = m.model_id;

-- Breaking it down:
-- SELECT models.model_id, models.framework ... runs.status — grabs specific columns from
--   specific tables. models.model_id takes the model_id column from the models table;
--   runs.accuracy takes the accuracy column from the runs table.
-- FROM runs JOIN models — order doesn't matter here; FROM models JOIN runs gives the same
--   result.
-- ON runs.model_id = models.model_id — matches the IDs, so we don't end up pairing an
--   accuracy of 94.8 with Random Forest when it actually belongs to XGBoost. This makes sure
--   every row gets matched to the right model automatically.

/*
Output:

 model_id |  framework   | accuracy | training_time_secs |  status
----------+--------------+----------+--------------------+-----------
        2 | xgboost      |     94.8 |                120 | Completed
        1 | scikit-learn |     91.2 |                 45 | Completed
        3 | PyTorch      |     96.1 |               1800 | Completed
        2 | xgboost      |     95.3 |                145 | Completed
        4 | lightgbm     |          |                 12 | Failed
        3 | PyTorch      |     89.4 |                600 | Completed
*/
-- It gave us everything we needed, automatically matching each run to the right model
-- (thanks to ON runs.model_id = models.model_id).
```

That's it for `JOIN` (a.k.a. **Inner Join**) — we won't cover `LEFT JOIN` or `RIGHT JOIN` here, because even though they're useful, this isn't a full Data Engineer track.

*"If I ever stumble upon needing them, I'll come back and write about them." — said on 3 July 2026, at 2:49 PM.*

*(If you see `LEFT JOIN` or `RIGHT JOIN` written below this line, it means I ran into them eventually. If not, you probably won't need them either.)*

### Window functions: RANK, LAG, LEAD, PARTITION BY

Now for our last two topics:
1. Window functions — hard
2. `CASE WHEN` — easy

Let's start with window functions, since they're the easiest in my eyes — though apparently the internet thinks they're the hardest of the bunch. As always, we won't go full spelunker on this (we'll save the real depth for ML topics and math — though PySpark is genuinely interesting here too. *Time-traveling spoiler: it'll be easy if you already know SQL.*)

So let's get through this already! Window functions — you'd think it's as easy as staring at a window and thinking about functions, but that's not quite it.

We'll break window functions down into these parts:
- `RANK()` with `OVER(ORDER BY ...)`
- `LAG()`
- `LEAD()`
- `PARTITION BY`

```sql
-- We'll use this table (called 'models') for the example:

--| run_id | model_name     | accuracy | training_time | dataset_size |
--| -----: | -------------- | -------: | ------------: | -----------: |
--|      1 | Random Forest  |     91.2 |            18 |        10000 |
--|      2 | Neural Network |     94.8 |            27 |        10000 |
--|      3 | Random Forest  |     87.5 |             5 |        10000 |
--|      4 | Neural Network |     96.1 |          NULL |        50000 |
--|      5 | SVM            |     90.0 |            34 |        25000 |
--|      6 | Neural Network |     95.3 |            19 |        50000 |

-- (I added a couple of repeated model names, since it makes the example easier to follow.)

-- We'll start with RANK() and OVER() — why not just OVER() alone? Because without it, most of
-- these commands are basically unusable.

-- 1) RANK and OVER

SELECT run_id, model_name, accuracy, RANK() OVER(ORDER BY accuracy DESC) AS accuracy_ranking FROM models;
-- RANK() ranks every row by whatever you tell it in OVER(). Writing OVER(ORDER BY accuracy
-- DESC) ranks by accuracy, so the biggest accuracy gets 1st place, the second-biggest gets
-- 2nd, and so on. (I used DESC because plain ORDER BY accuracy would rank from smallest to
-- biggest instead, giving 1st place to the worst score — DESC flips that around.)

-- Always use OVER(), otherwise SQL will choke and die. What is OVER(), exactly? It lets us
-- keep every single row, without squashing everything the way GROUP BY does (THAT'S WHY I
-- LEFT IT OUT EARLIER, OKAY????). Normally, asking SQL for an average, sum, or count forces it
-- to smash all your rows together into a single line. So I saved you from wondering "where did
-- my rows go? Where's row 6? Gone. Crushed into the machine."

-- Why ORDER BY accuracy inside OVER()? Imagine your boss hands you a stack of medals and says:
-- "Rank these students from 1st place to 6th place." If he just says that and walks away
-- (which is like writing RANK() OVER()), your first question is: "ranked by WHAT?" Height?
-- Running speed? Test scores? So if you just write RANK() OVER(), everyone gets rank 1. But if
-- you write RANK() OVER(ORDER BY accuracy), you're telling it to actually rank by accuracy,
-- and you get a proper output:

/*
Output:

 run_id |   model_name   | accuracy | accuracy_ranking
--------+----------------+----------+------------------
      4 | Neural Network |     96.1 |                1
      6 | Neural Network |     95.3 |                2
      2 | Neural Network |     94.8 |                3
      1 | Random Forest  |     91.2 |                4
      5 | SVM            |     90.0 |                5
      3 | Random Forest  |     87.5 |                6
(6 rows)
*/

-- One more example, so we don't just say "yeahhh, I get it" while clearly not getting it:

SELECT model_name, training_time, dataset_size, RANK() OVER(ORDER BY dataset_size) AS dataset_s_rank, MAX(dataset_size) OVER() AS max_d_s, MIN(dataset_size) OVER() AS min_d_s FROM models WHERE training_time IS NOT NULL ORDER BY dataset_size;

/*
Output:

model_name     | training_time | dataset_size |   d_s_rank   | max_d_s | min_d_s
-------------  +---------------+--------------+--------------+---------+--------
Random Forest  |             5 |        10000 |            1 |   50000 |  10000
Random Forest  |            18 |        10000 |            1 |   50000 |  10000
Neural Network |            27 |        10000 |            1 |   50000 |  10000
SVM            |            34 |        25000 |            4 |   50000 |  10000
Neural Network |            19 |        50000 |            5 |   50000 |  10000
(5 rows)
*/
-- Tied dataset_size values (the three 10000s) share rank 1, and the next distinct value jumps
-- to rank 4 (since 3 rows came before it) — that's how RANK() handles ties. When we use OVER()
-- next to AVG, SUM, MAX, or MIN, it prints the same aggregate value on every row, instead of
-- squashing everything the way GROUP BY would.
```

Now let's cover `LAG()` and `LEAD()`, since they're pretty useful depending on the situation.

In plain words:
- `LAG()` — shows the *previous* row's value for the column you picked
- `LEAD()` — shows the *next* row's value for the column you picked

```sql
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

-- We extended the table a bit, so we get a better output.

-- First, a universal law — just like gravity: LAG() will always leave an empty (NULL) value
-- on the first row, since it looks at the previous row, and row 1 has no previous row to look
-- at. Same goes for LEAD(), except it's the LAST row that ends up empty, since it looks at the
-- next row, and there's no 11th row in a 10-row table.

-- 1) LAG
-- Starting with LAG() — I like the name, reminds me of "Lagtrain" by Inabakumori.

SELECT run_id, model_name, accuracy, LAG(accuracy) OVER(ORDER BY run_id) AS previous_accuracy FROM models;
-- Prints every previous value — but why is this useful? For now you'd just eyeball the
-- differences manually, but the next trick makes it click. (I used ORDER BY run_id, so it
-- knows to start from run 1 and work through to run 10.)

/*
Output:

run_id |   model_name   | accuracy | previous_accuracy
--------+----------------+----------+-------------------
      1 | Random Forest  |     91.2 |              NULL
      2 | Neural Network |     94.8 |              91.2  <-- you can check the difference
      3 | Random Forest  |     87.5 |              94.8
      4 | Neural Network |     96.1 |              87.5
      5 | SVM            |     90.0 |              96.1
      6 | Neural Network |     95.3 |              90.0
      7 | XGBoost        |     92.4 |              95.3
      8 | XGBoost        |     94.1 |              92.4
      9 | Random Forest  |     89.9 |              94.1
     10 | Neural Network |     97.2 |              89.9
(10 rows)
*/

-- Now let's use LAG() somewhere genuinely useful: a bit of arithmetic.

SELECT run_id, accuracy, accuracy - LAG(accuracy) OVER(ORDER BY run_id) AS accuracy_growth FROM models;
-- We just did `accuracy - previous_accuracy`, and that's genuinely useful, because now you
-- can see the exact swing in numbers.
-- MAIN RULE: if the result is positive, accuracy rose by that many points (assuming a 0–100
-- scale); if negative, it fell by that many points.

-- Imagine this whole list is from the same model, and you're watching how its accuracy
-- changes over time.

/*
Output:

run_id | accuracy | accuracy_growth
--------+----------+-----------------
      1 |     91.2 |            NULL  <-- number - NULL = NULL (always)
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
*/

-- 2) LEAD
-- Now let's look at LEAD() — the twin of LAG(). You can use it the same way, but let's try a
-- different angle. First, a simple visual, so you can see how it works:

SELECT run_id, model_name, accuracy, LEAD(accuracy) OVER(ORDER BY run_id) AS next_accuracy FROM models;

/*
Output:

run_id |   model_name   | accuracy | next_accuracy
-------+----------------+----------+---------------
     1 | Random Forest  |     91.2 |          94.8  <-- peek at row 2
     2 | Neural Network |     94.8 |          87.5
     3 | Random Forest  |     87.5 |          96.1
     4 | Neural Network |     96.1 |          90.0
     5 | SVM            |     90.0 |          95.3
     6 | Neural Network |     95.3 |          92.4
     7 | XGBoost        |     92.4 |          94.1
     8 | XGBoost        |     94.1 |          89.9
     9 | Random Forest  |     89.9 |          97.2
    10 | Neural Network |     97.2 |          NULL  <-- universal law! No 11th row.
(10 rows)
*/

-- Why look at the NEXT row? Imagine your boss hands you this list and says: "I want to know
-- how much accuracy we're about to gain or lose in the very next run, so I can plan ahead for
-- when to bet all our company funds on red."

-- We can do the same arithmetic, just flipped: LEAD(accuracy) - accuracy.

SELECT run_id, accuracy, LEAD(accuracy) OVER(ORDER BY run_id) - accuracy AS future_variance FROM models;

/*
Output:

run_id | accuracy | future_variance
-------+----------+-----------------
     1 |     91.2 |             3.6  <-- next run is 3.6% higher!
     2 |     94.8 |            -7.3  <-- heads up, next run drops 7.3%!
     3 |     87.5 |             8.6  <-- all in on black immediately!
     4 |     96.1 |            -6.1
     5 |     90.0 |             5.3
     6 |     95.3 |            -2.9
     7 |     92.4 |             1.7
     8 |     94.1 |            -4.2
     9 |     89.9 |             7.3
    10 |     97.2 |            NULL  <-- NULL - number = NULL (always) <-- universal law
(10 rows)
*/
```

Now let's cover `PARTITION BY` — genuinely useful, and one of my favorites. Let's dive in.

This table will help:

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

```sql
-- PARTITION BY isn't even that hard — think of it like this: you still use RANK() to rank the
-- models, but this time each rank only counts within its own group.

-- It's like having a giant basket of 20 apples, 20 oranges, and 20 bananas, each with a
-- `weight` column. A normal RANK() ranks all 60 fruits together. But with
-- PARTITION BY fruit_type, SQL resets the ranking counter for each group — a fresh 1-to-20
-- for apples, a fresh 1-to-20 for oranges, and a fresh 1-to-20 for bananas.

-- Let's rank each model's accuracy and see how they did.

SELECT model_name, accuracy, training_time, RANK() OVER(PARTITION BY model_name ORDER BY accuracy DESC) AS model_rank FROM models;

/*
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
*/
-- The highest-accuracy Neural Network gets a 1, and the highest-accuracy Random Forest also
-- gets a 1 — they don't interfere with each other's scoreboards at all.

-- Let's try the same idea with LAG() and AVG().

SELECT model_name, training_time, accuracy, AVG(accuracy) OVER(PARTITION BY model_name) AS avg_accuracy, accuracy - LAG(accuracy) OVER(PARTITION BY model_name ORDER BY accuracy DESC) AS accuracy_drop FROM models;
-- We add an ORDER BY inside LAG() too, so it's sure to start from the biggest value in each
-- group and work its way down.

/*
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
*/
-- accuracy_drop shows how much accuracy dropped between one run and the one right before it,
-- within the same model group. For Neural Network: first a drop of 1.1%, then 0.8%, then
-- 0.5% — so you can see just how much weaker each earlier run was. (You'll see NULLs, since
-- LAG() always leaves the first row of each group empty.) avg_accuracy shows the average
-- accuracy for each model's whole group.
```

### CASE WHEN

That's how all of this works — now let's wrap up with `CASE WHEN`. It's simply:
- if → `CASE WHEN`
- elif → `WHEN`
- else → `ELSE`

Simple as that. Let's move quickly~

```sql
-- We'll use the same table as before:

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

SELECT model_name, accuracy, CASE WHEN accuracy >= 95 THEN 'Excellent' WHEN accuracy >= 90 THEN 'Good' ELSE 'Needs Improvement' END AS rating FROM models;
-- END AS just gives a name to the new column.

/*
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
*/

-- One more example:

SELECT model_name, training_time, CASE WHEN training_time < 20 THEN 'Fast' WHEN training_time <= 60 THEN 'Medium' ELSE 'Slow' END AS speed FROM models;

/*
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
*/
```

We survived all this madness — now we push forward, with more madness incoming, because this was all just baby steps... sadly. But chin up: the more we advance, the more we suffer. So get ready to suffer.
