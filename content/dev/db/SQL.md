---
layout: NotesPage
title: SQL
permalink: /work_files/dev/db/sql
prevLink: /work_files/dev/db.html
---

<div markdown="1" class = "TOC">
# Table of Contents

  * [Commands](#content1)
  {: .TOC1}
  * [Types and Definitions](#content2)
  {: .TOC2}
  * [Modifying a Table](#content4)
  {: .TOC4}
  * [Joins](#content3)
  {: .TOC3}
  * [PostGreSQL](#content5)
  {: .TOC5}
  * [NOTES](#content9)
  {: .TOC9}
</div>



***
***

## Commands
{: #content1}

1. **Creating a Table:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents11}
    :   ```sql
        CREATE TABLE <Table-Name> (<attr> <type>, ...);
        ```
    > BY CONVENTION: We add the "id" attribute as the first column.  

        > Example:  
        :   ```sql
            CREATE TABLE shopping (id TEXT, price INTEGER);
            ```   

2. **Inserting into a Table:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents12}
    :   ```sql
        INSERT INTO <Table-Name> VALUES (<Schema-Row>);
        ```
    Or, to not specify values to all of the attributes in the Schema, 
    :   ```sql
        INSERT INTO <Table-Name>(attr1, attr3, attr6) VALUES (val1, val3, val6);
        ```

3. **Querying a Table:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents13}
    :   ```sql
        SELECT <attr>
        FROM <Table-Name>
        WHERE <predicate>
        GROUP BY <attr1, attr2, ..., attrn> (i.e. Col. List)
        HAVING <predicate>
        ORDER BY <attr1, attr2, ..., attrn> (i.e. Col. List)
        LIMIT <int>
        ```
    > Example:  
    :   ```sql
        SELECT S.id, AVG(S.price)
        FROM shopping as S
        WHERE S.price < 100
        GROUP BY S.date_of_purchace
        HAVING S.color = 'red'
        ORDER BY S.date_of_purchace
        LIMIT 10
        ```

4. **Querying Methods:**{: style="color: SteelBlue"}{: .bodyContents1  #bodyContents14}
    * **GROUP BY:** Partition the table into Groups with the _same "GROUP BY" Cols_.
        * **Properties:**
            1. Can group by a list of columns.
            2. Produces an aggregate Result PER Group.
            3. Can put grouping columns in "SELECT" list
            4. **_Cardinality_** of the output = # of distinct Group Values.
            5. Must be used in Conjunction with the Aggregate Functions.

    * **HAVING:** A predicate to _Filter Groups_.
        * **Properties:**
            1. Applied after grouping and aggregating.
            2. Can contain anything tht could go in the "SELECT" list.
            3. Can only be used in Aggregate Queries.
            4. Optional.

    * **ORDER BY:** Specifies the attribute/column to sort data by.
        * **Properties:**
            1. Can sort by multiple Attributes/Columns.
            2. Order by Ascending or Descending Orders using ```ASC|DESC;```
                > **Example:**
                >   > ```sql
                >   > ORDER BY column1, column2, ... ASC|DESC;
                >   > ```

    * **LIMIT:** Produces the first n-rows.
        * **Properties:**
            1. Used with the "ORDER BY" Method.

41. **Nested Queries:**{: style="color: SteelBlue"}{: .bodyContents1  #bodyContents141}
    :   Queries can be nested inside queries for convenience.
    :   ```sql
        SELECT <attr> FROM <Table-Name-1> WHERE <predicate>
        IN || NOT IN
        (SELECT <attr-2> FROM <Table-Name-2> WHERE <predicate-2>)
        ```  
    :   > The correlated sub-query has to be recomputed for each of Table-1's tuples. 

42. **Correlated Queries:**{: style="color: SteelBlue"}{: .bodyContents1  #bodyContents142}
    :   Nested Sub-Queries can reference the original query.  
    <img src="/main_files/db/3.png" width="52%" style="position: relative;">

5. **Aggregate Functions:**{: style="color: SteelBlue"}{: .bodyContents1  #bodyContents155}
    :   Computes a (statistical) summary of the results and outputs it.

    * **Functions:**  
        AVG - SUM - MAX - MIN - COUNT - ROUND

    * **Distinct Aggregates:**  Computes the Aggregate function _distinct tuples_.
    > Example: 
    :   ```sql
        SELECT AVG(DISTINCT S.gpa);
        ```
        > Notice that the "DISTINCT" tag is inside the Aggregate Function.

    * **Properties:**
        1. Produces 1 Row and 1 Column of output.


5. **Auto-Incriment:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents15}
    :   To set the id of the table to be auto-assigned and auto-incrimented:
    :   ```sql
        CREATE TABLE <Table-Name> (id INTEGER PRIMARY KEY AUTOINCREMENT,...)
        ```

6. **Logical Operators [AND | OR]:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents16}
    :   Specify multiple predicates with Logical Operators.
    > **Example:**
    :   ```sql
        SELECT * FROM <Table-Name>

        /* AND */
        WHERE <attr> > 50 AND <attr> < 30;

        /* OR */
        WHERE <attr> > 50 OR <attr> > 100;
        ```

7. **Set Operators:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents17}
    :   SQL allows us to perform _set operations_ on the data.
    :   * **Set-Membership:**  Specify whether a value exists _IN_ the list specified after the clause .
    > **Example:**
    :   ```sql
        SELECT * FROM <Table-Name>

        /* IN */
        WHERE <attr> IN ("biking", "hiking", "tree climbing", "rowing");

        /* NOT IN */
        WHERE <attr> NOT IN ("biking", "hiking", "tree climbing", "rowing");

        /* (NOT) EXISTS */
        (NOT) EXISTS (<SubQuery>)

        /* ANY */
        WHERE 

        ```   
    :   * **Set Comparison:** Compares the elements in two different sets.
    > **Example:**
    :   ```sql
        SELECT * FROM <Table-Name>

        /* (NOT) EXISTS */
        (NOT) EXISTS (<SubQuery>)

        /* ANY */
        WHERE <attr-i> >= ANY (<SubQuery>)

        /* ALL */      
        WHERE <attr-i> >= ALL (<SubQuery>)
        ```   
    :   * **Union and Intersect:**  makes the _union_ or the _intersection_ of two sets.
    > **Example:**
    :   ```sql
        SELECT * FROM <Table-Name-1> WHERE <predicate-1>

        /* UNION */
        UNION
        SELECT * FROM <Table-Name-2> WHERE <predicate-2>


        /* INTERSECT */
        INTERSECT
        SELECT * FROM <Table-Name-3> WHERE <predicate-3>
        ```   
    :   * **Set Difference:** returns the difference between two sets returned by two queries.
    > **Example:**
    :   ```sql
        SELECT * FROM <Table-Name-1> WHERE <predicate-1>

        /* EXCEPT */
        EXCEPT
        SELECT * FROM <Table-Name-2> WHERE <predicate-2>
        ```   
    : > Notice: Set Operations does **NOT** return Multi-Sets.  
      > > The returned results are **distinct**.  

    <img src="/main_files/db/1.png" width="40%" style="position: relative;left:80px">

71. **Multi-Set Operators:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents171}
    :   Multi-Set operators are the same as the [set operators](#bodyContents17) with an added term, ```ALL```.
    <img src="/main_files/db/2.png" width="43%" style="position: relative;left:41px">

72. **Relational Division:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents172}
    :   **The Query:** "Find sailors who've reserved all boats."  
    __Equivalent to:__ "Sailors with no counter-exapmle missing boat."  
    <img src="/main_files/db/4.png" width="52%" style="position: relative;">


8. **Nesting Queries (Multiple Tables) and SubQueries:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents18}
    :   ```sql
        SELECT * FROM exercise_logs WHERE type IN (
            SELECT type FROM drs_favorites );
        ```

9. **Fuzzy Search [Like]:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents19}
    :   Querying/Finding "Non-Exact" Matches.
    :   ```sql
        /* LIKE */
        SELECT <attr1>, <attr2>, ...
        FROM <table_name>
        WHERE <attri> LIKE <pattern>;
        ```

10. **Regular Expressions:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents110}
    :   Matching Strings.
    :   ```sql
        /* LIKE */
        SELECT <attr1>, <attr2>, ...
        FROM <table_name>
        WHERE <attri> ~ <pattern>;
        ```


11. **'CASE' Statement:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents111} 
    :   Effectively behaves in the same way the "Case" or "Switch" statements do.
    :   It is a way of **SELECTING** from the table by creating your own column.
    :   ```sql
        SELECT <attr1>, <attr2>,
            CASE 
                WHEN <attri> > 220-30 THEN "above max"
                WHEN <attri>  > x THEN "above target"
                WHEN <attri>  > y THEN "within target"
                ELSE "below target"
            END as <attr-name>
        ```

12. **'WITH' clause [temp views]:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents112} 
    :   The ```WITH``` clause allows us to create a temporary view to use in our queries.  
        > Also know as, "Common Table Expressions".
    :   ```sql
        WITH view1(bid, scount) AS
                (SELECT * FROM <Table-1>),
            view2 AS
                (SELECT * FROM <Table-2>, view1)

        SELECT * FROM view2;
        ```
        > Notice that the second query can reference the first.  

    :   OR  
    :   ```sql
        SELECT bname, scount
        FROM Boats2 B,
            (SELECT * FROM <Table-1>) AS view1(bid, scount)

        WHERE <predicate>
        ```
    :   > Without using the ```WITH``` clause.

***

## Types and Definitions
{: #content2}
1. **Data-Types:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents21}
    :   * TEXT
    :   * INTEGER
    :   * FLOAT
    :   * CHAR
    :   * INTEGER PRIMARY KEY

2. **View:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22}
    :   **A view** is a named query.  
        Views are not materialized until run. They can be run multiple times.  
        > i.e. We have to evaluate it everytime we run it.  

        They, also, must have unique column names.
    :   ```sql
        CREATE VIEW view_name
        AS select_statement
        ```
    
***


## Modifying The Table
{: #content4}

1. **Moifying a Row:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents41}
    :   We can modify a row using the "UPDATE" Command.

    ```sql
    UPDATE <Table-Name> SET <attri> = "something"
    WHERE <attri> = "something else"
    ```
    > We use the "WHERE" command to tell the method which ROW to update.

2. **Deleting a Row:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents42}
    :   We can delete a row using the "DELETE" Command.

    ```sql
    DELETED FROM <Table-Name>
    WHERE <attri> = "something else"
    ```

3. **Altering Tables:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents43} 
    :   We can alter the table by adding columns/attributes.
    ```sql
        ALTER TABLE <Table-Name> ADD <new-attr> <new-type>
    ```

    > To change the Default Value:
    ```sql
        ALTER TABLE <Table-Name> ADD <new-attr> <new-type> default "UnKnown"
    ```

4. **Deleting a Table:**{: style="color: SteelBlue  "}{: .bodyContents4 #bodyContents44}
    :   Will completely erase the table with the data in it.
    ```sql
        DROP TABLE <Table-Name>
    ```     

***

## Joins
{: #content3}

#### JOIN variants  
<img src="/main_files/db/6.png" width="52%" style="position: relative;">

1. **Cross Join:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents31}
    :   Produces a result set (known as the Cartesian Product) which is the number of rows in the first table multiplied by the number of rows in the second table if no WHERE clause is used along with CROSS JOIN.

    :   > If WHERE clause is used with CROSS JOIN, it functions like an INNER JOIN.

    :   ```sql
        SELECT * 
        FROM table1 
        CROSS JOIN table2;
        ```
    OR
    :   ```sql
        SELECT * 
        FROM table1, table2 
        ```
        ![img](/main_files/web_dev/images/cj.png){: width="34%"}

2. **Inner Join:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents32}
    :   selects all rows from both participating tables as long as there is a match between the columns.

    :   > An inner join of A and B gives the result of A intersect B, i.e. the inner part of a Venn diagram intersection.

    :   ```sql
        SELECT * 
        FROM table1 INNER JOIN table2 
        ON table1.column_name = table2.column_name; 
        ```
    OR
    :   ```sql
        SELECT * 
        FROM table1
        JOIN table2 
        ON table1.column_name = table2.column_name; 
        ```
        ![img](/main_files/web_dev/images/ij.png){: width="34%"}

3. **Outter Join:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents33}
    :   Returns all rows from both the participating tables which satisfy the join condition along with rows which do not satisfy the join condition. The SQL OUTER JOIN operator (+) is used only on one side of the join condition only.

    * **Subtypes:**
        1. LEFT OUTER JOIN or LEFT JOIN
        2. RIGHT OUTER JOIN or RIGHT JOIN
        3. FULL OUTER JOIN

    ```sql
        Select * 
        FROM table1, table2 
        WHERE conditions [+]; 
    ```

4. **Left/Right and FULL [OUTER] Joins:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents34} 
    * **LEFT JOIN:** joins two tables and fetches all matching rows of two tables for which the SQL-expression is true, plus rows from the frist table that do not match any row in the second table.

    ```sql
        SELECT *
        FROM table1
        LEFT [ OUTER ] JOIN table2
        ON table1.column_name=table2.column_name;
    ```
    ![img](/main_files/web_dev/images/lj.png){: width="34%"}

    * **RIGHT JOIN:** joins two tables and fetches rows based on a condition, which is matching in both the tables, and the unmatched rows will also be available from the table written after the JOIN clause.

    ```sql
        SELECT *
        FROM table1
        RIGHT [ OUTER ] JOIN table2
        ON table1.column_name=table2.column_name;
    ```
    ![img](/main_files/web_dev/images/rj.png){: width="34%"}

    * **FULL OUTER JOIN:** Combines the results of both left and right outer joins and returns all (matched or unmatched) rows from the tables on both sides of the join clause.

    ```sql
        SELECT * 
        FROM table1 
        FULL OUTER JOIN table2 
        ON table1.column_name=table2.column_name;
    ```
    ![img](/main_files/web_dev/images/foj.png){: width="32%"}


5. **Self Joins [Joining Tables to themselves]:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents35}
    :   ```sql
        SELECT a.column_name, b.column_name... 
        FROM table1 a, table1 b 
        WHERE a.common_filed = b.common_field;
        ```
    :   > Use the names given to the table "left and right" (i.e. a & b) to specify the attributes for each respective table.

6. **Natural Join:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents36} 
    :   Behaves like the [Inner Join](#bodyContents32) but matches on columns that have the same name/attribute.
    :   > Natural Joins do **NOT** have ```ON``` clause. 
    :   > Natural Joins, unfortuntely, could change meaning on a different schema.

***

## PostGreSQL
{: #content5}

1. **Operators:**{: style="color: SteelBlue  "}{: .bodyContents5 #bodyContents51}
    * **\l:**  View List of DataBase
    * **\q:**  Quit (Back to Terminal)
    * **\c {name}:**  Connect to DataBase
    * **\d:**  View List of Schemas and their respective info (Tables)
    * **\d {Schema-Name}:**  View List of Attributes/Columns in the Schema

2. **Commands:**{: style="color: SteelBlue  "}{: .bodyContents5 #bodyContents52} \\
    * **Create DataBase:**  
    ```SQL
    CREATE DATABASE <Name>
    ```
    * **Create User:**  
    ```SQL
    CREATE USER {Name} WITH PASSWORD {'password'}
    ```  
    * **Delete Database:**  
    ```SQL
    DROP DATABASE {name}
    ```  
    * **Load DB from server:**  
    ```SQL
    psql  -U postgres -h server
    ```  

***

## Notes
{: #content9}
1. **Filter "GROUP BY" _ONLY_ using "HAVING":**{: style="color: SteelBlue"}{: .bodyContents9 #bodyContents91}
    * **Wrong:**
    :   ```sql
        SELECT type, SUM(calories) AS total_calories FROM exercise_logs
            WHERE total_calories > 150
            GROUP BY type;
        ```
    > This errors because you can NOT use the new created Aggregate value in Where.

    * Right:
    :   ```sql
        SELECT type, SUM(calories) AS total_calories FROM exercise_logs
            GROUP BY type
            HAVING total_calories > 150;
        ```

2. **To Select All Entries with at least "x" entries for a specific "attr":**{: style="color: SteelBlue"}{: .bodyContents9 #bodyContents92}
    :   ```sql
        SELECT <attr>
        FROM <Table>
        GROUP BY <attr>
        HAVING COUNT(*) >= x;
        ```

3. **Left Join, Right Join Equivalence:**{: style="color: SteelBlue"}{: .bodyContents9 #bodyContents93}
    :   Notice that you only need one of "LEFT JOIN" or "RIGHT JOIN".
    :   You can turn one into the other by changing the order of the "joined" tables.

4. **Default Values:**{: style="color: SteelBlue  "}{: .bodyContents9 #bodyContents94}
    :   The _Default Value_ of an attribute is "NULL".

5. **Nested Joins:**{: style="color: SteelBlue  "}{: .bodyContents9 #bodyContents95}
    :   Notice that JOINS can be nested.
    :   You can nest them by adding more "JOIN" statements.

6. **Avoiding the "NULL" Values:**{: style="color: SteelBlue  "}{: .bodyContents9 #bodyContents96}
    :   To change a value of an attribute from "NULL" to something you desire while _querying_, use the ```COALESCE(old_attr, new_val)``` Function.
    :   > **Example:**
    :   ```sql
        COUNT(COALESCE({attr}, {alternative}))
        ```
    :   > If the alternative is anything but "NULL", this will give count to the number of people having the same attr as a "NULL" value.

7. **Joining Tables with Different Rows:**{: style="color: SteelBlue  "}{: .bodyContents9 #bodyContents97}
    :   To have all the rows whether they match up or not, use the ```OUTER JOIN``` statement.
    :   > Outer Joins will concatenate a "NULL" value if a match isn't found.

8. **Set-Operators and Logical-Operators Equivalence:**{: style="color: SteelBlue  "}{: .bodyContents9 #bodyContents98}
    :   * ```UNION``` $$\iff$$ ```OR```
        > Equivalent
    :   * ```INTERSECT``` $$\nLeftrightarrow$$ ```OR```
        > NOT-Equivalent

9. **ARGMAX:**{: style="color: SteelBlue  "}{: .bodyContents9 #bodyContents99}
    :   The _ARGMAX_ function is not a native of SQL.  
    :   > In the _example_ below:  
        > > The query on the **left:** returns _ALL_ sailors that are tied for ARGMAX.
        > > The query on the **right:** returns _ONLY ONE_ sailor from those tied for max, randomly.
    <img src="/main_files/db/5.png" width="52%" style="position: relative;">