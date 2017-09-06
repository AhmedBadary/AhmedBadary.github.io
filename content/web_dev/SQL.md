---
layout: NotesPage
title: SQL
permalink: /work_files/web_dev/sql
prevLink: /work_files/web_dev.html
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
        * Properties:{: style="color: red"}
            1. Can group by a list of columns.
            2. Produces an aggregate Result PER Group.
            3. Can put grouping columns in "SELECT" list
            4. **_Cardinality_** of the output = # of distinct Group Values.
            5. Must be used in Conjunction with the Aggregate Functions.

    * **HAVING:** A predicate to _Filter Groups_.
        * Properties:{: style="color: red"}
            1. Applied after grouping and aggregating.
            2. Can contain anything tht could go in the "SELECT" list.
            3. Can only be used in Aggregate Queries.
            4. Optional.

    * **ORDER BY:** Specifies the attribute/column to sort data by.
        * Properties:{: style="color: red"}
            1. Can sort by multiple Attributes/Columns.

    * **LIMIT:** Produces the first n-rows.
        * Properties:{: style="color: red"}
            1. Used with the "ORDER BY" Method.


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

7. **Set Operators [IN | NOT IN]:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents17}
    :   Specify whether a value exists _IN_ the list specified after the clause.
    > **Example:**
    :   ```sql
        SELECT * FROM <Table-Name>

        /* IN */
        WHERE <attr> IN ("biking", "hiking", "tree climbing", "rowing");

        /* NOT IN */
        WHERE <attr> NOT IN ("biking", "hiking", "tree climbing", "rowing");
        ```    

8. **Nesting Queries (Multiple Tables) and SubQueries:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents18}
    :   ```sql
        SELECT * FROM exercise_logs WHERE type IN (
            SELECT type FROM drs_favorites );
        ```

9. **Fuzzy Search [Like]:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents19}
    :   Querying/Finding "Non-Exact" Matches.
    :   It is a way of **SELECTING** from the table by creating your own column.
    :   ```sql
        /* LIKE */
        SELECT <attr1>, <attr2>, ...
        FROM <table_name>
        WHERE <attri> LIKE <pattern>;
        ```

10. **'CASE' Statement:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents110} 
    :   Effectively behaves in the same way the "Case" or "Switch" statements do.
    :   ```sql
        SELECT <attr1>, <attr2>,
            CASE 
                WHEN <attri> > 220-30 THEN "above max"
                WHEN <attri>  > x THEN "above target"
                WHEN <attri>  > y THEN "within target"
                ELSE "below target"
            END as <attr-name>
        ```

***

## Types and Definitions
{: #content2}
1. **Data-Types:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents21}
    :   * TEXT
    :   * INTEGER
    :   * FLOAT
    :   * CHAR
    :   * INTEGER PRIMARY KEY

2. **ASYNCHRONUS:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22}

3. **ASYNCHRONUS:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents23} 
    
4. **ASYNCHRONUS:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents24}

5. **ASYNCHRONUS:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents25}  
    
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

6. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents36} 


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

6. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents9 #bodyContents96}

7. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents9 #bodyContents97}

8. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents9 #bodyContents98}