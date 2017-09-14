---
layout: NotesPage
title: HADOOP vs SQL
permalink: /work_files/dev/db/HvsS
prevLink: /work_files/dev/db.html
---

<div markdown="1" class = "TOC">
# Table of Contents

  * [Schema on Read vs Schema on Write](#content1)
  {: .TOC1}
  * [Data Storage](#content2)
  {: .TOC2}
  * [Query Delivery](#content3)
  {: .TOC3}
</div>


***
***

## HADOOP vs SQL $$\implies$$ Non-Relational vs Relational
<p class="message">
HADOOP and SQL are very different in the way they manage data. <br />  
Thankfully, there are <i>three</i> main differences between them.
</p>

## Schema on Read vs Schema on Write
{: #content1}

1. **HADOOP:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents11}
    :   > Uses Schema on Read.
    : * **In moving data from a Data-Base 'A' to a Data-Base 'B', we need nothing.**
    : * **In reading the data from the Data-Base, we:**  
        1. Apply the rules to the code that queries the data.  
            > Instead of configuring the structure of the data ahead of time.  
        2. How to adapt the data in 'A'

2. **SQL:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents12}
    :   > Uses Schema on Write.
    : * **In moving data from a Data-Base 'A' to a Data-Base 'B', we need to:**  
        1. Know the structure (Schema) of 'B'.
        2. Know how to adapt the data in 'A'.
        3. Ensure that the data-types of data in 'A' matches those of 'B'.

***

## Data Storage
{: #content2}

1. **HADOOP:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents21}
    :   > Data is stored as a compressed file; with text or other data-types.  
    : * **The moment the data enters into HADOOP:**  
        1. The data (file) is replicated into multiple nodes in the HADOOP distributed filing systems.  
            > i.e. It is archtichted for an unlimited number of servers.
        2. HADOOP will keep track of the data, its multiple copies, and the, respective, machines the copies are stored in.
    : * **The structure of a query in HADOOP comes in the form of a _JAVA program_ that:**  
        1. Defines the request (Query).
        2. Distribute the calculation of that search across all the copies of the data.
        3. _Parallizes_ instead of _serializes_ the calculation of the query across the servers.
            > i.e. Each server works on a _portion_ of the data.
        4. Sends the results from each server to a _reducer program_, on the same node-cluster.

2. **SQL:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents22}
    :   > Data is stored in a _Logical Form_; with inter-related tables and defined columns.
    : * **The moment the data enters into SQL-DB:**  
        1. Create a table that stores the information.
        2. Fit all of the table on the same server.

***

## Query Delivery
{: #content3}

**Question.** If a query is performed and the results of the query are _incomplete_, how does each framework deal with this situation?

1. **HADOOP:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents31}
    :   > One-phase Commit; immideate results (even if incomplete).

2. **SQL:**{: style="color: SteelBlue  "}{: .bodyContents3 #bodyContents32}
    :   > Two-phase Commit; must have consistency across all nodes before returning the result. 
