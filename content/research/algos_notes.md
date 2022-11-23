---
layout: NotesPage
title: Algorithms - Notes
permalink: /work_files/research/algos_notes
prevLink: /work_files/research.html
---


# Algorithms Notes


* __Time Complexity__: 
    * __Sum of natural numbers:__: 
        * Sum of numbers from $$1$$ to $$n-1$$:  $$ n\times\frac{(n-1)}{2}$$ 
        * Sum of numbers from $$1$$ to $$n$$:  $$ n\times\frac{(n+1)}{2}$$ 
        > Note: this is [equal to](https://math.stackexchange.com/questions/185728/intuition-on-the-sum-of-first-n-1-numbers-is-equal-to-the-number-of-ways-of-pi) the number of ways of picking $$2$$ items out of $$n$$.  
            I.E. $$n$$ choose $$2$$:   
            $$\frac{n(n-1)}{2}=\left(\begin{array}{l} n \\ 2 \end{array}\right)=1+2+\ldots(n-1) = [n!/((n-2)! * 2!)]$$ 



* __Dynamic Programming__: 
    * All DP problems can be represented as directed acyclic graphs.
    * Many (all?) reduce to finding the shortest path in a DAG
    * Equivalently, Shortest path in a DAG is a DP problem


* __Graph Algorithms__: 
    * __Dijkstras Algorithm__: ([resource (Blog!!!)](https://www.scaler.com/topics/data-structures/dijkstra-algorithm/)) finds the shortest path from one node to another node - infact it finds the shortest node from the starting node to every other node
        * __Complexity__: 
            * $$\mathcal{O}\left(V^2\right)$$  using the adjacency matrix representation of graph.
            * Reduced to $$\mathcal{O}((V+E) \log V)$$ using adjacency list representation of the graph.

        * Description:
            In this algorithm each vertex will have two properties defined for it:
                * Visited property:-
                    This property represents whether the vertex has been visited or not.
                    We are using this property so that we don't revisit a vertex.
                    A vertex is marked visited only after the shortest path to it has been found.
                * Path property:-
                    This property stores the value of the current minimum path to the vertex. Current minimum path means the shortest way in which we have reached this vertex till now.
                    This property is updated whenever any neighbour of the vertex is visited.
                    The path property is important as it will store the final answer for each vertex.

        * __Algorithm__: 
            1. Mark the source node with a current distance of 0 and the rest with infinity.
            1. Set the non-visited node with the smallest current distance as the current node, lets say C.
            1. For each neighbour N of the current node C: add the current distance of C with the weight of the edge connecting C-N. If it is smaller than the current distance of N, set it as the new current distance of N.
            1. Mark the current node C as visited.
            1. Go to step 2 if there are any nodes are unvisited.
        * __Pseudocode__: 
            <button>Pseudocode</button>{: .showText value="show" onclick="showTextPopHide(event);"}
            ![img](https://cdn.mathpix.com/snip/images/oVjogTEV6cYz82Q8je7rUaWpYFIW6b3vFTWxtDnMkYI.original.fullsize.png){: width="100%" hidden=""}  

    * __Shortest Path Algorithms__: 
        For a *__general weighted graph__*, we can calculate single source shortest distances in $$O(VE)$$  time using __Bellman–Ford Algorithm__.  
        For a graph with _no negative weights_, we can do better and calculate single source shortest distances in $$O(E + VLogV)$$ time using __Dijkstra’s algorithm__.
        For a DAG, we can in $$\mathcal{O}(V+E)$$ using *__topological sorting__*.  