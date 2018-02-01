---
layout: NotesPage
title: TensorFlow 
permalink: /work_files/research/dl/nlp/tf_intro
prevLink: /work_files/research/dl/nlp.html
---

1. **Big Idea:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents11}  
    :   Express a numeric computation as a __graph__.

2. **Main Components:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents12}
    :   * __Graph Nodes__: are __*Operations*__ which have any number of:  
            * Inputs  
            &  
            * Outputs
        * __Graph Edges__: are __*Tensors*__ which flow between nodes   

3. **Example:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents13}  
    :   $$h = \text{ReLU}(Wx+b) \\ 
            \rightarrow$$  
        ![img](/main_files/dl/nlp/t_f/1.png){: width="20%"}  

4. **Components of the Graph:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents14}  
    :   * __Variables__: are stateful nodes which output their current value.  
            * State is retained across multiple executions of a graph.  
            * It is easy to restore saved values to variables  
            * They can be saved to the disk, during and after training  
            * Gradient updates, by default, will apply over all the variables in the graph  
            * Variables are, still, by "definition" __operations__
            * They constitute mostly, __Parameters__   
            ![img](/main_files/dl/nlp/t_f/2.png){: width="20%"}  
        * __Placeholders__: are nodes whose value is fed in at execution time.  
            * They do __not__ have initial values
            * They are assigned a:  
                * data-type  
                * shape of a tensor 
            * They constitute mostly, __Inputs__ and __labels__   
            ![img](/main_files/dl/nlp/t_f/3.png){: width="20%"}  
        * __Mathematical Operations__:   
            ![img](/main_files/dl/nlp/t_f/4.png){: width="20%"}  

5. **Sample Code:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents15}  
    :   ```python
        import tensorflow as tf
        b = tf.Variable(tf.zeros((100,)))
        W = tf.Variable(tf.random_uniform((784, 100) -1, 1))
        x = tf.placeholder(tf.float32, (100, 784))  
        h = tf.nn.relu(tf.matmul(x, W) + b)
        ```

6. **Running the Graph:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents16}  
    :   After defining a graph, we can __deploy__ the graph with a  
        __Session__: a binding to a particular execution context  
        > i.e. the Execution Environment  
    :   * CPU  
        or  
        * GPU

7. **Getting the Output:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents17}  
    :   * Create the _session_ object
        * Run the _session_ object on:  
            * __Fetches__: List of graph nodes.  
              Returns the outputs of these nodes. 
            * __Feeds__: Dictionary mapping from graph nodes to concrete values.  
              Specifies the value of each graph node given in the dictionary.   
    :   * CODE:  
            ```python
            sess = tf.Session()
            sess.run(tf.initialize_all_variables())
            sess.run(h, {x: np.random.random(100, 784)})
            ```

8. **Defining the Loss:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents18}  
    :   * Use __placeholder__ for __labels__
        * Build loss node using __labels__ and __prediction__
    :   * CODE:  
            ```python
            prediction = tf.nn.softmax(...) # output of neural-net
            label = tf.placeholder(tf.float32, [100, 10])
            cross_entropy = -tf.reduce_sum(label * tf.log(prediction), axis=1)
            ```

9. **Computing the Gradients:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents19}  
    :   * We have an __Optimizer Object__:  
        ```tf.train.GradientDescentOptimizaer```
        * We, then, add __Optimization Operation__ to computation graph:  
        ```tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)```
    :   * CODE:  
            ```python
            train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
            ```

10. **Training the Model:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents110}  
    :   ```python
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        for i in range(1000):
            batch_x, batch_label = data.next_batch()
            sess.run(train_step, feed_dict={x: batch_x, label: batch_label})  
        ```

11. **Variable Scope:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents111}  
    :   ![img](/main_files/dl/nlp/t_f/5.png){: width="100%"}

***

## Commands and Notes
{: #content2}

1. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents21}  
    :   ``` ```

2. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22}  
    :   ``` ```

3. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents23}  
    :   ``` ```

4. **TensorBoard:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents24}  
    :   ```File_writer  = tf.summary.FileWriter('log_simple_graph', sess.graph)```  
        ```tensorboard --logdir="path"```

5. **Testing if GPU works:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents25}  
    :   ```import tensorflow as tf
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c)) ```

6. **GPU Usage:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents26}  
    :   ```!nvidia-smi ```

7. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents27}  
    :   ``` ```

8. **Asynchronous:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents28}  
    :   ``` ```
