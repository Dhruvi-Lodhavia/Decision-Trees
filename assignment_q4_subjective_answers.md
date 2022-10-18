Fit time - For RIDO and RIRO cases, the theoretical time complexity can be written as O(M x (N^2) X max_depth) whereas for DIDO and DIRO is O(M X N X max_depth).
1. Fit time vsM (N=30) -  In this case the maximum time is taken for Real Input, Discrete Output case, followed by Real Input Real Output case. The time taken in fitting for Real input is higher than discrete since here, I have calculated the best split at every point(greedy algorithm). This requires more complexity than it did in discrete input cases. <p>
Further in case of Real Input, Discrete Output case, the entropy calculation is more complex than calculation of variance in Real Input Real Output case. Hence Real Input cases take more time as the number of features increases.<p> 
Also, in the graph the complexity of real graphs is almost linear whereas for discrete cases it is linear and increases very slowly with increase in the attributes.
In Real input cases, the number of columns do not decrease with every iteration whereas for discrete cases it does. 
<p align = center>
<img width="500" src = "./Fit time vsM (N=30).png" >
</p>
2. Fit time vsN(M=5) -  In this case the maximum time is taken for Real Input, Discrete Output case, followed by Real Input Real Output case. The time taken in fitting for Real input is higher than discrete since here, I have calculated the best split at every point(greedy algorithm). This requires more complexity than it did in discrete input cases.<p> Further in case of Real Input, Discrete Output case, the entropy calculation is more complex than calculation of variance in Real Input Real Output case. Hence Real Input cases take more time as the number of features increases. <p>
Also, in the graph the complexity of real graphs is almost quadratic whereas for discrete cases it is linear and increases slowly with increase in the samples(slope is less).
In Real input cases, the number of columns do not decrease with every iteration whereas for discrete cases it does. Hence in discrete input cases the time complexity is O(N) whereas for real input cases if O(N^2)
<p align = center>
<img width="500" src = "./Fit time vsN(M=5).png" >
</p>
PREDICT TIME - In both the cases of predict time, as N and M increases, for all the cases the time remains almost the same since I am using DFS for transversal of the tree. In all the cases the process is the same. Theoretical time complexity can be written as O(max_depth+M) for DFS<p>
3. Predict time vsM (N=30) - 
<p align = center>
<img width="500" src = "./Predict time vsM (N=30).png" >
</p>
4. Predict time vsN(M=5) -Here as N(the samples) increases, we need to transverse the tree more number of times. This leads to increase in time as N increases.
<p align = center>
<img width="500" src = "./Predict time vsN(M=5).png" >
</p>

