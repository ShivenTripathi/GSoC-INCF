## Weekly Reports [Neurostars Forum](https://neurostars.org/t/gsoc-2021-project-idea-21-2-hdnet-projects-developing-a-python-based-codebase-for-analyzing-stimulus-prediction-capabilities-of-neurons-improve-stimulus-prediction)

### **Weekly Report 1**
June 4-June 11

1. What I've been doing:

    a. Literature Review: [Stimulus-dependent Maximum Entropy Models of Neural Population Codes](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002922), [Weak pairwise correlations imply strongly correlated network states in a neural population](https://www.nature.com/articles/nature04701), [The simplest maximum entropy model for collective behavior in a neural network](https://iopscience.iop.org/article/10.1088/1742-5468/2013/03/P03011/meta)

    b. From these papers identified validation methods in use and finalised them on discussion with mentors

    c. Discussing implementation strategies, tested existing codebase using demo files and real dataset

2. What I'll do next week:

    a. Continue with coding validation methods

3. Blockers:

    a. Was down with fever for a couple of days so could not make much progress on code this week

### **Weekly Report 2**

June 12-June 19

1. What I've been doing:

    a. Validation using log-likelihood for spike train data
    
    b. Working with HDNet to fit spike train data
2. What I'll do next week:

    a. Complete log-likelihood validation and discuss the next set of methods for estimation of predictive accuracy
3. Blockers:

    a. Had some difficulty working with the preexisting code in HDNet as not all of them had complete examples, so discussed that with mentors

### **Weekly Report 3**
June 20 -June 27

1. What I've been doing

    a. Completed code for validation using log-likelihood data, most common codewords, started for higher order interactions
2. What I'll do next week
    
    a. Complete code for higher order interactions, find other missing docs in HDNet I could fill
3. Blockers
    
    a. None

### **Weekly Report 4**
June 28 - July  4

1. What I've been doing
 
    a. Completed code for validation using higher-order interactions

    b. Testing code for pushing to source
 
2. What I'll do next week
 
    a. Complete documentation and testing to merge code with source

3. Blockers

    a. None

### **Weekly Report 5**
July 5 - July 12

1. What I've been doing

    a. Completed all validation methods for maximum entropy models
    
    b. restructured codebase after code review from mentors
 
    c. started reading on information estimation methods, and samplers for spiketrain for implementation next week
2. What I'll do next week
    
    a. Implement better sampler for spiketrain data and/or start coding on information estimation methods
    
    b. finish writing tests to merge with origin
3. Blockers

    a. None

### **Weekly Report 6**
July 13 - July 20

1. What I've been doing
    
    a. Added metropolis hastings sampler for spiketrain, modified it to work with validations like gibbs sampler
 
    b. Added detailed demos and examples for how someone could use the code without having to know too much about it
 
    c. Started implementing CDM Entropy estimate in HDNet, after which I'll pick up mutual information estimation
2. What I'll do next week
    
    a. Complete CDM Entropy validation
 
    b. Continue with NSB and Miller Madow information estimation
3. Blockers
 
    a. None

### **Weekly Report 7**
July 21 - July 28

1. What I’ve been doing
    
    a. Started working on integrating legacy MATLAB codebases to HDNet
    
    b. Had discussions on research directions project could take, and added task of building interface for .m files to use HDNet in Python

2. What I’ll do next week
    
    a. Test CDMEntropy interface, add MI methods

3. Blockers
    
    a. None

### **Weekly Report 8**
July 29 - August 5

1. What I’ve been doing
    
    a. Completed interface for CDME with examples
    
    b. Started extension for HDNet Contrib containing other such additional features to HDNet

2. What I’ll do next week
    
    a. Completed MI using CDME

3. Blockers
    
    a. None

### **Weekly Report 9**
August 6 - August 13

1. What I’ve been doing
    
    a. Completed MI using CDME, tested for data

    b. Completed a data loader for using raw stimulus data, and started testing NSB for MI estimates
2. What I’ll do next week
    
    a. Completed documentation and final work report
3. Blockers

    a. None

### **Weekly Report 10**
August 13 - August 20

1. What I’ve been doing
    
    a. Wrote documentation and examples for usage

    b. Completed final evaluation report

-----------------------------------------------------------------------------------------------
This became too verbose to be useful and most of progress was later updated on Neurostars forum.

### 24/05/2021
1. Added old codes
2. Start Maintaining Reading List/Tasks in README

### 25/05/2021
1. Maybe more frequent(biweekly) meetings, right now booked for Tuesdays
2. Also discuss with Chris often

### 27/05/2021
1. Types of Models:
    * First Order:
        * S1
        * T1
        * Independent Model
    * Second Order:
        * S2
        * T2
        * Independent Model + Correlation
2. For single neurons, S2 predicts firing rates better than S1 model
3. S2 is better in predicting population neural responses - quantified by log-likelihood ratio
3. S2 predicts probabilities of patterns of activity with minimal bias - S1 assigns low probab. to some codes, to the extent they dont get generated

### 01/06/2021
1. Watching course- Computational Neuroscience, Coursera UoW
2. Meeting:
    * Got demo file to use
    * One way to test validation: model returns 1st 2nd 3rd order correlations
    * What validation to use clearer now, run the demo, understand how it would be fit, and then discuss with Chris