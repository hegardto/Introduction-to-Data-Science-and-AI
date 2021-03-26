# -*- coding: utf-8 -*-

# -- Sheet --

# **DAT405 Introduction to Data Science and AI, 2021, Study Period 3** <br/>
# **Assignment 5: Reinforcement learning and Classification** <br/>
# 
# - David Arvidsson 19941029-1414, MPDSC, ardavid@student.chalmers.se: 17 hours
# - Johan Hegardt 19970714-6230, MPDSC, johan.hegardt@gmail.com: 17 hours


# # Primer
# 
# ## Decision Making
# The problem of **decision making under uncertainty** (commonly known as **reinforcement learning**) can be broken down into
# two parts. First, how do we learn about the world? This involves both the
# problem of modeling our initial uncertainty about the world, and that of drawing conclusions from evidence and our initial belief. Secondly, given what we
# currently know about the world, how should we decide what to do, taking into
# account future events and observations that may change our conclusions?
# Typically, this will involve creating long-term plans covering possible future
# eventualities. That is, when planning under uncertainty, we also need to take
# into account what possible future knowledge could be generated when implementing our plans. Intuitively, executing plans which involve trying out new
# things should give more information, but it is hard to tell whether this information will be beneficial. The choice between doing something which is already
# known to produce good results and experiment with something new is known
# as the **exploration-exploitation dilemma**.
# 
# ## The exploration-exploitation trade-off
# 
# Consider the problem of selecting a restaurant to go to during a vacation. Lets say the
# best restaurant you have found so far was **Les Epinards**. The food there is
# usually to your taste and satisfactory. However, a well-known recommendations
# website suggests that **King’s Arm** is really good! It is tempting to try it out. But
# there is a risk involved. It may turn out to be much worse than **Les Epinards**,
# in which case you will regret going there. On the other hand, it could also be
# much better. What should you do?
# It all depends on how much information you have about either restaurant,
# and how many more days you’ll stay in town. If this is your last day, then it’s
# probably a better idea to go to **Les Epinards**, unless you are expecting **King’s
# Arm** to be significantly better. However, if you are going to stay there longer,
# trying out **King’s Arm** is a good bet. If you are lucky, you will be getting much
# better food for the remaining time, while otherwise you will have missed only
# one good meal out of many, making the potential risk quite small.


# ## Overview
# * To make things concrete, we will first focus on decision making under **no** uncertainity, i.e, given we have a world model, we can calculate the exact and optimal actions to take in it. We will first introduce **Markov Decision Process (MDP)** as the world model. Then we give one algorithm (out of many) to solve it.
# 
# 
# * Next, we will work through one type of reinforcement learning algorithm called Q-learning. Q-learning is an algorithm for making decisions under uncertainity, where uncertainity is over the possible world model (here MDP). It will find the optimal policy for the **unknown** MDP, assuming we do infinite exploration.


# ## Markov Decision Process


# Markov Decision Process (MDP) provides a mathematical framework for modeling sequential decision making under uncertainty. A MDP consists of five parts: the specific decision times, the state space of the environment/system, the available actions for the decision maker, the rewards, and the transition probabilities between the states.
# 
# * Decision epochs: $t={1,2,...,T}$, where $T\leq \infty$
# * State space: $S=\{s_1,s_2,...,s_N\}$ of the underlying environment
# * Action space $A=\{a_1,a_2,...,a_K\}$ available to the decision maker at each decision epoch
# * Reward functions $R_t = r(a_t,s_t,s_{t+1})$ for the current state and action, and the resulting next state
# * Transition probabilities $p(s'|s,a)$ that taking action $a$ in state $s$ will lead to state $s'$
# 
# At a given decision epoch $t$ and system state $s_t$, the decions maker, or *agent*, chooses an action $a_t$, the system jumps to a new state $s_{t+1}$ according to the transition probability $p(s_{t+1}|s_t,a_t)$, and the agent receives a reward $r_t(s_t,a_t,s_{t+1})$. This process is then repeated for a finite or infinite number of times.
# 
# A *decision policy* is a function $\pi: s \rightarrow a$, that gives instructions on what action to choose in each state. A policy can either be *deterministic*, meaning that the action is given for each state, or *randomized* meaning that there is a probability distribution over the set of possible actions. Given a specific policy $\pi$ we can then compute the the *expected total reward* when starting in a given state $s_1 \in S$, which is also known as the *value* for that state, 
# 
# $$V^\pi (s_1) = E\left[ \sum_{t=1}^{T} r(s_t,a_t,s_{t+1}) {\Large |} s_1\right] = \sum_{t=1}^{T} r(s_t,a_t,s_{t+1}) p(s_{t+1} | a_t,s_t)$$ 
# 
# where $a_t = \pi(s_t)$. To ensure convergence and to control how much credit to give to future rewards, it is common to introduce a *discount factor* $\gamma \in [0,1]$. For instance, if you think all future rewards should count equally, you would use $\gamma = 1$, while if you only care less about future rewards you would use $\gamma < 1$. The expected total *discounted* reward becomes
# 
# $$V^\pi( s_1) = \sum_{t=1}^T \gamma^{t-1} r(s_t,a_t, s_{t+1}) p(s_{t+1} | s_t, a_t) $$
# 
# Now, to find the *optimal* policy we want to find the policy $\pi^*$ that gives the highest total reward $V^{\pi^*}(s)$ for all $s\in S$. That is
# 
# $$V^{\pi^*}(s) \geq V^\pi(s), s\in S$$
# 
# The problem of finding the optimal policy is a _dynamic programming problem_. It turns out that a solution to the optimal policy problem in this context is the *Bellman equation*. The Bellman equation is given by
# 
# $$V(s) = \max_{a\in A} \left\{\sum_{s'\in S} p(s'|s,a)( r(s,a,s') +\gamma V(s')) \right\}$$
# 
# Thus, it can be shown that if $\pi$ is a policy such that $V^\pi$ fulfills the Bellman equation, then $\pi$ is an optimal policy.
# 
# A real world example would be an inventory control system. Your states would be the amount of items you have in stock. Your actions would be the amount to order. The discrete time would be the days of the month. The reward would be the profit.  
# 
# A major drawback of MDPs is called the "Curse of Dimensionality". MDPs unfortunately do not scale very well with increasing sets of states or actions.   


# ## Question 1


# In this first question we work with the deterministic MDP, no code is necessary in this part.
# 
# Setup:
# 
# * The agent starts in state **S**
# * The actions possible are **N** (north), **S** (south), **E** (east), and **W** west. 
# * Note, that you cannot move outside the grid, thus all actions are not available in every box.
# * When reaching **F**, the game ends (absorbing state).
# * The numbers in the boxes represent the rewards you receive when moving into that box. 
# * Assume no discount in this model: $\gamma = 1$
# 
# The reward of a state $r(s=(x, y))$ is given by the values on the grid:
#     
# | | | |
# |----------|----------|---------|
# |-1 |1|**F**|
# |0|-1|1|  
# |-1 |0|-1|  
# |**S**|-1|1|
# 
# Let $(x,y)$ denote the position in the grid, such that $S=(0,0)$ and $F=(2,3)$.
# 
# **1 Answer the following questions (1p)**
# 
# **a)** What is the optimal path of the MDP above? Is it unique? Submit the path as a single string of directions. E.g. NESW will make a circle.
# 
# **Answer:** Optimal path = EENNN. 
# It is not unique, as EENNWNE gives the same expected reward of 0 given that $\gamma = 1$. It is also possible to go between to states with rewards -1 and 1 countless of times, since $\gamma = 1$.
# 
# **b)** What is the optimal policy (i.e. the optimal action in each state)?
# 
# **Answer:** The optimal policy in a state *s* is to chose the direction (action *a*) that takes the agent to the state *s+1* with the highest expected total reward from *s+1* to *F*. 
# 
# Example: In the start position, S = (0,0), the agent is given two possible actions: $a_1$ and $a_2$. In this case $a_1$ equals moving the agent one step north (N) and $a_2$ one step east (E). For these two actions, the highest expected reward is calculated from the state *s+1* to *F*. For $a_1$, the highest expected reward is -1 and for $a_2$ 0. Therefore, the agent moves one step east and recalculates the actions, i.e. the highest expected rewards, from the new state.
# 
# The optimal action for each state is presented below. 
# 
# | | | |
# |----------|----------|---------|
# |E|E|**F**|
# |S/N/E |N/E |N|  
# |N/E |N/S/E/W|S/N|  
# |**N/E**|E|N/E|
# 
# **c)** What is expected total reward for the policy in 1b)?
# 
# **Answer:** The expected total reward for the policy in 1b) is 0, as the optimal path between *S* and *F* is of total reward 0 (-1 + 1 - 1 + 1 = 0). 


# ## Value Iteration


# For larger problems we need to utilize algorithms to determine the optimal policy $\pi^*$. *Value iteration* is one such algorithm that iteratively computes the value for each state. Recall that for a policy to be optimal, it must satisfy the Bellman equation above, meaning that plugging in a given candidate $V^*$ in the right-hand side (RHS) of the Bellman equation should result in the same $V^*$ on the left-hand side (LHS). This property will form the basis of our algorithm. Essentially, it can be shown that repeated application of the RHS to any intial value function $V^0(s)$ will eventually lead to the value $V$ which statifies the Bellman equation. Hence repeated application of the Bellman equation will also lead to the optimal value function. We can then extract the optimal policy by simply noting what actions that satisfy the equation. The process of repeated application of the Bellman equation what we here call the _value iteration_ algorithm.


# The value iteration algorithm practically procedes as follows:
# 
# ```
# epsilon is a small value, threshold
# for x from i to infinity 
# do
#     for each state s
#     do
#         V_k[s] = max_a Σ_s' p(s′|s,a)*(r(a,s,s′) + γ*V_k−1[s′])
#     end
#     if  |V_k[s]-V_k-1[s]| < epsilon for all s
#         for each state s,
#         do
#             π(s)=argmax_a ∑_s′ p(s′|s,a)*(r(a,s,s′) + γ*V_k−1[s′])
#             return π, V_k 
#         end
# end
# 
# ```


# **Example:** We will illustrate the value iteration algorithm by going through two iterations. Below is a 3x3 grid with the rewards given in each state. Assume now that given a certain state $s$ and action $a$, there is a probability of 0.8 that that action will be performed and a probability of 0.2 that no action is taken. For instance, if we take action **E** in state $(x,y)$ we will go to $(x+1,y)$ 80 percent of the time (given that that action is available in that state, that is, we stay on the grid), and remain still 20 percent of the time. We will use have a discount factor $\gamma = 0.9$. Let the initial value be $V^0(s)=0$ for all states $s\in S$. 
# 
# | | | |  
# |----------|----------|---------|  
# |0|0|0|
# |0|10|0|  
# |0|0|0|  
# 
# 
# **Iteration 1**: The first iteration is trivial, $V^1(s)$ becomes the $\max_a \sum_{s'} p(s'|s,a) r(s,a,s')$ since $V^0$ was zero for all $s'$. The updated values for each state become
# 
# | | | |  
# |----------|----------|---------|  
# |0|8|0|
# |8|2|8|  
# |0|8|0|  
#   
# **Iteration 2**:  
#   
# Staring with cell (0,0) (lower left corner): We find the expected value of each move:  
# Action **S**: 0  
# Action **E**: 0.8( 0 + 0.9 \* 8) + 0.2(0 + 0.9 \* 0) = 5.76  
# Action **N**: 0.8( 0 + 0.9 \* 8) + 0.2(0 + 0.9 \* 0) = 5.76  
# Action **W**: 0
# 
# Hence any action between **E** and **N** would be best at this stage.
# 
# Similarly for cell (1,0):
# 
# Action **N**: 0.8( 10 + 0.9 \* 2) + 0.2(0 + 0.9 \* 8) = 10.88 (Action **N** is the maximizing action)  
# 
# Similar calculations for remaining cells give us:
# 
# | | | |  
# |----------|----------|---------|  
# |5.76|10.88|5.76|
# |10.88|8.12|10.88|  
# |5.76|10.88|5.76|  


# ## Question 2 
# 
# **2a)** Implement the value iteration algorithm just described here in python, and show the converging optimal value function and the optimal policy for the above 3x3 grid. Hint: use the pseudo-code above as a starting point, but be sure to explain what every line does.(2.5p)
# 
# Code in cell below.
# 
# **Converging optimal values:**
# | | | |  
# |----------|----------|---------|  
# |44.7|51.04|44.7|
# |51.04|47.14|51.04|  
# |44.7|51.04|44.7|  
# 
# **Optimal policy:**
# | | | |  
# |----------|----------|---------|  
# |E/S|S|S/W|
# |E|N/E/S/V|W|  
# |N/E|N|N/W|  
# 
# **2b)** Explain why the result of 2a) does not depend on the initial value $V_0$.(0.5p)
# 
# The convergence rate is determined by The Bellman equation, and the convergance values are in this case dependent on the discount rate, *gamma*, and the actual rewards of each state. Since the discount factor is < 1 the discount will after enough iterations be close to zero for the initial value $V_0$. Therefore, as the algorithm goes trough more and more iterations the initial value diminishes and is taken over by the weight of the reward and the more current values of the grid. This is what causes the optimal value function to converge.


#Task 2a)

#Initialize grids with rewards and initial values (as well as temp. values). 
rewards = [[0,0,0],[0,10,0],[0,0,0]]
values = [[0,0,0],[0,0,0],[0,0,0]]
values_temp = [[0,0,0],[0,0,0],[0,0,0]]

#Variables holding number of iterations and gamma
nr_iter = 100
discount_factor = 0.9

#For-loop that iterates maximum nr_iter of times or until the values converge
for t in range(1,nr_iter+1):
    
    #For every row in the grid
    for row in range(0,len(values[0][:])):

        #For every column in the grid
        for column in range(0,len(values[:][0])):
            
            #Initialize maxValue, holding the highest value for the current state 
            maxValue = 0
            
            #Separating calculation of V1 from the rest of the value calculations
            if t > 1:
                
                #Calculate south
                if row - 1 != -1:
                    x = 0.2*(rewards[row][column] + discount_factor * values[row][column]) + 0.8*(rewards[row-1][column] + discount_factor * values[row-1][column])
                    
                    #If value of moving in this direction > maxValue, update maxValue
                    if x > maxValue:
                        maxValue = x
                
                #Calculate north
                if row + 1 != len(values):
                    x = 0.2*(rewards[row][column] + discount_factor * values[row][column]) + 0.8*(rewards[row+1][column] + discount_factor * values[row+1][column])
                    
                    #If value of moving in this direction > maxValue, update maxValue
                    if x > maxValue:
                        maxValue = x
                
                #Calculate west
                if column - 1 != -1:
                    x = 0.2*(rewards[row][column] + discount_factor * values[row][column]) + 0.8*(rewards[row][column-1] + discount_factor * values[row][column-1])
                    
                    #If value of moving in this direction > maxValue, update maxValue
                    if x > maxValue:
                        maxValue = x
                
                #Calculate east
                if column + 1 != len(values):
                    x = 0.2*(rewards[row][column] + discount_factor * values[row][column]) + 0.8*(rewards[row][column+1] + discount_factor * values[row][column+1])
                    
                    #If value of moving in this direction > maxValue, update maxValue
                    if x > maxValue:
                        maxValue = x
                
                if maxValue > (values[row][column]):
                    values_temp[row][column] = round(maxValue,2)
                       
            else:
                #Calculate south
                if row - 1 != -1:
                    x = 0.2*(values[row][column]) + 0.8*(values[row-1][column])
                    
                    #If value of moving in this direction > maxValue, update maxValue
                    if x > maxValue:
                        maxValue = x
                
                #Calculate north
                if row + 1 != len(values):
                    x = 0.2*(values[row][column]) + 0.8*(values[row+1][column])
                    
                    #If value of moving in this direction > maxValue, update maxValue
                    if x > maxValue:
                        maxValue = x
                
                #Calculate west
                if column - 1 != -1:
                    x = 0.2*(values[row][column]) + 0.8*(values[row][column-1])
                    
                    #If value of moving in this direction > maxValue, update maxValue
                    if x > maxValue:
                        maxValue = x
                
                #Calculate east
                if column + 1 != len(values):
                    x = 0.2*(values[row][column]) + 0.8*(values[row][column+1])
                    
                    #If value of moving in this direction > maxValue, update maxValue
                    if x > maxValue:
                        maxValue = x

            values_temp[row][column] = round(maxValue,2)
            
            column += 1
        row += 1

    #Check convergence, defined by the maximum change of values between iterations < epsilon
    quit = False
    if t != 1:
        maximum = 0
        for row in range (0,len(values_temp)):
            for column in range (0,row):
                if abs(values[row][column] - values_temp[row][column]) > (maximum):
                    maximum = abs(values[row][column] - values_temp[row][column])
        if maximum < 0.1:
            print(t)
            quit = True

    #Break loop if converged
    if quit == True:
        break

    #Update the grid values
    for row in range (0,len(values_temp)):
        for column in range (0, len(values_temp)):
            values[row][column] = values_temp[row][column]

#The final values
print(values)

# ## Reinforcement Learning (RL)
# Until now, we understood that knowing the MDP, specifically $p(s'|a,s)$ and $r(a,s,s')$ allows us to efficiently find the optimal policy using the value iteration algorithm. Reinforcement learning (RL) or decision making under uncertainity, however, arises from the question of making optimal decisions without knowing the true world model (the MDP in this case).
# 
# So far we have defined the value function for a policy through $V^\pi$. Let's now define the *action-value function*
# 
# $$Q^\pi(s,a) = \sum_{s'} p(s'|a,s) [r(a,s,s') + \gamma V^\pi(s')]$$
# 
# The value function and the action-value function are directly related through
# 
# $$V^\pi (s) = \max_a Q^\pi (s,a)$$
# 
# i.e, the value of taking action $a$ in state $s$ and then following the policy $\pi$ onwards. Similarly to the value function, the optimal $Q$-value equation is:
# 
# $$Q^*(s,a) = \sum_{s'} p(s'|a,s) [r(a,s
# ]\,s') + \gamma V^*(s')]$$
# 
# and the relationship between $Q^*(s,a)$ and $V^*(s)$ is simply
# 
# $$V^*(s) = \max_{a\in A} Q^*(s,a).$$
# 
# ## Q-learning
# 
# Q-learning is a RL-method where the agent learns about its unknown environment (i.e. the MDP is unknown) through exploration. In each time step *t* the agent chooses an action *a* based on the current state *s*, observes the reward *r* and the next state *s'*, and repeats the process in the new state. Q-learning is then a method that allows the agent to act optimally. Here we will focus on the simplest form of Q-learning algorithms, which can be applied when all states are known to the agent, and the state and action spaces are reasonably small. This simple algorithm uses a table of Q-values for each $(s,a)$ pair, which is then updated in each time step using the update rule in step $k+1$
# 
# $$Q_{k+1}(s,a) = Q_k(s,a) + \alpha \left( r(s,a) + \gamma \max \{Q_k(s',a')\} - Q_k(s,a) \right) $$ 
# 
# where $\gamma$ is the discount factor as before, and $\alpha$ is a pre-set learning rate. It can be shown that this algorithm converges to the optimal policy of the underlying MDP for certain values of $\alpha$ as long as there is sufficient exploration. While a constant $\alpha$ generally does not guarantee us to reach true convergence, we keep it constant at $\alpha=0.1$ for this assignment.
# 
# ## OpenAI Gym
# 
# We shall use already available simulators for different environments (worlds) using the popular OpenAI Gym library. It just implements [different types of simulators](https://gym.openai.com/) including ATARI games. Although here we will only focus on simple ones, such as the [Chain enviroment](https://gym.openai.com/envs/NChain-v0/) illustrated below.
# ![alt text](https://chalmersuniversity.box.com/shared/static/6tthbzhpofq9gzlowhr3w8if0xvyxb2b.jpg)
# The figure corresponds to an MDP with 5 states $S = \{1,2,3,4,5\}$ and two possible actions $A=\{a,b\}$ in each state. The arrows indicate the resulting transitions for each state-action pair, and the numbers correspond to the rewards for each transition.
# 
# ## Question 3 (1.5p)
# You are to first familiarize with the framework using its [documentation](http://gym.openai.com/docs/), and then implement the Q-learning algorithm for the Chain enviroment (called 'NChain-v0') using default parameters. Finally print the $Q^*$ table at convergence. Convergence is **not** a constant value, rather a stable plateau with some noise. Take $\gamma=0.95$. You can refer to the Q-learning (frozen lake) Jupyter notebook shown in class, uploaded on Canvas. Hint: start with a small learning rate.
# 
# ## Reminder: You should write your explanations with your own words, not something found online or you risk being reported for plagiarism.
# 
# ## Question 4
# 
# **4a)** Define the MDP corresponding to the Chain environment above and verify that the optimal $Q^*$ value obtained using simple Q-learning is the same as the optimal value function $V^*$ for the corresponding MDP's optimal action. Hint: compare values obtained using value iteration and Q-learning. (1p)
# 
# **Comparison values obtained by value iteration and Q-learning, optimal strategies(move forward for all states)**
# 
# | | | | |
# |----------|----------|----------|----------| 
# |**State**|**Value** **iteration**|**Q**-**learning**|**Difference**|
# |1|60.16|59.49|0.66|  
# |2|63.64|63.14|0.50|
# |3|68.05|67.56|0.49|
# |4|74.81|72.86|1.95|
# |5|83.00|81.51|1.49| 
# 
# From the comparison it is clear that there is a small difference between the values found for Q-learning and the value iteration. This could be explained by the values for the learning rate, as well as the value for epsilon as these both are adjustable parameters that effect the results obtained by the Q-learning algorithm. 
# 
# 
# **4b)** What is the importance of exploration in RL? Come up with an example that will help you explain. (1p)
# 
# The agent in a RL algorithm has for every new observation a choice to make. Either it can explore new parts of a state space where it is possible that a better solution is situated, or it can exploit the parts of the state space which is already explored. The decision between exploring and exploiting is a trade-off. If the agent is more likely to exploit, a solution will be found faster. This is called greedy exploration and carries a big risk of not finding the optimal solution since it could be locking to one okay, but not great, solution. This is due to the fact that if it is more likely that the agent will explore new parts of the state space, it might find better solution but it will take longer time and more computational power. Many problems that can be solved with RL methods have multiple possible actions to make for every state. If we look at for example the game of chess, the state space grows exponentially for every move and it will very soon be clear that it is not possible to explore all different states of the state space. Therefore, the importance of exploration lies not only in deciding when the agent should explore rather than exploit but also how it should explore.
# 
# To explore the possible moves in a game of chess for every state randomly will simply not be possible. That is why exploration should be performed with some sort of heuristic function or probability distribution. For chess, this could for example be that the model should start exploring new parts of the state space if the probability of winning that a certain action leads to starts to decrease. If the search explores a tree with this heuristic, it could find parts of the state space where it is very likely to win at the beginning. The RL method will then exploit that part of the tree, if the probability of winning decreases the exploit/explore algorithm might tell the model to start exploring new parts of the state space instead. The model then starts exploring parts where it might not be as likely to win but where it is possible to find moves that quickly raises the winning probability.


#Task 3
import gym
import numpy as np
import random

#Create environemnt for Nchain-v0
env = gym.make("NChain-v0")
env.reset()
env.close()

#Set number of episodes to 15000 (assuming convergance/stable plateau after this),
#learning rate to predefined 0.1 and gamma/discount rate to predifined 0.95. 
num_episodes = 15000
gamma = 0.95
learning_rate = 0.1

#Epsilon set to 0.5 gives equal importance to exploration and exploitation
epsilon = 0.5

#Set Q-table, containing all state + action combinations, to 0
Q = np.zeros([5, 2])

#Iterate through all episodes and update Q-table
for _ in range(num_episodes):
    
    #Define variables
    state = env.reset()
    done = False
    count = 0
    
    #Assuming convergance/stable plateau per episode after 300 iterations
    while done == False:
    
    #Action selection (exploring if random number < epsilon, otherwise exploiting)
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state,:])

        #Perform chosen action and get observation back from the environment
        new_state, reward, done, info = env.step(action)

        #Update Q-table with value of selected action
        update = reward + (gamma*np.max(Q[new_state,:])) - Q[state, action]
        Q[state,action] += learning_rate*update 
        state = new_state

        #Assuming convergance/stable plateau per episode after 300 iterations
        count += 1
        if count == 300:
            done = True

#Print Q-table
Q

#Defining an array with rewards. 
#The first value for every state equals the reward of steping forward, the second of steping backward.
rewards = np.array([[0.0,2.0],[0.0,2.0],[0.0,2.0],[0.0,2.0],[10.0,2.0]])

#Defining an array with the max value (Qmax of both actions) for every state. 
state_values = np.array([0.0,0.0,0.0,0.0,0.0])

#Defining an array with the values of moving forward and backward in every state.
values = np.array([[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]])

nr_iter = 15000
discount_factor = 0.95

for t in range(0, nr_iter): 
    
    for index in range(0,len(state_values)):
        #
        prob = random.uniform(0.75,0.85)

        if index == 4:
            values[index][0] = prob * (rewards[index][0] + discount_factor * state_values[index]) + (1-prob) * (rewards[index][1] + discount_factor * state_values[0])
            values[index][1] = prob * (rewards[index][1] + discount_factor * state_values[0]) + (1-prob) * (rewards[index][0] + discount_factor * state_values[index])
            
        else:
            values[index][0] = prob * (rewards[index][0] + discount_factor * state_values[index+1]) + (1-prob) * (rewards[index][1] + discount_factor * state_values[0])
            values[index][1] = prob * (rewards[index][1] + discount_factor * state_values[0]) + (1-prob) * (rewards[index][0] + discount_factor * state_values[index+1])

        state_values[index] = max(values[index][0],values[index][1])

values

#Compare convergance of value iteration & Q-learning, best policies for each state (highest V)
q_star = Q[:,0].reshape(-1,1)
value_iter = values[:,0].reshape(-1,1)

print(q_star-value_iter)

# ## Question 5
# 
# **5a)** Give a summary of how a decision tree can be extended to random forests and how they work. (1.25p)
# 
# A random forest constitutes of many uncorrelated decision trees and makes predictions based on the individual prediction of the trees that are a part of the forest. A random forest can be used both for classification problems and regression tasks and is a supervised machine learning algorithm. 
# 
# In for example a classification task, each decision tree in the random forest will make its own label predictions for a certain datapoint. The random forest then sums the number of predictions for each label and the ultimate prediction of the random forest is the label with the most predictions from the individual trees. The principle behind random forests is commonly refered to as *the wisdom of crowds* which essentially is the idea that a large group of uncorrelated decision trees are able to make more accurate predicitions and produce better results than any individual decision tree. Some of the trees in the random forest will most likely make incorrect predcitions, but as a group the forest will produce predicitions with a higher accuracy. 
# 
# When developing a random forest it is essential it is important to make sure that the trees maintain a low correlation. This is done by considering feature randomness, which essentially means that each tree in a random forest don't have the same exact nodes, instead the nodes are selected out of a random subset of features. Uncorrelation is also maintained by training the individual decision trees on different training sets which is know as bagging or bootstrapping aggregation.
# 
# To ensure that a random forest makes accurate predicitions there needs to be access to features that have predicitive power of the label, the trees in the forest also needs to be relatively uncorrelated. So by creating many uncorrelated decision trees by implementing feature randomness and bagging and trusting the average predicition of those trees, decision trees can be extended to a random forest.
# 
# **5b)** Explain what makes reinforcement learning different from supervised learning tasks such as regression or classification. (1.25p)
# 
# The main difference between reinforcement learning (*RL*) and supervised learning (*SL*) lies in their methodology, and indirectly by the data that should be analyzed. While SL methods can only work with data with predefined labels, RL methods can work with data without any predefinitions. The reason for this is that RL used a trial-and-error methodology to explore its environment and to iteratively map out the different paths that could be taken to reach its goal. This trial-and-error method is based on an agent interacting with its environment and for every action, the agent gets a reward or a punishment which teaches the model which paths are superior to others. 
# 
# The types of problems that could be solved with the two methods are a bit different as well. While SL primarily focuses on classifying data to predefined labels or to create regressions of the available data, RL can solve problems that are based on rewards, e.g. finding the shortest path from a start to a goal or to avoid different states in a state-space. The rewards will tell the model to choose some paths in the state-space, i.e. combinations of actions, above others to maximize the total expected reward. This may then converge into an optimal policy which consists of an ordered number of actions. The RL methods can also be used e.g only for exploring and exploiting or for value learning. RL methods can therefore work without supervision and needs no interaction with anything but its own environment. 
# 
# In SL there are several algorithms that could be used based on the available data and the system but in RL, the modeling and decision making is based on Markov's decision process which could sometimes be limiting. Another difference is that SL uses examples in the form of input and output data in order to try finding a general formula for predicting and classifying labels for data points, whereas RL tries to find a sequence of decisions, or actions, where the next decision often depends on the one previously taken. This is where the trade-off between exploration and exploitation could appear for RL methods, but one issue that SL methods do not have is the path dependency that could affect RL methods.


# 
# # References
# Primer/text based on the following references:
# * http://www.cse.chalmers.se/~chrdimi/downloads/book.pdf
# * https://github.com/olethrosdc/ml-society-science/blob/master/notes.pdf


