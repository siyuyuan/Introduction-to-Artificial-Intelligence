import util, math, random
from collections import defaultdict
from util import ValueIteration

############################################################
# Problem 2a

# If you decide 2a is true, prove it in blackjack.pdf and put "return None" for
# the code blocks below.  If you decide that 2a is false, construct a counterexample.
class CounterexampleMDP(util.MDP):
    # Return a value of any type capturing the start state of the MDP.
    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 0
        #raise Exception("Not implemented yet")
        # END_YOUR_CODE

    # Return a list of strings representing actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return [-1,1] if state == 0 else [state]
        #raise Exception("Not implemented yet")
        # END_YOUR_CODE

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # Remember that if |state| is an end state, you should return an empty list [].
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return [(-1,0.1,10),(1,0.9,1)] if state == 0 else [(state,1,0)]
        #raise Exception("Not implemented yet")
        # END_YOUR_CODE

    # Set the discount factor (float or integer) for your counterexample MDP.
    def discount(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 1
        #raise Exception("Not implemented yet")
        # END_YOUR_CODE

############################################################
# Problem 3a

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look closely at this function to see an example of state representation for our Blackjack game.
    # Each state is a tuple with 3 elements:
    #   -- The first element of the tuple is the sum of the cards in the player's hand.
    #   -- If the player's last action was to peek, the second element is the index
    #      (not the face value) of the next card that will be drawn; otherwise, the
    #      second element is None.
    #   -- The third element is a tuple giving counts for each of the cards remaining
    #      in the deck, or None if the deck is empty or the game is over (e.g. when
    #      the user quits or goes bust).
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 38 lines of code, but don't worry if you deviate from this)
        total_value, peek_index, deck = state
        if deck is None:#the game is ended
            return []        
        if action == 'Take':
            states = []
            if peek_index:
                total_value = total_value + self.cardValues[peek_index]
                if total_value > self.threshold: #busting
                    return [((total_value, None, None), 1, 0)]
                else:
                    #tuple to list
                    deck_ls = list(deck)
                    deck_ls[peek_index] = deck_ls[peek_index] - 1
                    if sum(deck_ls) == 0:
                        #the game is over
                        return [((total_value, None, None), 1, total_value)]
                    else:
                        #the game continues
                        deck_ls = tuple(deck_ls)
                        return [((total_value, None, deck_ls), 1, 0)]
            else:
                '''
                we cannot directly update total_value because it has randomness
                We need to iterate through each value
                '''
                for i in range(len(deck)):
                    if deck[i]:
                        take_prob = float(deck[i]) / sum(deck)
                        total_value_hat = total_value + self.cardValues[i] 
                        if total_value_hat > self.threshold: #busting
                            states.append(((total_value_hat, None, None), take_prob, 0))
                        else:
                            deck_ls = list(deck)
                            deck_ls[i] = deck_ls[i] - 1
                            if sum(deck_ls) == 0:
                                #the game is over
                                states.append(((total_value_hat, None, None), take_prob, total_value_hat))
                            else:
                                #the game continues
                                deck_ls = tuple(deck_ls)
                                states.append(((total_value_hat, None, deck_ls), take_prob, 0))
                return states
        elif action == 'Peek':
            states = []
            if peek_index:#twice
                return []
            for i in range(len(deck)):
                if deck[i]:
                    take_prob = float(deck[i]) / sum(deck)
                    cost = self.peekCost
                    states.append(((total_value, i, deck), take_prob, -cost))
            return states
                        
        elif action == 'Quit':
             return [((total_value, None, None), 1, total_value)]
        # raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 3b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the
    optimal action at least 10% of the time.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    # raise Exception("Not implemented yet")
    return  BlackjackMDP(cardValues = [1,2,3,4,5,1000], multiplicity = 2, threshold = 20, peekCost = 1)
    # END_YOUR_CODE

############################################################
# Problem 4a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
        alpha = self.getStepSize()
        predict_Q = self.getQ(state,action)
        Q_max = 0
        if newState is not None:
            for next_action in self.actions(newState):
                if self.getQ(newState, next_action) > Q_max:
                    Q_max = self.getQ(newState, next_action)
        
        score = reward +  Q_max * self.discount
        temp = alpha * (score - predict_Q)
        for feature_name, feature_value in self.featureExtractor(state, action):
            self.weights[feature_name] =self.weights[feature_name] + temp * feature_value
        #  raise Exception("Not implemented yet")
        # END_YOUR_CODE

# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 4b: convergence of Q-learning
# Small test case
smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)

def simulate_QL_over_MDP(MDP, featureExtractor):
    # NOTE: adding more code to this function is totally optional, but it will probably be useful
    # to you as you work to answer question 4b (a written question on this assignment).  We suggest
    # that you add a few lines of code here to run value iteration, simulate Q-learning on the MDP,
    # and then print some stats comparing the policies learned by these two approaches.
    # BEGIN_YOUR_CODE
    # pass
    RL = QLearningAlgorithm(MDP.actions, MDP.discount(), featureExtractor, explorationProb=0)
    util.simulate(MDP, RL, numTrials=30000, maxIterations=1000, verbose=False, sort=False)
    MDP.computeStates()
    RL_policy = {}
    for state in MDP.states:
        RL_policy[state] = RL.getAction(state)
    val = util.ValueIteration()
    val.solve(MDP)
    val_policy = val.pi
    sum_ = []
    for key in RL_policy:
        if RL_policy[key] == val_policy[key]:
            sum_.append(1)
        else:
            sum_.append(0)
    print(float(sum(sum_))/len(RL_policy))
    return RL_policy, val_policy
    
smallMDP_RL_policy, smallMDP_val_policy = simulate_QL_over_MDP(smallMDP, identityFeatureExtractor)
largeMDP_RL_policy, largeMDP_val_policy = simulate_QL_over_MDP(largeMDP, identityFeatureExtractor)
    # END_YOUR_CODE

############################################################
# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs.
# (See identityFeatureExtractor() above for a simple example.)
# Include the following features in the list you return:
# -- Indicator for the action and the current total (1 feature).
# -- Indicator for the action and the presence/absence of each face value in the deck.
#       Example: if the deck is (3, 4, 0, 2), then your indicator on the presence of each card is (1, 1, 0, 1)
#       Note: only add this feature if the deck is not None.
# -- Indicators for the action and the number of cards remaining with each face value (len(counts) features).
#       Note: only add these features if the deck is not None.
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state

    # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
    features = []
    featureKey = (total, action)
    featureValue = 1
    features.append((featureKey, featureValue))
    if counts is not None:
        temp = []
        for ele in counts:
            if ele != 0:
                temp.append(1)
            elif ele == 0:
                temp.append(0)
        featureKey = (tuple(temp), action)
        featureValue = 1
        features.append((featureKey, featureValue))
    if counts is not None:
         for i in range(len(counts)):
             features.append(((i, counts[i], action), 1))
    return features
        
    # raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 4d: What happens when the MDP changes underneath you?!

# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)

def compare_changed_MDP(original_mdp, modified_mdp, featureExtractor):
    # NOTE: as in 4b above, adding more code to this function is completely optional, but we've added
    # this partial function here to help you figure out the answer to 4d (a written question).
    # Consider adding some code here to simulate two different policies over the modified MDP
    # and compare the rewards generated by each.
    # BEGIN_YOUR_CODE
    
    val = util.ValueIteration()
    val.solve(original_mdp)
    val_policy = val.pi
    RL1 = util.FixedRLAlgorithm(val_policy)
    result1 = util.simulate(modified_mdp, RL1, numTrials=50000, maxIterations=1000, verbose=False, sort=False)
    avg1 = sum(result1)/float(len(result1))
    print(avg1)
    RL2 = QLearningAlgorithm(modified_mdp.actions, modified_mdp.discount(), featureExtractor, explorationProb=0.2)
    result2 = util.simulate(modified_mdp, RL2, numTrials=50000, maxIterations=1000, verbose=False, sort=False)
    avg2 = sum(result2)/float(len(result2))
    print(avg2)
    # pass
    # END_YOUR_CODE

compare_changed_MDP(originalMDP, newThresholdMDP, blackjackFeatureExtractor)