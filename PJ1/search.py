# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    from util import Stack
    dfs_stack=Stack()#DFS needs to use stacks
    record=[]#Used to record passing nodes
    dfs_stack.push((problem.getStartState(),[]))#Import initial state
    while not dfs_stack.isEmpty():#When the stack is not empty, expand the search
        node,direction=dfs_stack.pop()#Get current node
        if problem.isGoalState(node):
            return direction#Execute the steps after the target node is found
        if node not in record:#This node never goes through
            next_step=problem.getSuccessors(node)#Follow up
            record.append(node)#Record current node
            for successor,action,stepCost in next_step:
                if successor not in record:
                    dfs_stack.push((successor,direction+[action]))#Into the stack
    return direction+[action]
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    bfs_queue=Queue()#BFSneeds to use queue
    record=[]#Used to record passing nodes
    bfs_queue.push((problem.getStartState(),[]))#Import initial state
    while not bfs_queue.isEmpty():#When the queue is not empty, expand the search
        node,direction=bfs_queue.pop()#Get current node
        if problem.isGoalState(node):
            return direction#Execute the steps after the target node is found
        if node not in record:#This node never goes through
            next_step=problem.getSuccessors(node)#Follow up
            record.append(node)#Record current node
            for successor,action,stepCost in next_step:
                if problem.isGoalState(successor):
                        return direction+[action]
                elif successor not in record:
                    bfs_queue.push((successor,direction+[action]))#Into the queue
    return direction+[action]
    util.raiseNotDefined()
 

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    ucs_queue=PriorityQueue()#UCS need to use priortiy queue
    record=[]#Used to record passing nodes
    g_n=0
    ucs_queue.push((problem.getStartState(),[]),g_n)#Import initial state
    while not ucs_queue.isEmpty():#When the queue is not empty, expand the search
        node,direction=ucs_queue.pop()#Get current node
        if problem.isGoalState(node):
            return direction#Execute the steps after the target node is found
        if node not in record:#This node never goes through
            next_step=problem.getSuccessors(node)#Follow up
            record.append(node)#Record current node
            for successor,action,stepCost in next_step:
                if successor not in record:
                    g_n=problem.getCostOfActions(direction+[action])
                    ucs_queue.push((successor,direction+[action]),g_n)#Into the queue                
    return direction+[action]
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    astar_queue=PriorityQueue()#astar need to use priortiy queue
    record=[]#Used to record passing nodes
    g_n=0
    h_n=heuristic(problem.getStartState(),problem)#And as we talked about in the lecture, calculate the ideal distance
    f_n=g_n+h_n
    astar_queue.push((problem.getStartState(),[]),f_n)#Import initial state
    while not astar_queue.isEmpty():#When the queue is not empty, expand the search
        node,direction=astar_queue.pop()#Get current node
        if problem.isGoalState(node):
            return direction#Execute the steps after the target node is found
        if node not in record:#This node never goes through
            next_step=problem.getSuccessors(node)#Get current node
            record.append(node)#Record current node
            for successor,action,stepCost in next_step:
                if successor not in record:
                    #g_n, h_n, f_n just like the ppt
                    g_n=problem.getCostOfActions(direction+[action])
                    h_n=heuristic(successor,problem)
                    f_n=g_n+h_n
                    astar_queue.push((successor,direction+[action]),f_n)#Into the queue                   
    return direction+[action]
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
