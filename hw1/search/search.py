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
    return [s, s, w, s, w, w, s, w]


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
    explored = set()
    parent = {}
    action = {}
    frontier = util.Stack()

    def get_solution(child):
        solution = []
        while child != problem.getStartState():
            solution.insert(0, action[child])
            child = parent[child]
        return solution

    start_state = problem.getStartState() # start state
    frontier.push(start_state)
    while not frontier.isEmpty():
        current_state = frontier.pop() # state to explore
        explored.add(current_state)  # add states to explored states
        if problem.isGoalState(current_state):
            return get_solution(current_state)
        children = problem.getSuccessors(current_state)
        for state, direction, _ in children:
            if state not in explored:
                frontier.push(state)
                action[state] = direction
                parent[state] = current_state
    return []
    # util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    explored = set()
    parent = {}
    action = {}
    frontier = util.Queue()

    def get_solution(child):
        solution = []
        while child != problem.getStartState():
            solution.insert(0, action[child])
            child = parent[child]
        return solution

    start_state = problem.getStartState()  # state of start state
    frontier.push(start_state) # push start state to frontier
    while not frontier.isEmpty():
        current_state = frontier.pop()  # state to explore
        # print(current_state)
        explored.add(current_state)  # add states to explored states
        if problem.isGoalState(current_state):
            return get_solution(current_state)
        children = problem.getSuccessors(current_state)
        for state, direction, _ in children:
            if state not in explored:
                frontier.push(state)  # ['A', 'B', 'C'] expanded states
                action[state] = direction
                parent[state] = current_state
                # TODO: why? is it right to consider a node explored
                #  though it's not actually expanded ?
                #  otherwise how can we solve the many paths problem?
                explored.add(state)
    return []
    # util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    explored = set()
    parent = {}
    action = {}
    cost = {}
    frontier = util.PriorityQueue()

    def get_solution(node):
        solution = []
        while node != problem.getStartState():
            solution.insert(0, action[node])
            node = parent[node]
        return solution

    start_state = problem.getStartState()  # start state
    cost[start_state] = 0
    frontier.update(start_state, cost[start_state])
    while not frontier.isEmpty():
        current_state = frontier.pop()  # state to explore
        explored.add(current_state)  # add state to explored states
        if problem.isGoalState(current_state):
            return get_solution(current_state)
        children = problem.getSuccessors(current_state)
        for state, direction, step_cost in children:
            if state not in explored:
                if state in cost:
                    old_cost = cost[state]
                    new_cost = cost[current_state] + step_cost
                    if old_cost < new_cost:
                        continue
                action[state] = direction
                parent[state] = current_state
                cost[state] = cost[current_state] + step_cost
                frontier.update(state, cost[state])
                # TODO: w/o adding to explored, failed manypaths but passed goalAtDequeue
                #  adding to explored passes manypaths but fails goalAtDequeue
    return []
    # util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    explored = set()
    parent = {}
    action = {}
    backward_cost = {}
    forward_cost = {}
    frontier = util.PriorityQueue()

    def get_solution(node):
        solution = []
        while node != problem.getStartState():
            solution.insert(0, action[node])
            node = parent[node]
        return solution

    start_state = problem.getStartState()  # start state
    backward_cost[start_state] = 0
    forward_cost[start_state] = heuristic(start_state, problem)
    frontier.update(start_state, backward_cost[start_state] + forward_cost[start_state])
    while not frontier.isEmpty():
        current_state = frontier.pop()  # state to explore
        explored.add(current_state)  # add state to explored states
        if problem.isGoalState(current_state):
            return get_solution(current_state)
        children = problem.getSuccessors(current_state)
        for state, direction, step_cost in children:
            if state not in explored:
                if state in backward_cost:
                    old_cost = backward_cost[state] + heuristic(state, problem)
                    new_cost = backward_cost[current_state] + step_cost + heuristic(state, problem)
                    if old_cost < new_cost:
                        continue
                backward_cost[state] = backward_cost[current_state] + step_cost
                forward_cost[state] = heuristic(state, problem)
                action[state] = direction
                parent[state] = current_state
                frontier.update(state, backward_cost[state] + forward_cost[state])
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
