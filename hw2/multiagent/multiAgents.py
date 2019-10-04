# multiAgents.py
# solutions by Jimin Sun
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        self.walls = gameState.getWalls().asList()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        "*** YOUR CODE HERE ***"        
        currentFood = currentGameState.getFood()
        currentGhostStates = currentGameState.getGhostStates()
        
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()

        newGhostStates = successorGameState.getGhostStates()
        currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
        newGhostDistance = [manhattanDistance(newPos, newGhost.getPosition()) for newGhost in newGhostStates]
        minGhostDistance = min(newGhostDistance)

        def count_walls_between(pos, food):
            x, y = pos
            fx, fy = food
            return sum([wx in range(min(x, fx), max(x, fx)+1) and
                        wy in range(min(y, fy), max(y, fy)+1) for (wx, wy) in self.walls])

        if len(currentFood.asList()):  
            foodDistance = [manhattanDistance(newPos, food) + 2.0*count_walls_between(newPos, food) 
                            for food in currentFood.asList()]
            minFoodDistance = min(foodDistance)
            if min(currentScaredTimes) > 1:
                score = min(minGhostDistance, 3) / (minFoodDistance + 0.01)
            else:
                score = min(minGhostDistance, 7) / (minFoodDistance + 0.01)
        else:
            score = 100

        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        utility = util.Counter()
        last_agent = gameState.getNumAgents() - 1 # last min agent in ply

        def max_value(state, agent, depth):
            value = -1e5 # initial utility of max node
            if depth == self.depth: 
                # reached max depth -> evaluate utility of state
                return self.evaluationFunction(state)
            if len(state.getLegalActions(agent)):
                # max <- min agent (idx: agent+1) of same layer
                for action in state.getLegalActions(agent):
                    min_v = min_value(state.generateSuccessor(agent, action), agent+1, depth)
                    value = max(value, min_v)
                return value
            else:
                # when pacman dies even before exploring to max depth
                return self.evaluationFunction(state)

        def min_value(state, agent, depth):
            value = 1e5 # initial utility of min node
            if state.getLegalActions(agent):
                if agent == last_agent:
                    # min <- max agent (idx: 0) of next layer
                    for action in state.getLegalActions(agent):
                        max_v = max_value(state.generateSuccessor(agent, action), 0, depth+1)
                        value = min(value, max_v)
                else:
                    # min <- min agent (idx: agent+1) of same layer
                    for action in state.getLegalActions(agent):
                        min_v = min_value(state.generateSuccessor(agent, action), agent+1, depth)
                        value = min(value, min_v)
                return value
            else:
                # when pacman dies even before exploring to max depth
                return self.evaluationFunction(state)

        for action in gameState.getLegalActions(0):
            utility[action] = min_value(gameState.generateSuccessor(0, action), 1, 0)
        return utility.argMax()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        
        utility = util.Counter()
        last_agent = gameState.getNumAgents() - 1 # last min agent in ply
        alpha = float('-inf')
        beta = float('inf')

        def max_value(state, agent, depth, alpha, beta):
            value = float('-inf') # initial utility of max node
            if depth == self.depth: 
                # reached max depth -> evaluate utility of state
                return self.evaluationFunction(state)
            if len(state.getLegalActions(agent)):
                # max <- min agent (idx: agent+1) of same layer
                for action in state.getLegalActions(agent):
                    min_v = min_value(state.generateSuccessor(agent, action), agent+1, depth, alpha, beta)
                    value = max(value, min_v)
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            else:
                # when pacman dies even before exploring to max depth
                return self.evaluationFunction(state)

        def min_value(state, agent, depth, alpha, beta):
            value = float('inf') # initial utility of min node
            if state.getLegalActions(agent):
                if agent == last_agent:
                    # min <- max agent (idx: 0) of next layer
                    for action in state.getLegalActions(agent):
                        max_v = max_value(state.generateSuccessor(agent, action), 0, depth+1, alpha, beta)
                        value = min(value, max_v)
                        if value < alpha:
                            return value
                        beta = min(beta, value)
                else:
                    # min <- min agent (idx: agent+1) of same layer
                    for action in state.getLegalActions(agent):
                        min_v = min_value(state.generateSuccessor(agent, action), agent+1, depth, alpha, beta)
                        value = min(value, min_v)
                        if value < alpha:
                            return value
                        beta = min(beta, value)
                return value
            else:
                # when pacman dies even before exploring to max depth
                return self.evaluationFunction(state)
        
        value = float('-inf')
        for action in gameState.getLegalActions(0):
            value = min_value(gameState.generateSuccessor(0, action), 1, 0, alpha, beta)
            utility[action] = value
            alpha = max(alpha, value)

        return utility.argMax()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        utility = util.Counter()
        last_agent = gameState.getNumAgents() - 1 # last min agent in ply
        
        def exp_value(state, agent, depth):
            value = 0
            if state.getLegalActions(agent):
                prob = 1.0 / len(state.getLegalActions(agent))
                if agent == last_agent:
                    # exp <- max agent (idx: 0) of next layer
                    for action in state.getLegalActions(agent):
                        max_v = max_value(state.generateSuccessor(agent, action), 0, depth+1)
                        value += prob * max_v # expectation
                else:
                    # exp <- exp agent (idx: agent+1) of same layer
                    for action in state.getLegalActions(agent):
                        exp_v = exp_value(state.generateSuccessor(agent, action), agent+1, depth)
                        value += prob * exp_v # expectation
                return value
            else:
                # when pacman dies even before exploring to max depth
                return self.evaluationFunction(state)

        def max_value(state, agent, depth):
            value = -1e5 # initial utility of max node
            if depth == self.depth: 
                # reached max depth -> evaluate utility of state
                return self.evaluationFunction(state)
            if len(state.getLegalActions(agent)):
                # max <- min agent (idx: agent+1) of same layer
                for action in state.getLegalActions(agent):
                    exp_v = exp_value(state.generateSuccessor(agent, action), agent+1, depth)
                    value = max(value, exp_v)
                return value
            else:
                # when pacman dies even before exploring to max depth
                return self.evaluationFunction(state)

        for action in gameState.getLegalActions(0):
            utility[action] = exp_value(gameState.generateSuccessor(0, action), 1, 0)
        return utility.argMax()


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood().asList()
    currentGhostStates = currentGameState.getGhostStates()
    currentCapsule = currentGameState.getCapsules()
    currentScore = currentGameState.getScore()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
    walls = currentGameState.getWalls().asList()

    def count_walls_between(pos, food):
        x, y = pos
        fx, fy = food
        fx, fy = int(fx), int(fy)
        
        return sum([wx in range(min(x, fx), max(x, fx)+1) and
                    wy in range(min(y, fy), max(y, fy)+1) for (wx, wy) in walls])

    foodDistance = [manhattanDistance(currentPos, food) for food in currentFood]
    
    if currentGameState.isWin():
        score = 10000
    else:
        closestFood = sorted(foodDistance)
        closeFoodDistance = sum(closestFood[-5:])
        closestFoodDistance = sum(closestFood[-3:])
        ghostDistance = [manhattanDistance(currentPos, ghost.getPosition()) + 2* count_walls_between(currentPos, ghost.getPosition()) for ghost in currentGhostStates]
        minGhostDistance = min(min(ghostDistance), 6)
        
        score = currentScore + 0.5 * currentScaredTimes[0] + 1.0 / len(currentFood) - len(currentCapsule) + \
                minGhostDistance + 2.0 / closeFoodDistance + 2.0 / closestFoodDistance
    return score

# Abbreviation
better = betterEvaluationFunction

