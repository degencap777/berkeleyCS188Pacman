# multiAgents.py
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
        currentFood = currentGameState.getFood()
        

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        
        # print(newFood.asList())
        # get number of food in same direction
        xp, yp = newPos
        
        
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newGhostDistance = [manhattanDistance(newPos, newGhost.getPosition()) for newGhost in newGhostStates]
        minGhostDistance = min(newGhostDistance)
        currentFoodCount = len(currentFood.asList()) + 1
        numPossibleActions = len(successorGameState.getLegalActions())
        # if minGhostDistance > 3:
            # score = (earnedFood + 1) / maxFoodDistance
        # else:
        newFoodCount = len(newFood.asList())
        if newFoodCount:
            potential_food = [1]
            if action == Directions.LEFT:
              potential_food = [x < xp and y == yp for (x, y) in newFood.asList()]
            elif action == Directions.RIGHT:
              potential_food = [x > xp and y == yp for (x, y) in newFood.asList()]
            elif action == Directions.SOUTH:
              potential_food = [y < yp and x == xp for (x, y) in newFood.asList()]
            elif action == Directions.NORTH:
              potential_food = [y > yp and x == xp for (x, y) in newFood.asList()]
            potentialFoodCount = sum(potential_food) / len(newFood.asList())

            alignedFoodCount = sum([x == xp or y == yp for (x, y) in newFood.asList()]) / len(newFood.asList())
            
            newFoodDistance = [manhattanDistance(newPos, food) for food in newFood.asList()]
            maxFoodDistance = max(newFoodDistance)
            minFoodDistance = min(newFoodDistance)
            earnedFood = currentFoodCount - newFoodCount # > 0 when pacman ate food with action
        
            score = (min(minGhostDistance, 5)) / minFoodDistance
            score += 10 * earnedFood
            if not earnedFood:
                score += potentialFoodCount
            #if action != Directions.STOP:
            #    score += 0.1 * numPossibleActions
            # print(score)
        else:
            score = 1000
        # score = 0
        # for future_action in successorGameState.getLegalActions():
        #     future_state = successorGameState.generatePacmanSuccessor(action)
        #     futureGhostDistance = [manhattanDistance(future_state.getPacmanPosition(), newGhost.getPosition()) for newGhost in newGhostStates]
        #     minFutureGhostDistance = min(futureGhostDistance)
        #     futureFood = len(future_state.getFood().asList()) + 0.001
        #     score += minFutureGhostDistance / futureFood
        # score /= len(successorGameState.getLegalActions())

        return score
        # return successorGameState.getScore()


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
        self.last_agent = gameState.getNumAgents() - 1 # last min agent in ply

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
                if agent == self.last_agent:
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
        util.raiseNotDefined()

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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

