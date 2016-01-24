import random

class Position:
    """A grid position (x,y)"""
    
    def __init__(self, x, y ):
        self.coords = (x,y)

    def get_x(self):
        """x-coordinate"""
        return self.coords[0]

    def get_y(self):
        """y-coordinate"""
        return self.coords[1]

    def coordinates(self):
        """Returns position as tuple, (x,y)."""
        return self.coords

    def move(self, move):
        """Returns new position after specified move on an infinite grid"""
        x, y = self.coords
        dx, dy = move.vector()
        return Position(x+dx, y+dy)

    def clone(self):
        """Copy self"""
        return Position(self.coords[0], self.coords[1])

    def __str__(self):
        return "P({:2},{:2})".format(self.coords[0], self.coords[1])

    def __repr__(self):
        return "P({},{})".format(self.coords[0], self.coords[1])

    def __eq__(self, other):
        return self.coords == other.coords

    def __hash__(self):
        return hash(self.coords)
        

class Move:
    """A move on the grid, e.g., (+3,-2)"""
    def __init__(self, delta_x, delta_y):
        self._vector = (delta_x, delta_y)

    def vector(self):
        return self._vector

    def __add__(self, other):
        selfx, selfy = self.vector()
        otherx, othery = other.vector()
        return Move(selfx+otherx, selfy+othery)
        
    def __str__(self):
        return "m({:2},{:2})".format(self._vector[0], self._vector[1])

    def __repr__(self):
        return "m({},{})".format(self._vector[0], self._vector[1])

    def __eq__(self, other):
        return self._vector == other._vector
    
    def __hash__(self):
        return hash(self._vector)


class InvalidPositionException(BaseException):
    """Exception for referring to a position not on the board"""
    pass

class SouthWind:
    """Rules for wind blowing upwards"""

    def __init__(self, windSpeed=dict()):
        self.windSpeed = windSpeed

    def blow(self, position):
        """Returns the move the wind makes on agent"""
        x = position.coordinates()[0]
        windSpeed = self.windSpeed.get(x, 0)
        return Move(0, windSpeed)

class Board:
    """Grid world. Contains all the world rules"""
 
    def __init__(self, width, height, wind):
        self.max_x = width - 1
        self.max_y = height - 1
        self.wind = wind
    
    def move(self, position, agentMove):
        """Move agent from specified position under the rules of the board. Returns new position"""
        self._checkPosition(position)

        actualMove = agentMove + self.wind.blow(position)

        newPos = position.move(actualMove)
        newPos = self._restrictToBoard(newPos)

        return newPos
                           
    def dimensions(self):
        """Board dimensions (A,B): 0<=x<=A, 0<=y<=B"""
        return self.max_x, self.max_y

    def _restrictToBoard(self, position):
        x,y = position.coordinates()
        x = min(self.max_x, max(0, x))
        y = min(self.max_y, max(0, y))
        return Position(x,y)

    def _checkPosition(self, position):
        """Returns (x,y). Raises exception if position not on board"""
        x,y = position.coordinates()

        if x < 0 or x > self.max_x or y < 0 or y > self.max_y:
            raise InvalidPositionException

def maxArg(array):
    """Returns index of highest value. In case of ties, returns lowest index."""
    max_index = 0
    max_value = array[0]

    for idx, val in enumerate(array):
        if val > max_value:
            max_index = idx
            max_value = val

    return max_index

def printPolicy(board, Q, moveSymbolMap, goalPosition):
    """Prints ASCII representation of the best move for each board position"""
    unknownMove = "*"
    allowedMoves = list(moveSymbolMap.keys())
    max_x, max_y = board.dimensions()

    def moveSymbol(position):
        if position == goalPosition:
            return "@"
        
        vals = [Q.get((position, m), -1) for m in allowedMoves]
        
        if max(vals) == -1:
            return unknownMove
        else:
            return moveSymbolMap[allowedMoves[maxArg(vals)]]
            
    symbols = [[moveSymbol(Position(x,y)) for x in range(max_x+1)] for y in range(max_y, -1, -1)]
    text = "\n".join([" ".join(line) for line in symbols])
    print(text)

def defaultActionValue():
    """Initial estimated value of state-action pair"""
    return 1

def generateStrategy(board,
                     startPos,
                     goalPos,
                     totalSteps=25000,
                     alpha=0.05,
                     gamma=0.9,
                     epsilon=0.05):
    """Run temporal difference method and print final best actions"""
    allowedMoves = [Move(-1,0), Move(1,0), Move(0,1), Move(0,-1)]
    moveSymbolMap = {Move(-1,0):"L", Move(1,0):"R", Move(0,1):"U", Move(0,-1):"D"}

    Q = dict()
    pos = startPos.clone()
    print("Starting wild wandering ...")
    for step in range(totalSteps):
        if random.random() < epsilon:
            chosenMove = random.choice(allowedMoves)
        else:
            actionVals = [Q.setdefault((pos, m), defaultActionValue()) for m in allowedMoves]
            chosenMove = allowedMoves[maxArg(actionVals)]
            newPos = board.move(pos, chosenMove)
            if random.random() < epsilon:
                newPosMove = random.choice(allowedMoves)
            else:
                newPosMove = allowedMoves[maxArg([Q.setdefault((newPos, m), defaultActionValue()) for m in allowedMoves])]
                
            newPosVal = gamma * Q.setdefault((newPos, newPosMove), defaultActionValue())

            valueIncrement = alpha * (newPosVal - Q[(pos,chosenMove)])
            if newPos == goalPos:
                valueIncrement += alpha * 100
            Q[(pos,chosenMove)] += valueIncrement

            if newPos == goalPos:
                pos = startPos
            else:
                pos = newPos

    print("Exhausted after {} steps... stopping.\n".format(totalSteps))

    printPolicy(board, Q, moveSymbolMap, goalPos)


def example1():
    """Example of 10x7 board with a wind from the south"""
    windSpeeds = {3:1, 4:1, 5:1, 6:2, 7:2, 8:1} # Increment y by this amount when movement ends with x on the key-value
    wind = SouthWind(windSpeeds)
    board = Board(10, 7, wind)
    startPos = Position(1, 3)
    goalPos = Position(7,3)

    generateStrategy(board, startPos, goalPos)
    

    

    
