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

    def get_delta_x(self):
        """Move along x-axis"""
        return self._vector[0]

    def get_delta_y(self):
        """Move along y-axis"""
        return self._vector[1]

    def vector(self):
        return self._vector

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
    
    def move(self, position, movement):
        """Move agent from specified position under the rules of the board. Returns new position"""
        
        oldX, oldY = self._checkPosition(position)

        newX = oldX + movement.get_delta_x()
        newY = oldY + movement.get_delta_y()

        newX = min(self.max_x, max(0, newX))
        newY = min(self.max_y, max(0, newY))

        windMove = self.wind.blow(Position(newX, newY))
        wind_x, wind_y = windMove.vector()

        newX = min(self.max_x, max(0, newX + wind_x))
        newY = min(self.max_y, max(0, newY + wind_y))
                   
        return Position(newX, newY)
                           
    def dimensions(self):
        """Board dimensions (A,B): 0<=x<=A, 0<=y<=B"""
        return self.max_x, self.max_y

    def _checkPosition(self, position):
        """Returns (x,y). Raises exception if position not on board"""
        x,y = position.coordinates()

        if x < 0 or x > self.max_x or y < 0 or y > self.max_y:
            raise InvalidPositionException

        return x,y

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
    return 1



def generateStrategy(board, startPos, goalPos):
    allowedMoves = [Move(-1,0), Move(1,0), Move(0,1), Move(0,-1)]
    moveSymbolMap = {Move(-1,0):"L", Move(1,0):"R", Move(0,1):"U", Move(0,-1):"D"}
    gamma = 0.9
    epsilon = 0.05
    alpha = 0.05
    totalSteps = 10000

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

    print("Exhausted ... stopping.\n")

    printPolicy(board, Q, moveSymbolMap, goalPos)


def example1():
    windSpeeds = {3:1, 4:1, 5:1, 6:2, 7:2, 8:1} # Increment y by this amount when movement ends with x on the key-value
    wind = SouthWind(windSpeeds)
    board = Board(10, 7, wind)
    startPos = Position(0, 3)
    goalPos = Position(8,3)

    generateStrategy(board, startPos, goalPos)
    

    

    
