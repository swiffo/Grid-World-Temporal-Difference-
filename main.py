import random

##    The Windy Gridworld learning problem is the task of finding an optimal way of going from one point to
##    another on a rectangular grid while being pushed around by wind. We will solve this using the Sarsa
##    algorithm, an on-policy temporal difference learning algorithm.
##
##    The main thing to implement for the learning algorithm is the world itself. That is the
##    "windy grid" which we will call a 'Board'. It consists of a bounded grid and the wind specifications.
##
##    Before getting to that we will implement position and move classes. Arguably these could be implemented
##    simply using tuples, (x,y), saving a lot of code. Implementing them as classes, however, has some benefits:
##        (a) It conceptually separates space and tangent space (They look the same in R^2 but aren't).
##        (b) It makes clear whether an object is a position or a move (both in code and string representation).
##        (c) It allows for a few helpful methods (like adding moves together)
##
##    Against this stands that most of the implementation of these classes would come for free when using tuples.
##    Mainly for reason (a) and (b) we choose to implement them as classes.                                                           

class Position:
    """A grid position (x,y)"""
    
    def __init__(self, x, y ):
        self.coords = (x,y)

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
        
##    In implementing the Move class, we see one important difference between moves and positions
##    that would not be clear or enforceable using tuples: Moves can be added together; positions cannot.

class Move:
    """A move on the grid, e.g., (+3,-2)"""
    def __init__(self, delta_x, delta_y):
        self._vector = (delta_x, delta_y)

    def vector(self):
        """Returns move as tuple, (delta_x, delta_y)"""
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

##    We now move on to defining the world. It consists of a bounded grid and the wind conditions.
##    The wind is implemented as a class with a single method, 'blow', giving a move based on
##    position.

class SouthWind:
    """Rules for wind blowing upwards"""
    # On an irrelevant note, the south wind in Greek mythology is called Notos.

    def __init__(self, wind_speed=dict()):
        self.wind_speed = wind_speed

    def blow(self, position):
        """Returns the move the wind contributes"""
        x = position.coordinates()[0]
        wind_speed = self.wind_speed.get(x, 0)
        return Move(0, wind_speed)

class WestWind:
    """Rules for wind blowing to the right"""
    # The most famous of the Greek anemoi (winds), Zephyrus (or Zephyr). **APPLAUSE**

    def __init__(self, wind_speed=dict()):
        self.wind_speed = wind_speed

    def blow(self, position):
        """Returns the move the wind contributes"""
        y = position.coordinates()[1]
        wind_speed = self.wind_speed.get(y, 0)
        return Move(wind_speed, 0)

##    Finally we implement the board. We restrict to rectangular boards and a such the board/world is
##    determined entirely by the width, height and wind conditions. The purpose of the board is to
##    convert the agent action (an intended move) in a given state (position) into a new state (position).
##
##    Our world has walls around it. Hence the agent cannot be blown off the board but will be stopped
##    by the wall if any move would result in the agent otherwise careening off into empty space.
                                                                                          
class Board:
    """Grid world. Contains all the world rules"""
 
    def __init__(self, width, height, wind):
        self.max_x = width - 1
        self.max_y = height - 1
        self.wind = wind
    
    def move(self, position, agent_move):
        """Move agent from specified position under the rules of the board. Returns new position"""
        self._check_position(position)

        actual_move = agent_move + self.wind.blow(position)

        new_pos = position.move(actual_move)
        new_pos = self._restrict_to_board(new_pos)

        return new_pos
                           
    def dimensions(self):
        """Board dimensions (A,B): 0<=x<=A, 0<=y<=B"""
        return self.max_x, self.max_y

    def _restrict_to_board(self, position):
        """Pulls position on infinite grid to closest position on board"""
        x,y = position.coordinates()
        x = min(self.max_x, max(0, x))
        y = min(self.max_y, max(0, y))
        return Position(x,y)

    def _check_position(self, position):
        """Returns (x,y). Raises exception if position not on board"""
        x,y = position.coordinates()

        if x < 0 or x > self.max_x or y < 0 or y > self.max_y:
            raise InvalidPositionException

##    Lastly we implement some helpful functions.

def max_arg(array):
    """Returns index of highest value. In case of ties, returns lowest index."""
    # I'm sure this function must be in some library somewhere but ...
    max_index = 0
    max_value = array[0]

    for idx, val in enumerate(array):
        if val > max_value:
            max_index = idx
            max_value = val

    return max_index

def print_policy(board, Q, move_symbol_map, goal_position):
    """Prints ASCII representation of the best move for each board position"""
    unknownMove  = "*"
    allowed_moves = list(move_symbol_map.keys())
    max_x, max_y = board.dimensions()

    def moveSymbol(position):
        if position == goal_position:
            return "@"
        
        vals = [Q.get((position, m), -1) for m in allowed_moves]
        
        if max(vals) == -1:
            return unknownMove
        else:
            return move_symbol_map[allowed_moves[max_arg(vals)]]
            
    symbols = [[moveSymbol(Position(x, y)) for x in range(max_x+1)] for y in range(max_y, -1, -1)]
    text = "\n".join([" ".join(line) for line in symbols])
    print(text)

def default_action_value():
    """Initial estimated value of state-action pair"""
    # This is turned into a function to allow for other ways of generating default estimates.
    # E.g., random numbers. The value should be considered relative to the win-reward (100).
    return 1

def generate_strategy(board,
                     start_pos,
                     goal_pos,
                     total_steps=25000, # Number of steps (whether we reach the goal or not) involved in the learning
                     alpha=0.05,       # The learning speed
                     gamma=0.9,        # Time value of future rewards (affects how much we appreciate speedy solutions)
                     epsilon=0.05,     # The epsilon of the epsilon-greedy strategy (how often we explore randomly)
                     move_symbol_map = {Move(-1,0):"L", Move(1,0):"R", Move(0,1):"U", Move(0,-1):"D"}
                     ):
    """Run temporal difference method and print final best actions"""

    # Create the moves the agent can makes as well as their ASCII representations
    allowed_moves  = list(move_symbol_map.keys())

    Q = dict() # The state-action (estimated) values. Entry keys are (pos, move).
    
    print("Starting wild wandering ...")
    pos = start_pos.clone()
    for _step in range(total_steps):
        if random.random() < epsilon:
            # With chance epsilon we do a random exploratory move
            chosen_move = random.choice(allowed_moves) 
        else:
            # Otherwise we choose the estimated optimal move

            # Find the best move
            action_values = [Q.setdefault((pos, m), default_action_value()) for m in allowed_moves]
            chosen_move = allowed_moves[max_arg(action_values)]

            # Estimate the value of the position we arrive at. It's the value of the state-action pair chosen under
            # our epsilon-greedy strategy.
            new_pos = board.move(pos, chosen_move)
            if random.random() < epsilon:
                new_pos_move = random.choice(allowed_moves)
            else:
                new_pos_move = allowed_moves[max_arg([Q.setdefault((new_pos, m), default_action_value()) for m in allowed_moves])]
                
            new_pos_val = gamma * Q.setdefault((new_pos, new_pos_move), default_action_value())

            # Adjust the value of the (current position, chosen_move) pair.
            value_increment = alpha * (new_pos_val - Q[(pos, chosen_move)])
            if new_pos == goal_pos:
                value_increment += alpha * 100 # If we have arrived at the goal, that's worth 100!!
            Q[(pos, chosen_move)] += value_increment

            # If we arrived at the goal, end the episode and start over.
            if new_pos == goal_pos:
                pos = start_pos
            else:
                pos = new_pos

    print("Exhausted after {} steps... stopping.\n".format(total_steps))
    print_policy(board, Q, move_symbol_map, goal_pos)


##    Some examples of the learning algorithm in action:

def example1():
    """Example of 10x7 board with a wind from the south"""
    wind_speeds = {3:1, 4:1, 5:1, 6:2, 7:2, 8:1} 
    wind = SouthWind(wind_speeds)
    board = Board(10, 7, wind)
    start_pos = Position(1, 3)
    goal_pos = Position(7,3)

    generate_strategy(board, start_pos, goal_pos)

def example2():
    """Example of an 'express-way' wind. Strong wind to the right (east) on a single y-level"""
    wind  = WestWind({1:4})
    board = Board(12, 7, wind)

    starting_position = Position(1,3)
    goal_position     = Position(10, 3)

    generate_strategy(board, starting_position, goal_position, total_steps=50000)

def example3():
    """Same as example1 but with diagonal moves available too."""
    wind_speeds = {3:1, 4:1, 5:1, 6:2, 7:2, 8:1} 
    wind = SouthWind(wind_speeds)
    board = Board(10, 7, wind)
    start_pos = Position(1, 3)
    goal_pos = Position(7,3)
    moves = {
        Move(1,0):"R",
        Move(-1,0):"L",
        Move(0,1):"U",
        Move(0,-1):"D",
        Move(1,1):"3",
        Move(1,-1):"9",
        Move(-1,1):"1",
        Move(-1,-1):"7"
        }

    generate_strategy(board, start_pos, goal_pos, total_steps=50000, move_symbol_map=moves)
    
    

    

    
