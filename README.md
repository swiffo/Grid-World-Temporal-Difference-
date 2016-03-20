# Grid World Temporal Difference Learning

The *Windy Gridworld* learning problem is the task of finding an optimal way of going from one point to
another on a rectangular grid while being pushed around by wind. We will solve this using the Sarsa
algorithm, an on-policy temporal difference learning algorithm.

The problem is taken from the book *Reinforcement Learning* by Richard S. Sutton and Andrew G. Barto (Example 6.5, exercises 6.6, 6.7).

> On a rectangular grid, the Grid World, an agent, starting at a given point, has for each discrete time-step a number of moves available to him (one step right, left, up or down). At the same time a wind blows cold, pushing him around the grid depending on his position. **The agent must find a way to get to a given node in the grid as fast as possible.** 

> The grid is bounded by an invisible wall. Any movement that would take the agent outside the grid instead places him at the edge. The wind contribution to the agent's move is based on his starting position in each time-step.



## Example
As an example consider a 10-by-7 grid, with start at (1,3) and goal at (7,3) (lower-left coordinate is (0,0)). Let there be a wind that blows north (upwards), moving the agent involuntarily upwards by one step when his x-coordinate is 3,4,5 or 8 and up by two steps when his x-coordinate is 6 or 7. With 25,000 steps the algorithm learns the following (randomizer not seeded):

```
U U L D R R R R R D
L U R D R R R D D D
R U R D R R R D L D
R R R R R R D @ D D
L R R R R U * * L L
R D D R U * * * D D
D D R D * * * * * D
```

U = Up, D = Down, L = Left, R = Right, @ = Goal, * = unreachable

In the code `example1()` learns and prints this strategy.
