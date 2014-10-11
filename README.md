chess-tournament-pairing
========================

/* this C++ code finds the best pairings for one section and one round according to USCF rules
 * the greedy algorithm considers all possible swaps of any two players and keeps the best single swap (if better) for each iteration
 * multiple swaps per iteration are considered after no improvements can be made with a single swap, but only if multiple swaps are requested
 * the initial position is one possible pairing, a hint that might yield a better local minimum in this global minimization problem
 * the cost function to be minimized is determined by the USCF Swiss System rules and their priorities as given in the USCF rule book/updates
 */
