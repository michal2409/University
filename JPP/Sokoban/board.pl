% sokoban\board.pl: board layout for Sokoban puzzle
%     ... learning Prolog with SWI-Prolog
% Author: Javier G. Sogo

/***************************************************************************/
/* Sokoban board: layout and relation rules                                */
/***************************************************************************/
/* The board is built based on adjacent squares using vertical (up|down)   */
/* and horizontal (right|left) directions. This set of rules defines the   */
/* relation between these concepts and the neighbourhoods for any square   */
/***************************************************************************/

/* Adjacent positions given a direction                                    */
:- table neib/3.
neib(P1, P2, up) :- top(P1, P2).
neib(P1, P2, down) :- top(P2, P1).
neib(P1, P2, right) :- right(P1, P2).
neib(P1, P2, left) :- right(P2, P1).

/* A corner is a square which has at most one neighbour square in vertical */
/* and one in horizontal. It is defined using the counter-rule noncorner.  */
:- table corner/1.
corner(X) :- \+ noncorner(X).
noncorner(X) :- top(_,X),top(X,_).
noncorner(X) :- right(_,X),right(X,_).
