
% sokoban\game.pl: game rules for the Sokoban puzzle
%     ... learning Prolog with SWI-Prolog
% Author: Javier G. Sogo

/***************************************************************************/
/* The problem: Sokoban puzzle                                             */
/***************************************************************************/
/* Sokoban is a puzzle created in 1981 by Hiroyuki Imabayashi and          */
/* published in 1982 by Thinking Rabbit (Japan). It is a transport puzzle  */
/* plyaed on a board of squares where the player has to push boxes in a    */
/* warehouse from its initial position to the storage locations.           */
/*                                                                         */
/* The player can only move horizontally or vertically in the board, so    */
/* movements are constrained by the boxes and the walls. The sokoban can't */
/* go through walls or boxes.                                              */
/*                                                                         */
/* The Sokoban can only push (not pull) boxes through the warehouse.       */
/* The puzzle is solved when all boxes are at the storage positions.       */
/***************************************************************************/

/***************************************************************************/
/* To solve the puzzle we need to represent state and movements, we do it  */
/* as follows:                                                             */
/*                                                                         */
/*  - Representation of state:                                             */
/*      state(WhereTheSokobanIs, PositionOfBoxes)                          */
/*                                                                         */
/*  - Representation of the moves:                                         */
/*      move(BoxPosition, PusDirection)                                    */
/***************************************************************************/

/***************************************************************************/
/* Board constraints and definitions are implemented in file board.pl      */
/***************************************************************************/
:-include(board).


/***************************************************************************/
/* Game rules: solution or final state                                     */
/***************************************************************************/

/* The puzzle is solved when all boxes are in a solution position, this is */
/* the final_state.                                                        */
final_state(state(_Sokoban, Boxes, _)) :-
    all_boxes_in_solution(Boxes), !.

/* All boxes are in a the solution position when each one of them is at a  */
/* solution position.                                                      */
all_boxes_in_solution([]).
all_boxes_in_solution([Box|Boxes]) :-
    solution(Box),
    all_boxes_in_solution(Boxes).


/***************************************************************************/
/* Game rules: positions to avoid                                          */
/***************************************************************************/

/* Some position for boxes must be avoided (unless they are solutions)     */
/* because the Sokoban won't be able to move them further:                 */
/*  - corners: the Sokoban can't move a box which is placed at a corner.   */
:- table stuck/1.
stuck(X) :-
    \+ solution(X),
    corner(X).

/*  - horizontally adjacent boxes can only be moved up or down... if there */
/*    are empty squares for the Sokoban to push and for the boxes to move. */
:- table stuck/2.
stuck(X, Y) :-
    (right(X,Y); right(Y,X)),
    (\+ solution(X); \+ solution(Y)),
    (\+ top(X,_); \+ top(_,X)),
    (\+ top(Y,_); \+ top(_,Y)).

/*  - vertically adjacent boxes can only be moved right or left... if there*/
/*    are empty squares for the Sokoban to push and for the boxes to move. */
stuck(X, Y) :-
    (top(X,Y); top(Y,X)),
    (\+ solution(X); \+ solution(Y)),
    (\+ right(X,_); \+ right(_,X)),
    (\+ right(Y,_); \+ right(_,Y)).

/***************************************************************************/
/* Game rules: move selection                                              */
/***************************************************************************/

/* The Sokoban can move to any empty position in the board, but cannot go  */
/* through boxes.                                                          */
can_reach(P1, P1, _Boxes, _Visited, []):- !.
can_reach(P1, P2, Boxes, _Visited, [move(P1,Dir)]) :-
    neib(P1, P2, Dir),
    \+ member(P2, Boxes), !.
can_reach(P1, P2, Boxes, Visited,[move(P1,Dir)|SokobanMoves]) :-
    neib(P1, P3, Dir),
    P3 \== P2,
    \+ member(P3, Visited),
    \+ member(P3, Boxes),
    can_reach(P3, P2, Boxes, [P3|Visited], SokobanMoves), !.

/* A good place to move a box is one that:                                 */
/*  - is not already occupied by a box.                                    */
/*  - is not one of the positions to avoid regarding the board and boxes.  */
good_move(X, Boxes) :-
    \+ member(X, Boxes),
    \+ stuck(X),
    foreach(member(Box, Boxes), \+ stuck(X, Box)).

getXY(solution(P), X, Y) :-
    getXY(P, X, Y), !.

getXY(P, X, Y) :-
    atom_chars(P, S),
    nth0(1, S, X_char),
    nth0(3, S, Y_char),
    atom_number(X_char, X),
    atom_number(Y_char, Y).

dist(P1, P2, Dist) :-
    getXY(P1, X1, Y1),
    getXY(P2, X2, Y2),
    abs(X1-X2, X_Dist),
    abs(Y1-Y2, Y_Dist),
    Dist = X_Dist + Y_Dist.

get_dir(Loc, Dest, NextDir, NextLoc) :-
    dist(Loc, Dest, Dist),
    neib(Loc, NextLoc, NextDir),
    dist(NextLoc, Dest, NextDist),
    NextDist < Dist.

/* Selection of a good movement given a state:                             */
/*  - any valid movement for every box                                     */
/*  - the Sokoban must be able to access the push position                 */
movement1(state(Sokoban, Boxes, Solutions), push(Box, Dir), SokobanMoves) :-
    select(Box, Boxes, BoxesRemain),
    select(Target, Solutions, _),
    \+ member(Target, Boxes),
    get_dir(Box, Target, Dir, NextLoc),
    good_move(NextLoc, BoxesRemain),
    neib(PushPosition, Box, Dir),
    can_reach(Sokoban, PushPosition, Boxes, [], SokobanMoves),
    \+ member(PushPosition, Boxes).

movement2(state(Sokoban, Boxes, _), push(Box, Dir), SokobanMoves) :-
    select(Box, Boxes, BoxesRemain),
    neib(Box, NextLoc, Dir),
    good_move(NextLoc, BoxesRemain),
    neib(PushPosition, Box, Dir),
    can_reach(Sokoban, PushPosition, Boxes, [], SokobanMoves),
    \+ member(PushPosition, Boxes).

/***************************************************************************/
/* Implementation of the state update functionality.                       */
/***************************************************************************/
update(state(_Sokoban, Boxes, S), push(Box, Dir), state(NewSokoban, NewBoxes, S)) :-
    NewSokoban = Box,
    subtract(Boxes, [Box], TempList),
    neib(Box, NewPosition, Dir),
    append(TempList, [NewPosition], NewBoxes).
