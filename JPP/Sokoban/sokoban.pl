:-include(game).

:- dynamic right/2.
:- dynamic top/2.
:- dynamic solution/1.

solve([Tops, Rights, Boxes, Solutions, sokoban(Sokoban)], Solution) :-
    abolish_all_tables,
    retractall(top(_,_)),
    findall(_, ( member(P, Tops), assert(P) ), _),
    retractall(right(_,_)),
    findall(_, ( member(P, Rights), assert(P) ), _),
    retractall(solution(_)),
    findall(_, ( member(P, Solutions), assert(P) ), _),
    findall(Box, member(box(Box), Boxes), BoxLocs),
    State = state(Sokoban, BoxLocs, Solutions),
    solve_problem(State, Solution).

solve_problem(State, Solution) :-
    ( catch(call_with_time_limit(15, solve1(State, [State], S1)), Error, true) ->
        (   var(Error) -> Solution = S1
        ;   solve2(State, [State], Solution)
        )
    ; solve2(State, [State], Solution)
    ).

solve1(State, _History, []) :-
    final_state(State), !.

solve1(State, History, Moves) :-
    movement1(State, BoxMove, SokobanMoves),
    update(State, BoxMove, NewState),
    \+ member(NewState, History),
    append(SokobanMoves,[BoxMove], CurrentMoves),
    append(CurrentMoves, Moves2, Moves),
    solve1(NewState, [NewState|History], Moves2).

solve2(State, _History, []) :-
    final_state(State), !.

solve2(State, History, Moves) :-
    movement2(State, BoxMove, SokobanMoves),
    update(State, BoxMove, NewState),
    \+ member(NewState, History),
    append(SokobanMoves,[BoxMove], CurrentMoves),
    append(CurrentMoves, Moves2, Moves),
    solve2(NewState, [NewState|History], Moves2).
