#include <stdlib.h>
#include <stdio.h>
#include "tree.h"

// Make tree node with initialized fields.
Tree makeUser(Tree parent) {
    Tree newUser = (Tree)malloc(sizeof(struct TreeNode));

    if (!newUser)  // Failure of memory allocation.
        exit(1);

    newUser->nextBrother = newUser->prevBrother = NULL;
    newUser->firstChild = newUser->lastChild = NULL;
    newUser->movieList = NULL;
    newUser->parent = parent;

    return newUser;
}

// Add user node to given parent.
void addUser(Tree parent, Tree newUser) {
    newUser->prevBrother = parent->lastChild;
    parent->lastChild = newUser;
    if(newUser->prevBrother)
        newUser->prevBrother->nextBrother = newUser;
    else
        parent->firstChild = newUser;
}

// Moves child user's up to parent node.
static void moveChildUsersUp(Tree user) {
    if (user->firstChild) {
        user->lastChild->nextBrother = user->nextBrother;
        user->firstChild->prevBrother = user;
        if (user->nextBrother)
            user->nextBrother->prevBrother = user->lastChild;
        else { // user is lastChild.
            user->parent->lastChild = user->lastChild;
            user->lastChild->parent = user->parent;
        }
        user->nextBrother = user->firstChild;
    }
}

// Remove and delete user from tree.
void delUser(Tree user) {
    deleteList(user->movieList);

    moveChildUsersUp(user);

    // Removing user from tree.
    if (user->nextBrother)
        user->nextBrother->prevBrother = user->prevBrother;
    else { // user is last child in list.
        user->parent->lastChild = user->prevBrother;
        if (user->prevBrother && user->prevBrother->parent != user->parent)
            user->prevBrother->parent = user->parent;
    }

    if (user->prevBrother)
        user->prevBrother->nextBrother = user->nextBrother;
    else { // user is first child in list.
        user->parent->firstChild = user->nextBrother;
        if (user->nextBrother && user->nextBrother->parent != user->parent)
            user->nextBrother->parent = user->parent;
    }

    free(user);
}

// Insert up to k user's top movies to marathonMovies.
// marathonList is initialy empty.
static void marathonInsertTopMoviesSimple(Tree user, List *marathonMovies, List *lastMovie, int k, int *listLength) {
    List userMovies = user->movieList;

    while (userMovies && *listLength < k) {
        appendMovie(marathonMovies, lastMovie, makeMovie(userMovies->movieRating));
        (*listLength)++;
        userMovies = userMovies->next;
    }
}

// Insert user's top movies of rating greater than minRating into marathonMovies
// list if they are higher rated than movies on the list.
// Movies are inserted into proper place keeping marathonMovies sorted in descending order.
static void marathonInsertTopMovies(Tree user, List *marathonMovies, List *lastMovie, int k, int *listLength, int minRating) {
    List marMovie = *marathonMovies, userMovie = user->movieList;

    for (int i = 0; i < k && userMovie && userMovie->movieRating > minRating; i++) {
        if (marMovie) {
            if (userMovie->movieRating > marMovie->movieRating) {
                List movieToAdd = NULL;
                if (*listLength == k) { // Reuse the last element of marathonMovies to insert before marMovie element.
                    (*lastMovie)->movieRating = userMovie->movieRating;
                    if (*lastMovie != marMovie) {
                        movieToAdd = *lastMovie;
                        *lastMovie = movieToAdd->prev;
                        (*lastMovie)->next = NULL;
                    }
                }
                else { // Make a new element.
                    movieToAdd = makeMovie(userMovie->movieRating);
                    (*listLength)++;
                }
                if (movieToAdd)
                    insertMovieBefore(marathonMovies, marMovie, movieToAdd);
                userMovie = userMovie->next;
            }
            else { // userMovie is not better than marMovie
                if (userMovie->movieRating == marMovie->movieRating) // Avoid duplicates.
                    userMovie = userMovie->next;
                marMovie = marMovie->next;
            }
        }
        else { // No marathon movie to compare against.
            if (*listLength < k) { // Append userMovie.
                appendMovie(marathonMovies, lastMovie, makeMovie(userMovie->movieRating));
                (*listLength)++;
                userMovie = userMovie->next;
            }
            else // List is full, no more userMovie can get added.
                break;
        }
    }
}

// Get up to k top movies from given user and recurisvely from child users.
// User movies rating must be > than minRating.
// Child user's movies rating must be > than max(minRating, user's top movie rating).
void runMarathon(Tree user, List *marathonMovies, List *lastMovie, int k, int *listLength, int minRating) {
    if (!(*marathonMovies))
        marathonInsertTopMoviesSimple(user, marathonMovies, lastMovie, k, listLength);
    else
        marathonInsertTopMovies(user, marathonMovies, lastMovie, k, listLength, minRating);

    if (user->movieList && user->movieList->movieRating > minRating)
        minRating = user->movieList->movieRating;

    Tree child = user->firstChild;
    while (child) {
        runMarathon(child, marathonMovies, lastMovie, k, listLength, minRating);
        child = child->nextBrother;
    }
}

void deleteTree(Tree t) {
    Tree child = t->firstChild;
    while (child) {
        Tree nextBrother = child->nextBrother;
        deleteTree(child);
        child = nextBrother;
    }
    deleteList(t->movieList);
    free(t);
}
