#include <stdio.h>
#include <stdlib.h>
#include "list.h"

// Make list node with initialized fields.
List makeMovie(int movieRating) {
    List movie = (List)malloc(sizeof(struct ListNode));

    if (!movie)  // Failure of memory allocation.
        exit(1);

    movie->movieRating = movieRating;
    movie->next = NULL;
    movie->prev = NULL;

    return movie;
}

// Add movie as the first element of list.
static void insertAsFirstMovie(List *firstMovie, List movieToAdd) {
    movieToAdd->next = *firstMovie;
    *firstMovie = movieToAdd;
    if (movieToAdd->next)
        movieToAdd->next->prev = movieToAdd;
}

// Append movie to the end of list.
void appendMovie(List *firstMovie, List *lastMovie, List movieToAdd) {
    movieToAdd->prev = *lastMovie;
    *lastMovie = movieToAdd;
    if (movieToAdd->prev)
        movieToAdd->prev->next = movieToAdd;
    else
        *firstMovie = movieToAdd;
}

// Insert movie before node element.
void insertMovieBefore(List *firstMovie, List node, List movieToAdd) {
    if (node == *firstMovie)
        insertAsFirstMovie(firstMovie, movieToAdd);
    else {
        movieToAdd->next = node;
        movieToAdd->prev = node->prev;
        node->prev->next = movieToAdd;
        node->prev = movieToAdd;
    }
}

// Create movie list node and insert into the list sorted by movieRating in descending order preserving order.
// Returns true if movie was succesfully added, false otherwise.
bool addMovie(List *firstMovie, int movieRating) {
    if (!(*firstMovie)) // Adding movie to empty list.
        insertAsFirstMovie(firstMovie, makeMovie(movieRating));
    else {
        // Searching for first lower rated movie.
        List movie = *firstMovie, prevMovie = NULL;
        while (movie && movie->movieRating > movieRating) {
            prevMovie = movie;
            movie = movie->next;
        }

        if (movie && movie->movieRating == movieRating) // Movie already added.
            return false;

        if (!movie) // All movies were better, append at the end of list.
            appendMovie(firstMovie, &prevMovie, makeMovie(movieRating));
        else // Add before the first lower rated movie.
            insertMovieBefore(firstMovie, movie, makeMovie(movieRating));
    }
    return true;
}

// Find movie of rating movieRating, remove it from the list and delete.
// Return true if movie found, false otherwise.
bool delMovie(List *firstMovie, int movieRating) {
    List movie = *firstMovie;

    while (movie && movie->movieRating > movieRating)
        movie = movie->next;

    if (movie && movie->movieRating == movieRating) { // Movie found.
        if (movie->prev)
            movie->prev->next = movie->next;
        else
            *firstMovie = movie->next;

        if (movie->next)
            movie->next->prev = movie->prev;

        free(movie);
        return true;
    }
    return false;
}

void deleteList(List l) {
    while (l) {
        List toDelete = l;
        l = l->next;
        free(toDelete);
    }
}
