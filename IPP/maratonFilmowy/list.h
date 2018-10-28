#ifndef LIST_H
#define LIST_H
#include <stdbool.h>

struct ListNode;

typedef struct ListNode* List;

struct ListNode {
    int movieRating;
    List next, prev;
};

extern List makeMovie(int movieRating);

extern void insertMovieBefore(List *firstMovie, List node, List movieToAdd);

extern void appendMovie(List *firstMovie, List *lastMovie, List movieToAdd);

extern bool addMovie(List *firstMovie, int movieRating);

extern bool delMovie(List *firstMovie, int movieRating);

extern void deleteList(List l);

#endif /* LIST_H */
