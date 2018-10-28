#ifndef TREE_H
#define TREE_H
#define MAX_USER_ID 65535
#include "list.h"

struct TreeNode;

typedef struct TreeNode* Tree;

struct TreeNode {
    Tree firstChild, lastChild, nextBrother, prevBrother, parent;
    List movieList;
};

extern Tree makeUser(Tree parent);

extern void addUser(Tree parent, Tree user);

extern void delUser(Tree user);

extern void runMarathon(Tree user, List *marathonMovies, List *lastMovie, int k, int *listLength, int minRating);

extern void deleteTree(Tree t);

#endif /* TREE_H */
