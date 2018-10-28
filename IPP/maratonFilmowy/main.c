#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>
#include "tree.h"

#define INPUT_ERROR -1
#define MAX_MOVIE_RATING 2147483647
#define COMMANDS_NUMBER 5
#define MAXCMLEN 8

#define ADD_USER 0
#define DEL_USER 1
#define ADD_MOVIE 2
#define DEL_MOVIE 3
#define MARATHON 4

// Pointers to users of respective id.
Tree *usersById;

// Array of commands.
char *commands[COMMANDS_NUMBER] = {"addUser", "delUser", "addMovie", "delMovie", "marathon"};

// Add user if not already added.
void addUserOp(unsigned short int parentId, unsigned short int userId) {
    if (usersById[userId]) // User has been already added.
        fprintf(stderr, "ERROR\n");
    else {
        Tree parent = usersById[parentId];
        Tree newUser = makeUser(parent);
        addUser(parent, newUser);
        usersById[userId] = newUser;
        printf("OK\n");
    }
}

// Delete user if not root.
void delUserOp(unsigned short int userId) {
    if (userId == 0) // It's forbidden to delete user 0.
        fprintf(stderr, "ERROR\n");
    else {
        delUser(usersById[userId]);
        usersById[userId] = NULL;
        printf("OK\n");
    }
}

// Add movie to user's movie list if not already added.
void addMovieOp(unsigned short int userId, int movieRating) {
    Tree user = usersById[userId];

    if (addMovie(&user->movieList, movieRating))
        printf("OK\n");
    else // Already added.
        fprintf(stderr, "ERROR\n");
}

// Delete movie from user's movie list.
void delMovieOp(unsigned short int userId, int movieRating) {
    Tree user = usersById[userId];

    if (delMovie(&user->movieList, movieRating))
        printf("OK\n");
    else // Not found movie of given rating.
        fprintf(stderr, "ERROR\n");
}


// Find and print up to k movies for marathon operation.
void marathonOp(unsigned short int userId, int k) {
    Tree user = usersById[userId];
    int listLength = 0;
    List lastMovie = NULL;
    List marathonMovies = NULL;

    runMarathon(user, &marathonMovies, &lastMovie, k, &listLength, -1);

    if (marathonMovies) { // Print found movies.
        List movieList = marathonMovies;
        bool firstEl = true;

        while (movieList) {
            if (!firstEl)
                printf(" ");
            else
                firstEl = false;
            printf("%d", movieList->movieRating);
            movieList = movieList->next;
        }
        printf("\n");

        deleteList(marathonMovies);
    }
    else
        printf("NONE\n");
}

// Excecute command checking first argument.
void executeCommand(int command, unsigned short int arg1, int arg2) {
    if (usersById[arg1]) { // First user has to be present.
        switch (command) {
            case ADD_USER:
                addUserOp(arg1, arg2);
                break;

            case DEL_USER:
                delUserOp(arg1);
                break;

            case ADD_MOVIE:
                addMovieOp(arg1, arg2);
                break;

            case DEL_MOVIE:
                delMovieOp(arg1, arg2);
                break;

            case MARATHON:
                marathonOp(arg1, arg2);
                break;
        }
    }
    else
        fprintf(stderr, "ERROR\n");
}

// Parse line for command. Return command index or INPUT_ERROR if not found.
int getCommand(char *line, int *idx) {
    char command[MAXCMLEN+1];
    int i = 0;

    while (isalpha(line[*idx]) && i < MAXCMLEN)
        command[i++] = line[(*idx)++];
    command[i] = '\0';

    for (i = 0; i < COMMANDS_NUMBER; i++)
        if (!strcmp(commands[i], command))
            return i;
    return INPUT_ERROR;
}

// Parse line for number of max value maxNumber. Return number or INPUT_ERROR if error.
int getNumber(char *line, int *idx, long long maxNumber) {
    // Check if first character is digit but not 0.
    if (!isdigit(line[*idx]) || (line[*idx] - '0' == 0 && isdigit(line[*idx+1])))
        return INPUT_ERROR;

    long long int number = 0;

    while (isdigit(line[*idx])) {
        number = 10*number + (line[(*idx)++]-'0');

        if (number > maxNumber)  // Check if number exceeds maxNumber.
            return INPUT_ERROR;
    }

    return number;
}

int main() {
    usersById = (Tree *)malloc((MAX_USER_ID+1) * sizeof(Tree));
    if (!usersById) // Failure of memory allocation.
        exit(1);
    for (int i = 0; i <= MAX_USER_ID; i++)
        usersById[i] = NULL;
    usersById[0] = makeUser(NULL);

    char *line = NULL;
    size_t len = 0;
    ssize_t nread;

    while ((nread = getline(&line, &len, stdin)) != -1) { // Reading line from stdin.
        if (line[0] == '#' || line[0] == '\n')
            continue;

        bool succ = false;  // Set to true if parsing was succesful.
        while (true) { // For breaking out on error.
            int idx = 0, command = getCommand(line, &idx);
            if (command == INPUT_ERROR || line[idx] != ' ')
                break;

            idx++;
            int arg1 = getNumber(line, &idx, MAX_USER_ID), arg2 = -1; // arg1 always userId.
            if (arg1 == INPUT_ERROR)
                break;

            if (command == DEL_USER) { // Only delUser has one argument.
                if (line[idx] != '\n')
                    break;
                executeCommand(command, arg1, arg2);
                succ = true;
                break;
            }

            if (line[idx] != ' ')
                break;

            // Two arguments case.
            idx++;
            arg2 = getNumber(line, &idx, (command == ADD_USER) ? MAX_USER_ID : MAX_MOVIE_RATING);
            if (arg2 == INPUT_ERROR || line[idx] != '\n')
                break;

            executeCommand(command, arg1, arg2);
            succ = true;
            break;
        }
        if (!succ)
            fprintf(stderr, "ERROR\n");
    }

    free(line);
    deleteTree(usersById[0]);
    free(usersById);

    return 0;
}
