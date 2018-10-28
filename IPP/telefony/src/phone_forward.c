/** @file
 * Implementacja interfeju klasy przechowującej przekierowania numerów telefonicznych
 *
 * @author Michał Futrega <michal.futrega@student.uw.edu.pl>
 * @copyright Uniwersytet Warszawski
 * @date 09.04.2018
 */

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "phone_forward.h"

/** @brief Tworzy nową struktuę.
 * Tworzy nową struktuę TreeNode i ustawia jej pola na NULL.
 * @return Wskaźnik na utworzoną strukturę lub NULL, gdy nie
 *         udało się zaalokować pamięci.
 */
static TreeNode * newTN() {
    TreeNode *t = malloc(sizeof(TreeNode));
    if (!t)
        return NULL;
    t->fwdToList = NULL;
    t->fwdFrom = NULL;
    for (int i = 0; i < DIGIT_COUNT; i++)
        t->child[i] = NULL;
    return t;
}

/** @brief Tworzy nową struktuę.
 * Tworzy nową struktuę ListNode w której pole num ustawiono
 * na @p num oraz pole next na NULL.
 * @param[in] num – wskaźnik na napis reprezentujący numer telefonu.
 * @return Wskaźnik na utworzoną strukturę lub NULL, gdy nie
 *         udało się zaalokować pamięci.
 */
static ListNode * newLN(char const *num) {
    ListNode *l = malloc(sizeof(ListNode));
    if (!l)
        return NULL;
    l->num = malloc((strlen(num)+1)*sizeof(char));
    if (!l->num) {
        free(l);
        return NULL;
    }
    strcpy(l->num, num);
    l->next = NULL;
    return l;
}

/** @brief Usuwa węzeł ListNode.
 * Usuwa węzeł listy wskazywany przez @p l. Nic nie robi,
 * jeśli wskaźnik ten ma wartość NULL.
 * @param[in, out] l – wskaźnik na usuwany węzeł listy.
 */
static void delLN(ListNode *l) {
    if (!l)
        return;
    free(l->num);
    free(l);
}

/** @brief Usuwa listę @p l.
 * @param[in, out] l – wskaźnik na początek usuwanej listy.
 */
static void delList(ListNode *l) {
    while (l) {
        ListNode *toDel = l;
        l = l->next;
        delLN(toDel);
    }
}

/** @brief Usuwa węzeł listy.
 * Usuwa z listy @p l węzeł o numerze @p num.
 * @param[in, out] l – wskaźnik do wskaźnika początku listy;
 * @param[in] num    – wskaźnik na napis odpowiadający numerowi, który usuwamy.
 */
static void delNumLN(ListNode **l, char const *num) {
    ListNode *curr = *l, *prev = NULL;
    while (curr) {
        if (strcmp(curr->num, num) == 0) {
            if (prev)
                prev->next = curr->next;
            else
                *l = (*l)->next;
            delLN(curr);
            return;
        }
        prev = curr;
        curr = curr->next;
    }
}

/** @brief Usuwa węzeł drzewa.
 * Usuwa węzeł drzewa wskazywany przez @p t. Nic
 * nie robi, jeśli wskaźnik ten ma wartość NULL.
 * @param[in, out] t – wskaźnik na usuwany węzeł.
 */
static void delTN(TreeNode *t) {
    if (!t)
        return;
    delList(t->fwdToList);
    free(t->fwdFrom);
    free(t);
}

/** @brief Usuwa drzewo.
 * @param[in, out] t – wskaźnik na korzeń usuwanego drzewa.
 */
static void delTree(TreeNode *t) {
    for (int i = 0; i < DIGIT_COUNT; i++)
        if (t->child[i])
            delTree(t->child[i]);
    delTN(t);
}

/** @brief Usuwa niepotrzebne węzły w drzewie.
 * Usuwa niepotrzebne węzły ze scieżki od początkowej wartości
 * @p t do wezła odpowiadającemu numerowi @p num lub jego
 * maksymalnemu prefiksowi, gdy @p num nie ma w drzewie.
 * @param[in, out] t        – wskaźnik do węzeła drzewa;
 * @param[in] parent   – wskaźnik do rodzica węzła t;
 * @param[in] num      – wskaźnik na napis odpowiadający numerowi;
 * @param[in] childIdx – indeks taki, że parent->child[childIdx] = t.
 */
static bool delEmptyNodes(TreeNode *t, TreeNode *parent, char const *num, int childIdx) {
    if (!t)
        return true;
    if (num[0] == '\0' || delEmptyNodes(t->child[num[0] - '0'], t, num + 1, num[0] - '0')) {
        if (t->fwdToList)
            return false;
        if (!parent) // Proba usuniecia korzenia.
            return false;
        for (int i = 0; i < DIGIT_COUNT; i++) // Jesli ma dzieci to konczymy.
            if (t->child[i])
                return false;
        delTN(t);
        parent->child[childIdx] = NULL;
        return true;
    }
    return false;
}

/** @brief Tworzy nową strukturę.
 * Tworzy strukturę PhoneNumbers z tablicą o zadanym rozmiarze @p size.
 * @param[in] size – rozmiar tablicy na numery telefonów.
 * @return Wskaźnik na utworzoną strukturę lub NULL, gdy nie udało się
 *         zaalokować pamięci.
 */
struct PhoneNumbers * newPhnNum(size_t size) {
    struct PhoneNumbers *phoneBook = malloc(sizeof(struct PhoneNumbers));
    if (!phoneBook)
        return NULL;
    phoneBook->phnums = malloc(size*sizeof(char*));
    if (!phoneBook->phnums) {
        free(phoneBook);
        return NULL;
    }
    phoneBook->size = size;
    for (size_t i = 0; i < size; i++)
        phoneBook->phnums[i] = NULL;
    return phoneBook;
}

/** @brief Znajduje węzeł odpowiadający numerowi.
 * Znajduje w drzewie o korzeniu @p t węzeł odpowiadający numerowi @p num.
 * @param[in] t   – wskaźnik na korzeń drzewa;
 * @param[in] num – wskaźnik na napis reprezentujący numer.
 * @return Wskaźnik na znaleziony węzeł lub NULL, gdy takiego węzła nie ma.
 */
static TreeNode * getTN(TreeNode *t, char const *num) {
    for (int i = 0; num[i] != '\0'; i++) {
        int digit = num[i] - '0';
        if (!t->child[digit])
            return NULL;
        t = t->child[digit];
    }
    return t;
}

/** @brief Zmienia przekierowanie.
 * W drzewie przekierowań dla węzła @p t zmieniamy przekierowanie na @p fwd.
 * @param[in, out] t – wskaźnik na węzeł drzewa, w którym zmieniamy przekierowanie;
 * @param[in] fwd    – wskaźnik na napis reprezentujący nowe przekierowanie.
 * @return Wartość @p true, jeśli pomyślnie zmieniono przekierowanie.
 *         Wartość @p false, jeśli nie udało się zaalokować pamięci.
 */
static bool changeFwd(TreeNode *t, char const *fwd) {
    free(t->fwdToList->num);
    t->fwdToList->num = malloc((strlen(fwd)+1)*sizeof(char));
    if (!t->fwdToList->num)
        return false;
    strcpy(t->fwdToList->num, fwd);
    return true;
}

/** @brief Usuwa przekierowanie.
 * Usuwa przekierowanie z @p num1 na @p num2 w drzewie o korzeniu @p t.
 * Nic nie robi jeśli w drzewie nie występuje przekierowanie z @p num1.
 * @param[in, out] t – wskaźnik na korzeń drzewa;
 * @param[in] num1   – wskaźnik na napis reprezentujący prefiks numerów
 *                     przekierowywanych;
 * @param[in] num2   – wskaźnik na napis reprezentujący prefiks numerów, na które
 *                     jest wykonywane przekierowanie.
 */
static void delFwd(TreeNode *t, char const *num1, char const *num2) {
    TreeNode *tree = getTN(t, num1);
    if (!tree)
        return;
    delNumLN(&tree->fwdToList, num2);
}

/** @brief Sprawdza czy napis reprezentuje numer telefonu.
 * Sprawdza czy napis reprezentuje numer telefonu. Numer telefonu
 * jest to niepusty ciąg składający się z cyfr 0,1,2,3,4,5,6,7,8,9,:,;.
 * @param[in] num – wskaźnik na napis.
 * @return Wartość @p true, jeśli napis reprezentuje numer telefonu.
 *         Wartość @p false, jeśli napis nie reprezentuje numeru
 *         telefonu lub @p num ma wartość NULL.
 */
static bool isNum(char const *num) {
    if (!num)
        return false;
    int i = 0;
    for (i = 0; isdigit(num[i]) || num[i] == ':' || num[i] == ';'; i++);
    return (i > 0 && num[i] == '\0');
}

struct PhoneForward * phfwdNew() {
    struct PhoneForward *pf = malloc(sizeof(struct PhoneForward));
    if (!pf)
        return NULL;

    pf->fwdTree = newTN();
    if (!pf->fwdTree) {
        free(pf);
        return NULL;
    }
    pf->inverseFwdTree = newTN();
    if (!pf->inverseFwdTree) {
        free(pf);
        free(pf->fwdTree);
        return NULL;
    }
    return pf;
}

void phfwdDelete(struct PhoneForward *pf) {
    if (!pf)
        return;
    delTree(pf->fwdTree);
    delTree(pf->inverseFwdTree);
    free(pf);
}

/** @brief Dodaje przekierowanie.
 * Dodaje przekierowanie w drzewie @p fwdTree z @p num1 na @p num2. Jeśli wcześniej zostało
 * dodane przekierowanie z takim samym parametrem @p num1, to jest ono zastępowane.
 * Jesli @p invFwdTree != NULL to dodawane jest do niego przekierowanie z @p num2 na @p num1.
 * @param[in, out] fwdTree – wskaźnik na drzewo z przekierowaniami;
 * @param[in] num1         – wskaźnik na napis reprezentujący prefiks numerów
 *                           przekierowywanych;
 * @param[in] num2         – wskaźnik na napis reprezentujący prefiks numerów, na które
 *                           jest wykonywane przekierowanie;
 * @param[in, out] invFwdTree – wskaźnik na drzewo z przekierowaniami odwrotnymi.
 * @return Wartość @p true, jeśli przekierowanie zostało dodane.
 *         Wartość @p false, jeśli wystąpił błąd, np. podany napis nie
 *         reprezentuje numeru, oba podane numery są identyczne lub nie udało
 *         się zaalokować pamięci.
 */
static bool addFwd(TreeNode *fwdTree, char const *num1, char const *num2, TreeNode *invFwdTree) {
    for (int i = 0; num1[i] != '\0'; i++) { // Znajduje wezel odp. num1, jesli taki wezel nie istnieje to dobudowuje do niego sciezke.
        int digit = num1[i] - '0';
        if (!fwdTree->child[digit]) {
            fwdTree->child[digit] = newTN();
            if (!fwdTree->child[digit])
                return false;
        }
        fwdTree = fwdTree->child[digit];
    }
    if (invFwdTree && fwdTree->fwdToList) { // Zamiana przekierowania.
        if (strcmp(fwdTree->fwdToList->num, num2) == 0) // Proba dodania tego samego przekierowania.
            return true;
        delFwd(invFwdTree, fwdTree->fwdToList->num, num1);
        delEmptyNodes(invFwdTree, NULL, fwdTree->fwdToList->num, 0);
        if (!changeFwd(fwdTree, num2))
            return false;
        if (!addFwd(invFwdTree, num2, num1, NULL))
            return false;
        return true;
    }
    if (!fwdTree->fwdFrom) { // Uzupelnienie informacji o przekierowywanym numerze.
        fwdTree->fwdFrom = malloc((strlen(num1)+1)*sizeof(char));
        if (!fwdTree->fwdFrom)
            return false;
        strcpy(fwdTree->fwdFrom, num1);
    }
    // Dodawanie przekierowania.
    ListNode *newFwd = newLN(num2);
    if (!newFwd)
        return false;
    newFwd->next = fwdTree->fwdToList;
    fwdTree->fwdToList = newFwd;

    if (invFwdTree)
        if (!addFwd(invFwdTree, num2, num1, NULL))
            return false;
    return true;
}

bool phfwdAdd(struct PhoneForward *pf, char const *num1, char const *num2) {
    if (!pf || !isNum(num1) || !isNum(num2) || !strcmp(num1, num2))
        return false;
    return addFwd(pf->fwdTree, num1, num2, pf->inverseFwdTree);
}

/** @brief Usuwa wszystkie przekierowania odwrotne.
 * Usuwa wszystkie przekierowania odwrotne z drzewa @p invFwdTree wystepujace w poddrzewie
 * drzewa przekierowań wyznaczonym przez węzeł @p fwdTree.
 * @param[in] fwdTree         – wskaźnik na węzeł w drzewie przekierowań;
 * @param[in, out] invFwdTree – wskaźnik na korzeń drzewa przekierowań odwrotnych.
 */
static void removeInvFwd(TreeNode *fwdTree, TreeNode *invFwdTree) {
    if (fwdTree->fwdToList) { // Istnieja przekierowania.
        delFwd(invFwdTree, fwdTree->fwdToList->num, fwdTree->fwdFrom);
        delEmptyNodes(invFwdTree, NULL, fwdTree->fwdToList->num, 0);
    }
    for (int i = 0; i < DIGIT_COUNT; i++)
        if (fwdTree->child[i])
            removeInvFwd(fwdTree->child[i], invFwdTree);
}

void phfwdRemove(struct PhoneForward *pf, char const *num) {
    if (!pf || !isNum(num))
        return;
    TreeNode *t = pf->fwdTree, *parent = NULL;
    for (int i = 0; num[i] != '\0'; i++) {
        int digit = num[i] - '0';
        if (!t->child[digit])
            return;
        parent = t;
        t = t->child[digit];
    }
    removeInvFwd(t, pf->inverseFwdTree);
    delTree(t);
    if (parent)
        parent->child[num[strlen(num)-1]-'0'] = NULL;
    delEmptyNodes(pf->fwdTree, NULL, num, 0);
}

struct PhoneNumbers const * phfwdGet(struct PhoneForward *pf, char const *num) {
    if (!pf || !isNum(num)) // Jeśli podany napis nie reprezentuje numeru, wynikiem jest pusty ciąg.
        return newPhnNum(0);
    int idx = -1;
    TreeNode *t = pf->fwdTree, *maxFwdNode = NULL;
    for (int i = 0; num[i] != '\0'; i++) { // Zapisanie na fwdNode wezla z przekierowaniem odp. max prefiksowi num.
        int digit = num[i] - '0';
        if (!t->child[digit])
            break;
        if (t->child[digit]->fwdToList) {
            maxFwdNode = t->child[digit];
            idx=i;
        }
        t = t->child[digit];
    }
    struct PhoneNumbers *phnNumber = newPhnNum(1);
    if (!phnNumber)
        return NULL;
    if (idx != -1) { // Znaleziono przekierowanie.
        phnNumber->phnums[0] = malloc((strlen(maxFwdNode->fwdToList->num) + strlen(num+idx+1) + 1) * sizeof(char));
        if (!phnNumber->phnums[0]) {
            free(phnNumber);
            return NULL;
        }
        strcpy(phnNumber->phnums[0], maxFwdNode->fwdToList->num);
        strcat(phnNumber->phnums[0], num + idx+1);
    } else { // Jeśli dany numer nie zostal przekierowany, to wynikiem jest ten numer.
        phnNumber->phnums[0] = malloc((strlen(num)+1)*sizeof(char));
        if (!phnNumber->phnums[0]) {
            free(phnNumber);
            return NULL;
        }
        strcpy(phnNumber->phnums[0], num);
    }
    return phnNumber;
}

/** @brief Dodaje do listy nowy numer.
 * Dodaje zachowując porządek leksykograficzny do listy @p l numer @p num.
 * Jeśli numer @ num występuje już w liście to nie jest ponownie dodawany.
 * @param[in, out] l – wskaźnik do wskaźnika na początek listy;
 * @param[in] num    – wskaźnik na dodawany napis reprezentujący numer.
 * @return Wartość @p true, jeśli numer został dodany lub wystpował już w liście.
 *         Wartość @p false, jeśli nie udało się zaalokować pamięci.
 */
static bool insertSortedNumToList(ListNode **l, char *num) {
    ListNode *prev = NULL, *curr = *l;
    while (curr) {
        int strDiff = strcmp(curr->num, num);
        if (strDiff == 0) // Numer już został dodany do listy.
            return true;
        if (strDiff > 0) // Znaleziono pierwszy numer wiekszy leksykograficznie
            break;
        prev = curr;
        curr = curr->next;
    }
    // Dodawanie przed pierwszy wiekszy numer.
    ListNode *newNode = newLN(num);
    if (!newNode)
        return false;
    newNode->next = curr;
    if (!prev)
        *l = newNode;
    else
        prev->next = newNode;
    return true;
}

/** @brief Dodaje przekierowania odwrotne do listy.
 * Znajduje przekierowania odwrotne w drzewie @p invTree numeru @p num
 * i dodaje je do listy @ l w porządku leksykograficznym.
 * @param[in] invTree – wskaźnik do drzewa z przekierowaniami odwrotnymi;
 * @param[in, out] l  – wskaźnik do wskaźnika na początek listy;
 * @param[in] num     – wskaźnik napis reprezentujący numer;
 * @return Wartość @p true, jeśli nie wystąpił błąd z alokacją pamięci.
 *         Wartość @p false, jeśli nie udało się zaalokować pamięci.
 */
static bool findInvFwdNums(TreeNode *invTree, ListNode **l, char const *num) {
    for (int i = 0; num[i] != '\0'; i++) {
        int digit = num[i]-'0';
        if (!invTree->child[digit]) // Nie ma wiecej przekierowan.
            return true;
        ListNode *fwdList = invTree->child[digit]->fwdToList;
        while (fwdList) { // Dodawanie przekierowan do listy.
            char *number = malloc((strlen(fwdList->num) + strlen(num+i+1) + 1) * sizeof(char));
            if (!number)
                return false;

            strcpy(number, fwdList->num);
            strcat(number, num+i+1);
            bool succ = insertSortedNumToList(l, number);

            free(number);
            if (!succ)
                return false;
            fwdList = fwdList->next;
        }
        invTree = invTree->child[digit];
    }
    return true;
}

struct PhoneNumbers const * phfwdReverse(struct PhoneForward *pf, char const *num) {
    if (!pf || !isNum(num)) // Jeśli podany napis nie reprezentuje numeru, wynikiem jest pusty ciąg.
        return newPhnNum(0);

    ListNode *resNumList = newLN(num); // Lista na przekierowania odwrotne.
    if (!resNumList)
        return NULL;
    if (!findInvFwdNums(pf->inverseFwdTree, &resNumList, num)) { // Znajduje przekier odwr, zwraca NULL w przyp niepowodzenia
        delList(resNumList);
        return NULL;
    }
    size_t listLen = 0;
    ListNode *l = resNumList;
    while (l) {
        listLen++;
        l = l->next;
    }
    struct PhoneNumbers *pnum = newPhnNum(listLen);
    if (!pnum) {
        delList(resNumList);
        return NULL;
    }
    l = resNumList;
    for (size_t i = 0; i < listLen; i++) { // Przepisanie listy na tablice.
        pnum->phnums[i] = l->num;
        l->num = NULL;
        l = l->next;
    }
    delList(resNumList);
    return pnum;
}

void phnumDelete(struct PhoneNumbers const *pnum) {
    if (!pnum) // Nic nie robi, jeśli wskaźnik ten ma wartość NULL.
        return;
    for (size_t i = 0; i < pnum->size; i++)
        free(pnum->phnums[i]);
    free(pnum->phnums);
    free((void*)pnum);
}

char const * phnumGet(struct PhoneNumbers const *pnum, size_t idx) {
    return (pnum && (size_t)pnum->size > idx) ? pnum->phnums[idx] : NULL;
}

/** @brief Oblicza x^y modulo dwa do potęgi liczba bitów reprezentacji typu size_t.
 * @param[in] x - podstawa potęgi.
 * @param[in] y - wykładnik potęgi.
 * @return x^y mod dwa do potęgi liczba bitów reprezentacji typu size_t.
 */
size_t powSizeT(size_t x, size_t y) {
    if (y == 0)
        return 1;
    if (y % 2 == 1)
        return x * powSizeT(x, y - 1);
    size_t t = powSizeT(x, y / 2);
    return t * t;
}

/** @brief Oblicza liczbę nietrywialych przekierowań.
 * @param[in] invFwd        - wskaźnik na drzewo z przekierowaniami odwrotnymi
 * @param[in] len           - długość nietrywialnego numeru
 * @param[in] numLen        - długość numeru odpowiadającego obecnemu węzłowi @param invFwd
 * @param[in] digitsInSet   - tablica boolowska reprezentujaca wystujace cyfry w @p set
 * @param[in] digitCountSet - liczba roznych cyfr w @p set
 * @return - Liczba nietrywialnych numerów modulo dwa do potęgi liczba bitów reprezentacji typu size_t.
 */
static size_t nonTrivialCount(TreeNode *invFwd, size_t len, size_t numLen, bool digitsInSet[DIGIT_COUNT], size_t digitCountSet) {
    if (!invFwd || numLen > len)
        return 0;
    if (invFwd->fwdToList) // Znalezlismy nietrywialny numer konczymy rekurencje.
        return powSizeT(digitCountSet, (len - numLen));
    size_t count = 0;
    for (int i = 0; i < DIGIT_COUNT; i++)
        if (digitsInSet[i]) // Wywolanie rekurencji dla cyfr nalezacych do set.
            count += nonTrivialCount(invFwd->child[i], len, numLen + 1, digitsInSet, digitCountSet);
    return count;
}

size_t phfwdNonTrivialCount(struct PhoneForward *pf, char const *set, size_t len) {
    if (!pf || !set || set[0] == '\0' || len == 0)
        return 0;
    bool digitsInSet[DIGIT_COUNT];
    for (int i = 0; i < DIGIT_COUNT; i++)
        digitsInSet[i] = false;
    size_t digitCountSet = 0;
    for (int i = 0; set[i] != '\0'; i++) { // Uzupelnienie informacji o cyfrach w set.
        if (!digitsInSet[set[i] - '0']) {
            digitsInSet[set[i] - '0'] = true;
            digitCountSet++;
            if (digitCountSet == DIGIT_COUNT) // Znaleziono wszystkie mozliwe cyfry.
                break;
        }
    }
    if (digitCountSet == 0)
        return 0;
    return nonTrivialCount(pf->inverseFwdTree, len, 0, digitsInSet, digitCountSet);
}