/** @file
 * Implementacja modułu udostępniającego operacje na numerach telefonów przez interfejs tekstowy.
 *
 * @author Michał Futrega <michal.futrega@student.uw.edu.pl>
 * @copyright Uniwersytet Warszawski
 * @date 30.05.2018
 */

#include "phone_forward.h"
#include <stdio.h>
#include <memory.h>
#include <ctype.h>

#define ERR_EOF 0      ///< Stała błędu niespodziwanego zakończenia programu.
#define ERR_SYNTAX 1   ///< Stała błędu błędu składniowego.
#define ERR_MALLOC 2   ///< Stała błędu alokacji pamięci.
#define ERR_OP 3       ///< Stała błędu wykonania.
#define MAX_OP_LEN 4   ///< Maksymalna długości operacji.
#define ID_CHAR_NUM 64 ///< Liczba dozwolonych znaków identyfikatora.

/**
 * Drzewo trie przechowujące bazy przekierowań.
 */
typedef struct PfDB {
    char *id;                 ///< Wskaźnik na napis reprezentujący identyfikator bazy.
    struct PhoneForward *pf;  ///< Wskaźnik na bazę przekierowań.
    struct PfDB *child[ID_CHAR_NUM];   ///< Tablica wskaźników na synów węzła. Indeks tablicy odpowiada następnemu znakowi identyfiaktora.
} PfDB;

long long int byteCount = 0;  ///< Liczba wczytanych bajtow.
long long int opByteNum = 0;  ///< Numer wczytanego bajtu dla pierwszego znaku operacji.
PfDB *pfBase = NULL;          ///< Struktura przechowująca bazy przekierowań.
char *num1 = NULL;            ///< Bufor na numer.
char *num2 = NULL;            ///< Bufor na numer.
char *id = NULL;              ///< Bufor na identyfiaktor.

/**
 * Wyświetla napis błędu oraz kończy program.
 * @param[in] err     - numer odpowiadający błędowi.
 * @param[in] errByte - numer wczytanego bajtu po którym nastąpił bład.
 * @param[in] op      - napis odpowiadający operatorowi jeśli nastąpił bład wywołania lub wpp NULL.
 */
static void reportError(int err, long long int errByte, char *op) {
    if (err == ERR_EOF)
        fprintf(stderr, "ERROR EOF\n");
    else if (err == ERR_SYNTAX)
        fprintf(stderr, "ERROR %lli\n", errByte);
    else if (err == ERR_MALLOC)
        fprintf(stderr, "Memory allocation fail\n");
    else if (err == ERR_OP)
        fprintf(stderr, "ERROR %s %lli\n", op, errByte);
    exit(1);
}

/**
 * Sprawdza czy @param c jest cyfrą.
 * @param[in] c - kod ascii znaku.
 * @return @p true jeśli @param c odpowiada cyfrze.
 */
static bool isDigit(int c) {
    return (isdigit(c) || c == ':' || c == ';');
}

/**
 * Oblicza indeks odpowiadający znakowi @param c w tablicy synów bazy przekierowań.
 * @param[in] c - kod ascii znaku dla którego obliczany jest indeks.
 * @return Indeks odpowiadający znakowi @param c w tablicy synów bazy przekierowań.
 */
static int getIdx(int c) {
    if (isDigit(c))
        return c - '0';
    if (islower(c))
        return c - 'a' + 12;
    return c - 'A' + 38;
}

/** @brief Tworzy nową strukturę.
 * Tworzy pustą bazę przekierowań. W przypadku gdy nie udalo sie zaalokowac pamieci przerywa program.
 * @return Wskaźnik na utworzoną strukturę.
 */
static PfDB * newPfDB() {
    PfDB *newDB = malloc(sizeof(PfDB));
    if (!newDB)
        reportError(ERR_MALLOC, 0, NULL);
    newDB->id = NULL;
    newDB->pf = NULL;
    for (int i = 0; i < ID_CHAR_NUM; i++)
        newDB->child[i] = NULL;
    return newDB;
}

/** @brief Dodaje nowa baze przekierowan.
 * Dodaje nowa baze przekierowan o identyfikatorze @param id oraz ustawia ja jako aktualna baze.
 * @param[in] id          - identyfiaktor dodawanej bazy.
 * @param[in, out] currDB - aktualna baza przekierowań.
 */
static void addPfToDb(char *id, PfDB **currDB) {
    PfDB *t = pfBase;
    for (int i = 0; id[i] != '\0'; i++) {
        int idx = getIdx(id[i]);
        if (!t->child[idx])
            t->child[idx] = newPfDB();
        t = t->child[idx];
    }
    if (!t->pf) {
        t->pf = phfwdNew();
        if (!t->pf)
            reportError(ERR_MALLOC, 0, NULL);
        t->id = malloc((strlen(id) + 1)*sizeof(char));
        if (!t->id)
            reportError(ERR_MALLOC, 0, NULL);
        strcpy(t->id, id);
    }
    *currDB = t;
}

/** @brief Usuwa węzeł drzewa PfDB.
 * Usuwa węzeł drzewa PfDB.
 * @param[in, out] t - wskaźnik na usuwany węzeł
 */
static void delPfDBNode(PfDB *t) {
    free(t->id);
    phfwdDelete(t->pf);
    free(t);
}

/** @brief Usuwa niepotrzebne węzły w drzewie.
 * Usuwa niepotrzebne węzły ze scieżki od początkowej wartości
 * @p t do wezła odpowiadającemu identyfikatorowi @p id lub gdy @p
 * id nie ma w drzewie jego maksymalnemu prefiksowi.
 * @param[in, out] t   – wskaźnik do węzeła drzewa;
 * @param[in] parent   – wskaźnik do rodzica węzła t;
 * @param[in] id       – wskaźnik na napis odpowiadający numerowi;
 * @param[in] childIdx – indeks taki, że parent->child[childIdx] = t.
 */
static bool delEmptyNodes(PfDB *t, PfDB *parent, char *id, int childIdx) {
    if (!t)
        return true;
    if (id[0] == '\0' || delEmptyNodes(t->child[getIdx(id[0])], t, id + 1, getIdx(id[0]))) {
        if (t->pf)
            return false;
        if (!parent) // Proba usuniecia korzenia.
            return false;
        for (int i = 0; i < ID_CHAR_NUM; i++) // Jesli ma dzieci to konczymy.
            if (t->child[i])
                return false;
        delPfDBNode(t);
        parent->child[childIdx] = NULL;
        return true;
    }
    return false;
}

/** @brief Usuwa bazę przekierowań.
 * Usuwa z drzewa bazę przekierowan o identyfiaktorze @param id
 * @param[in] id - napis odpowiadający identyfiaktorowi usuwanej bazy.
 * @return Wartość @p true, baza została usunięta.
 *         Wartość @p false, jeśli probowano usunac nieistniejaca baze.
 */
static bool delPfDB(char *id) {
    PfDB *t = pfBase, *parent = NULL;
    for (int i = 0; id[i] != '\0'; i++) {
        int idx = getIdx(id[i]);
        if (!t->child[idx]) // Nie istnieje baza przekierowań
            return false;
        parent = t;
        t = t->child[idx];
    }
    if (!t->pf) // Nie istnieje baza przekierowan.
        return false;

    bool isLeaf = true;
    for (int i = 0; i < ID_CHAR_NUM; i++)
        if (t->child[i]) {
            isLeaf = false;
            break;
        }
    phfwdDelete(t->pf);
    free(t->id);
    t->pf = NULL;
    t->id = NULL;
    if (isLeaf) {
        free(t);
        parent->child[getIdx(id[strlen(id) - 1])] = NULL;
        delEmptyNodes(pfBase, NULL, id, 0);
    }
    return true;
}

/** @brief Usuwa strukturę.
 * Usuwa drzewo przechowujące bazy przekierowań.
 * @param[in, out] pfBase - wskaźnik na usuwany węzeł drzewa.
 */
static void delPfBase(PfDB *pfBase) {
    for (int i = 0; i < ID_CHAR_NUM; i++)
        if (pfBase->child[i])
            delPfBase(pfBase->child[i]);
    phfwdDelete(pfBase->pf);
    free(pfBase->id);
    free(pfBase);
}

/** Dodaje znak @param c do bufora @param buffer
 * Dodaje znak @param c do bufora @param buffer W przypadku braku miejsca w buforze rozszerza go.
 * @param[in, out] buffer   - wskaznik do bufora;
 * @param[in, out] buffSize - wskaznik do rozmiaru bufora;
 * @param[in, out] idx      - wskaznik do indeksu pod ktorym zapisujemy znak w buforze;
 * @param[in] c             - zapisywany znak do bufora.
 */
static void addCharToBuffer(char **buffer, long long int *buffSize, long long int *idx, int c) {
    if (*idx + 1 == *buffSize) {
        *buffSize *= 2;
        char *tmpRealloc = realloc(*buffer, *buffSize * sizeof(char));
        if (!tmpRealloc) {
            free(*buffer);
            reportError(ERR_MALLOC, 0, NULL);
        }
        *buffer = tmpRealloc;
    }
    (*buffer)[(*idx)++] = (char)c;
}

/**
 * Wczytuje znak oraz wieksza liczbe wczytanych znakow.
 * @return wczytany znak ze standardowego wejscia.
 */
static int getNextChar() {
    byteCount++;
    return getchar();
}

/**
 * Zwraca znak @param c na standardwowe wejscie oraz zmiejsza liczbe wczytnych znakow.
 * @param[in] c - Kod ascii zwracanego znaku.
 */
static void ungetChar(int c) {
    byteCount--;
    ungetc(c, stdin);
}

/** @brief Pomija białe znaki i komentarze.
 * @param[in] reportEOF - Jesli @p true, to po wczytaniu znaku EOF zgłasza bład ERR_EOF.
 * @return pierwszy znak niebędący komentarzem oraz bialym znakiem.
 */
static int getCharSkip(bool reportEOF) {
    while (1) {
        int c = getNextChar();
        if (reportEOF && c == EOF)
            reportError(ERR_EOF, 0, NULL);
        if (isspace(c))
            continue;
        if (c != '$')
            return c;
        if (getchar() != '$')
            reportError(ERR_SYNTAX, byteCount, NULL);
        byteCount++;
        bool lastChar$ = false;
        while (1) {
            c = getNextChar();
            if (c == EOF) // Brak zakonczenia komentarza.
                reportError(ERR_EOF, 0, NULL);
            if (lastChar$ && c == '$')
                break;
            lastChar$ = (c == '$');
        }
    }
}

/** @brief Wczytuje numer.
 * Wczytuje numer. W przypadku wczytania blednego numeru zglasza blad.
 * @param[in, out] num     - wskaznik do bufora na numer;
 * @param[in, out] sizeNum - wskaznik do rozmiaru bufora.
 */
static void readNum(char **num, long long int *sizeNum) {
    int c = getCharSkip(true);
    if (!isDigit(c))
        reportError(ERR_SYNTAX, byteCount, NULL);
    long long int idx = 0;
    while (isDigit(c)) {
        addCharToBuffer(num, sizeNum, &idx, c);
        c = getNextChar();
    }
    (*num)[idx] = '\0';
    ungetChar(c);
}

/** @brief Wczytuje identyfiaktor.
 * Wczytuje identyfiaktor. W przypadku wczytania blednego identyfiaktora zglasza blad.
 * @param[in, out] id     - wskaznik do bufora na identyfiaktor;
 * @param[in, out] sizeId - wskaznik do rozmiaru bufora.
 */
static void readId(char **id, long long int *sizeId) {
    int c = getCharSkip(true);
    if (!isalpha(c)) // Identyfiaktor musi zaczynac sie od litery.
        reportError(ERR_SYNTAX, byteCount, NULL);
    long long int idx = 0;
    while (isDigit(c) || isalpha(c)) {
        addCharToBuffer(id, sizeId, &idx, c);
        c = getNextChar();
    }
    (*id)[idx] = '\0';
    ungetChar(c);
    if (strcmp(*id, "NEW") == 0 || strcmp(*id, "DEL") == 0) // Nie może być identyfikatorów NEW i DEL.
        reportError(ERR_SYNTAX, byteCount - 2, NULL);
}

/** @brief Wczytuje operacje.
 * Wczytuje operacje. W przypadku wczytania błednej operacji zgłasza bład.
 * @param[in, out] op - tablica na opeartor.
 */
static void readOp(char op[MAX_OP_LEN]) {
    int c = getCharSkip(true);
    op[0] = (char) c;
    opByteNum = byteCount;
    if (c == '?' || c == '>' || c == '@') {
        op[1] = '\0';
        return;
    }
    for (int i = 1; i < MAX_OP_LEN - 1; i++)
        op[i] = (char) getNextChar();
    op[MAX_OP_LEN - 1] = '\0';
    if (strcmp(op, "NEW") != 0 && strcmp(op, "DEL") != 0) // Wczytano błędny operator.
        reportError(ERR_SYNTAX, opByteNum, NULL);
    c = getchar();
    if (isalpha(c) || isDigit(c)) // Po operatorze NEW/DEL nie moze byc litery/cyfry.
        reportError(ERR_SYNTAX, opByteNum, NULL);
    ungetc(c, stdin);
}

/**
 * @brief Zwalnia pamięc zaalokowana przez program.
**/
static void memFree() {
    free(num1);
    free(num2);
    free(id);
    delPfBase(pfBase);
}
/**
 * Wczytuje i wykonuje operacje. W przypadku napotkania błędu wypisuje komunikat i kończy program z wartoscią 1.
 * @return @p 0 jeśli program zakończył się pomyślnie @p 1 wpp.
 */
int main() {
    atexit(memFree);
    pfBase = newPfDB();
    PfDB *currBase = NULL;
    long long int sizeNum1 = 10, sizeNum2 = 10, sizeId = 10;
    if (!(num1 = malloc(sizeNum1 * sizeof(char))))
        reportError(ERR_MALLOC, 0, NULL);
    if (!(num2 = malloc(sizeNum2 * sizeof(char))))
        reportError(ERR_MALLOC, 0, NULL);
    if (!(id = malloc(sizeId * sizeof(char))))
        reportError(ERR_MALLOC, 0, NULL);
    char op[MAX_OP_LEN]; op[0] = '\0';

    while (1) {
        int c = getCharSkip(false);
        if (c == EOF)
            break;
        if (isDigit(c)) {
            ungetChar(c);
            readNum(&num1, &sizeNum1);
            readOp(op);
            if (op[0] == '?') { // num1 ?
                if (!currBase)
                    reportError(ERR_OP, opByteNum, "?");
                struct PhoneNumbers const *pnum = phfwdGet(currBase->pf, num1);
                if (pnum->size == 0)
                    reportError(ERR_OP, opByteNum, "?");
                printf("%s\n", phnumGet(pnum, 0));
                phnumDelete(pnum);
                continue;
            } else if (op[0] == '>') { // num1 > num2
                readNum(&num2, &sizeNum2);
                if (!currBase)
                    reportError(ERR_OP, opByteNum, ">");
                if (!phfwdAdd(currBase->pf, num1, num2))
                    reportError(ERR_OP, opByteNum, ">");
                continue;
            }
            byteCount = opByteNum; // Wczytano niewlasciwa operacje. Bład nastąpił na pierwszym znaku operacji.
        } else {
            ungetChar(c);
            readOp(op);
            if (op[0] == '?') { // ? num1
                readNum(&num1, &sizeNum1);
                if (!currBase)
                    reportError(ERR_OP, opByteNum, "?");
                struct PhoneNumbers const *pnum = phfwdReverse(currBase->pf, num1);
                size_t idx = 0;
                const char *num;
                while ((num = phnumGet(pnum, idx++)) != NULL)
                    printf("%s\n", num);
                phnumDelete(pnum);
                continue;
            } else if (op[0] == '@') {
                readNum(&num1, &sizeNum1);
                if (!currBase)
                    reportError(ERR_OP, opByteNum, "@");
                size_t num1Len = strlen(num1);
                size_t len = num1Len > 12 ? num1Len - 12 : 0;
                printf("%zu\n", phfwdNonTrivialCount(currBase->pf, num1, len));
                continue;
            } else if (strcmp(op, "NEW") == 0) { // NEW id
                readId(&id, &sizeId);
                addPfToDb(id, &currBase);
                continue;
            } else if (strcmp(op, "DEL") == 0) {
                c = getCharSkip(true);
                if (isDigit(c)) { // DEL num
                    ungetChar(c);
                    readNum(&num1, &sizeNum1);
                    if(!currBase)
                        reportError(ERR_OP, opByteNum, "DEL");
                    phfwdRemove(currBase->pf, num1);
                    continue;
                } else if (isalpha(c)) { // DEL id
                    ungetChar(c);
                    readId(&id, &sizeId);
                    if (currBase && strcmp(currBase->id, id) == 0) // Usuwanie aktualnej bazy.
                        currBase = NULL;
                    if (!delPfDB(id))
                        reportError(ERR_OP, opByteNum, "DEL");
                    continue;
                }
            }
        }
        reportError(ERR_SYNTAX, byteCount, NULL);
    }
    return 0;
}