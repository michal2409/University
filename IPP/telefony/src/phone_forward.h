/** @file
 * Interfejs klasy przechowującej przekierowania numerów telefonicznych
 *
 * @author Michał Futrega <michal.futrega@student.uw.edu.pl>
 * @copyright Uniwersytet Warszawski
 * @date 09.04.2018
 */

#ifndef __PHONE_FORWARD_H__
#define __PHONE_FORWARD_H__
#define DIGIT_COUNT 12 ///< Liczba możliwych cyfr w numerze telefonu.

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

/**
 * Lista jednokierunkowa przechowująca numery telefonów.
 */
typedef struct ListNode{
    char *num; ///< Wskaźnik na napis reprezentujący numer telefonu.
    struct ListNode *next; ///< Wskaźnik na następny element listy.
} ListNode;

/**
 * Drzewo trie przechowujące informację o przekierowaniach.
 */
typedef struct TreeNode {
    char *fwdFrom; ///< Wskaźnik na napis reprezentujący przekierowywany prefiks.
    ListNode *fwdToList; ///< Wskaźnik na listę przechowującą numery na które przekierowywany jest prefiks 'fwdFrom'.
    struct TreeNode *child[DIGIT_COUNT]; ///< Tablica wskaźników na synów węzła. Indeks tablicy odpowiada kolejnej cyfrze numeru.
} TreeNode;

/**
 * Struktura przechowująca przekierowania numerów telefonów.
 */
struct PhoneForward {
    TreeNode *fwdTree; ///< Drzewo trie przechowujące informację o przekierowaniach.
    TreeNode *inverseFwdTree; ///< Drzewo trie przechowujące informację o przekierowaniach odwrotnych.
};

/**
 * Struktura przechowująca ciąg numerów telefonów.
 */
struct PhoneNumbers {
    char **phnums; ///< Tablica napisów reprezentujące numery telefonów.
    size_t size; ///< Rozmiar tablicy `phnums`.
};

/** @brief Tworzy nową strukturę.
 * Tworzy nową strukturę niezawierającą żadnych przekierowań.
 * @return Wskaźnik na utworzoną strukturę lub NULL, gdy nie udało się
 *         zaalokować pamięci.
 */
struct PhoneForward * phfwdNew(void);

/** @brief Usuwa strukturę.
 * Usuwa strukturę wskazywaną przez @p pf. Nic nie robi, jeśli wskaźnik ten ma
 * wartość NULL.
 * @param[in] pf – wskaźnik na usuwaną strukturę.
 */
void phfwdDelete(struct PhoneForward *pf);

/** @brief Dodaje przekierowanie.
 * Dodaje przekierowanie wszystkich numerów mających prefiks @p num1, na numery,
 * w których ten prefiks zamieniono odpowiednio na prefiks @p num2. Każdy numer
 * jest swoim własnym prefiksem. Jeśli wcześniej zostało dodane przekierowanie
 * z takim samym parametrem @p num1, to jest ono zastępowane.
 * @param[in, out] pf  – wskaźnik na strukturę przechowującą przekierowania numerów;
 * @param[in] num1     – wskaźnik na napis reprezentujący prefiks numerów
 *                       przekierowywanych;
 * @param[in] num2     – wskaźnik na napis reprezentujący prefiks numerów, na które
 *                       jest wykonywane przekierowanie.
 * @return Wartość @p true, jeśli przekierowanie zostało dodane.
 *         Wartość @p false, jeśli wystąpił błąd, np. podany napis nie
 *         reprezentuje numeru, oba podane numery są identyczne lub nie udało
 *         się zaalokować pamięci.
 */
bool phfwdAdd(struct PhoneForward *pf, char const *num1, char const *num2);

/** @brief Usuwa przekierowania.
 * Usuwa wszystkie przekierowania, w których parametr @p num jest prefiksem
 * parametru @p num1 użytego przy dodawaniu. Jeśli nie ma takich przekierowań
 * lub napis nie reprezentuje numeru, nic nie robi.
 *
 * @param[in, out] pf  – wskaźnik na strukturę przechowującą przekierowania numerów;
 * @param[in] num      – wskaźnik na napis reprezentujący prefiks numerów.
 */
void phfwdRemove(struct PhoneForward *pf, char const *num);

/** @brief Wyznacza przekierowanie numeru.
 * Wyznacza przekierowanie podanego numeru. Szuka najdłuższego pasującego
 * prefiksu. Wynikiem jest co najwyżej jeden numer. Jeśli dany numer nie został
 * przekierowany, to wynikiem jest ten numer. Jeśli podany napis nie
 * reprezentuje numeru, wynikiem jest pusty ciąg. Alokuje strukturę
 * @p PhoneNumbers, która musi być zwolniona za pomocą funkcji @ref phnumDelete.
 * @param[in] pf  – wskaźnik na strukturę przechowującą przekierowania numerów;
 * @param[in] num – wskaźnik na napis reprezentujący numer.
 * @return Wskaźnik na strukturę przechowującą ciąg numerów lub NULL, gdy nie
 *         udało się zaalokować pamięci.
 */
struct PhoneNumbers const * phfwdGet(struct PhoneForward *pf, char const *num);

/** @brief Wyznacza przekierowania na dany numer.
 * Wyznacza wszystkie przekierowania na podany numer. Wynikowy ciąg zawiera też
 * dany numer. Wynikowe numery są posortowane leksykograficznie i nie mogą się
 * powtarzać. Jeśli podany napis nie reprezentuje numeru, wynikiem jest pusty
 * ciąg. Alokuje strukturę @p PhoneNumbers, która musi być zwolniona za pomocą
 * funkcji @ref phnumDelete.
 * @param[in] pf  – wskaźnik na strukturę przechowującą przekierowania numerów;
 * @param[in] num – wskaźnik na napis reprezentujący numer.
 * @return Wskaźnik na strukturę przechowującą ciąg numerów lub NULL, gdy nie
 *         udało się zaalokować pamięci.
 */
struct PhoneNumbers const * phfwdReverse(struct PhoneForward *pf, char const *num);

/** @brief Wyznacza liczbę nietrywialnych numerów.
 * Oblicza liczbę nietrywialnych numerów długości @param len zawierających tylko cyfry,
 * które znajdują się w napisie @param set Jeśli wskaźnik @param pf ma
 * wartość NULL, @param set ma wartość NULL lub jest pusty lub nie zawiera żadnej
 * cyfry lub parametr @param len jest równy zeru, wynikiem jest zero.
 * @param[in] pf  - wskaźnik na strukturę przechowującą przekierowania numerów;
 * @param[in] set - wskaźnik na napis zawierający dozwolone cyfry;
 * @param[in] len - długość dozwolonych numerów.
 * @return Liczba nietrywialnych numerów modulo dwa do potęgi liczba bitów
 * reprezentacji typu size_t.
 */
size_t phfwdNonTrivialCount(struct PhoneForward *pf, char const *set, size_t len);

/** @brief Usuwa strukturę.
 * Usuwa strukturę wskazywaną przez @p pnum. Nic nie robi, jeśli wskaźnik ten ma
 * wartość NULL.
 * @param[in] pnum – wskaźnik na usuwaną strukturę.
 */
void phnumDelete(struct PhoneNumbers const *pnum);

/** @brief Udostępnia numer.
 * Udostępnia wskaźnik na napis reprezentujący numer. Napisy są indeksowane
 * kolejno od zera.
 * @param[in] pnum – wskaźnik na strukturę przechowującą ciąg napisów;
 * @param[in] idx  – indeks napisu.
 * @return Wskaźnik na napis. Wartość NULL, jeśli wskaźnik @p pnum ma wartość
 *         NULL lub indeks ma za dużą wartość.
 */
char const * phnumGet(struct PhoneNumbers const *pnum, size_t idx);

#endif /* __PHONE_FORWARD_H__ */
