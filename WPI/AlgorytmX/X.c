#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#define ROZMIAR_POCZATKOWY 100
#define MNOZNIK 3

char ** dane_wejsciowe;
int l_wier = 0;
int l_kol = 0;

void wczytaj_dane () { // wczytanie kolejnych wierszy do dane_wejsciowe
    int rozm_danych_wej = ROZMIAR_POCZATKOWY, dl_linii = 0, l_wczyt_wier = 0;
    char *wczytana_linia = NULL;
    size_t dl = 0;
    dane_wejsciowe = malloc(rozm_danych_wej * sizeof(*dane_wejsciowe));
    while ((dl_linii = getline(&wczytana_linia, &dl, stdin)) != -1) {
        if (l_wczyt_wier + 1 == rozm_danych_wej) { // przepelnienie bufora, zwiekszenie rozmiaru dane_wejsciowe
            rozm_danych_wej *= MNOZNIK;
            dane_wejsciowe = realloc(dane_wejsciowe, rozm_danych_wej * sizeof(*dane_wejsciowe));
        }
        dane_wejsciowe[l_wczyt_wier] = malloc((dl_linii + 1) * sizeof(char));
        strcpy(dane_wejsciowe[l_wczyt_wier++], wczytana_linia);
    }
    if (wczytana_linia)
        free(wczytana_linia);
    if (l_wczyt_wier > 0) { // aktualizacja liczby wczytanych wierszy i kolumn
        l_wier = l_wczyt_wier;
        l_kol = strlen(dane_wejsciowe[0])-1;
    }
}

// sortowanie dane_wejsciowe ze względu na pierwsze wystąpienie znaku '_' w kolumnie
// dla posortowanej macierzy wykonanie rekurencji wyswietla pokrycia w dobrej kolejnosci
void sortuj () {
    int l_posort_wier = 1;
    for (int kol = 0; kol < l_kol; kol++)
        for (int wier = l_posort_wier; wier < l_wier; wier++)
            if (dane_wejsciowe[wier][kol] != '_') {
                for (int i = 0; i < wier - l_posort_wier; i++) {
                    char pom[l_kol+1];
                    strcpy(pom, dane_wejsciowe[l_posort_wier+i]);
                    strcpy(dane_wejsciowe[l_posort_wier+i], dane_wejsciowe[wier]);
                    strcpy(dane_wejsciowe[wier], pom);
                }
                l_posort_wier++;
            }
}

// sprawdza czy wszystkie miejsca w tablicy pokryte zostaly juz pokryte
bool spr_czy_pokr (bool pokr[]) {
    for (int i = 0; i < l_kol; i++)
        if (!pokr[i])
            return false;
    return true;
}

// sprawdza czy dany wiersz mozna dodac do pokrycia
bool spr_czy_mozna_dodac (int wiersz, bool pokr[]) {
    for (int kol = 0; kol < l_kol; kol++)
        if (dane_wejsciowe[wiersz][kol] != '_' && pokr[kol]) // to miejsce zostalo wczesniej pokryte
            return false;
    return true;
}

// dodaje dany wiersz do pokrycia zaznaczajac dodane miejsca
void dodaj_pokr (int wier, bool pokr[]) {
    for (int i = 0; i < l_kol; i++)
        if (dane_wejsciowe[wier][i] != '_')
            pokr[i] = true;
}

// usuwa z tablicy pokryte miejsca odpowiadajace wierszowi wier
void usun_pokr (int wier, bool pokr[]) {
    for (int i = 0; i < l_kol; i++)
        if (dane_wejsciowe[wier][i] != '_')
            pokr[i] = false;
}

// dla danej tablicy wierszy wypisuje odpowiadajace im pokrycie uwzgledniajac filtr
void wypisz_pokrycie (int wier_pokr[], int l_pokr_wier) {
    char pokrycie[l_kol];
    for (int i = 0; i < l_pokr_wier; i++) // dla kazdego wiersza z pokrycia
        for (int kol = 0; kol < l_kol; kol++) // dodajemy znaki rozne od '_'
            if (dane_wejsciowe[wier_pokr[i]][kol] != '_') // do tablicy pokrycie
                pokrycie[kol] = dane_wejsciowe[wier_pokr[i]][kol];

    for (int kol = 0; kol < l_kol; kol++) // uwzglednienie filtra przy wypisywaniu
        if (dane_wejsciowe[0][kol] != '-')
            printf("%c", pokrycie[kol]);
    printf("\n");
}

// głowna funkcja znajdujaca wszystkie pokrycia dla danych wejsciowych
// tworzy tablice z numerami wierszy ktore skladaja sie na pokrycie
void znajdz_pokrycia(bool pokr[], int wier_pokr[], int l_pokr_wier, int pocz_wier) {
    if (spr_czy_pokr(pokr))
        wypisz_pokrycie(wier_pokr, l_pokr_wier);
    else {
        for (int wier = pocz_wier; wier < l_wier; wier++)
            if (spr_czy_mozna_dodac(wier, pokr)) {
                wier_pokr[l_pokr_wier] = wier;
                dodaj_pokr(wier, pokr);
                znajdz_pokrycia(pokr, wier_pokr, l_pokr_wier + 1, wier + 1);
            }
    }
    if (l_pokr_wier >= 1)
        usun_pokr(wier_pokr[l_pokr_wier - 1], pokr);
}

void zwolnij_pamiec() {
    if (dane_wejsciowe) {
        for (int i = 0; i < l_wier; i++)
            if (dane_wejsciowe[i])
                free(dane_wejsciowe[i]);
        free(dane_wejsciowe);
    }
}

int main() {
    wczytaj_dane();
    sortuj();
    if (l_wier > 0 && l_kol > 0) { // sprawdzenie czy dane wejsciowe nie sa puste
        int wier_pokr[l_kol];
        bool pokr[l_kol];
        for (int i = 0; i < l_kol; i++)
            pokr[i] = false;
        znajdz_pokrycia(pokr, wier_pokr, 0, 1);
    }
    zwolnij_pamiec();
    return 0;
}
