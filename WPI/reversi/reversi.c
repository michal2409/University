#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define WIERSZE 8
#define KOLUMNY 8

char plansza[WIERSZE][KOLUMNY];

/*
Zwraca liczbe zdobytych pionkow w ruchu na pole (kolumna, wiersz) w zadanym kierunku, np.
dla kier_k = 1, kier_w = 0 poruszamy się w poziomie w prawą stronę,
dla kier_k = 1, kier_w = -1 po skosie w prawy gorny rog.
Ponadto jezeli podany ruch jest legalny i zmienna akt_plansze == true to aktualizuje plansze.
*/

int licz_zdobyte_pionki(char kolor, int kolumna, int wiersz, int kier_k, int kier_w, bool akt_plansze) {
	int k, w, ile_zamian = 0;
	char kolor_p = 'B' + 'C' - kolor;

	k = kolumna + kier_k;
	w = wiersz + kier_w;

	// Dopoki jestesmy w planszy, na polu o przeciwnym kolorze poruszamy sie
	// w zadanym kierunku
	while (0 <= k && k < KOLUMNY && 0 <= w && w < WIERSZE && plansza[w][k] == kolor_p) {
		k += kier_k;
		w += kier_w;
		ile_zamian++;
	}
	// Jezeli nie wyszlismy z planszy i trafilismy na nasz kolor
	if (0 <= w && w < WIERSZE && 0 <= k && k < KOLUMNY && plansza[w][k] == kolor) {
		// Aktualizacja pionkow w podanym kierunku po wykonanym ruchu
		if (akt_plansze)
			for (int i = 1; i <= ile_zamian; i++)
				plansza[wiersz + i * kier_w][kolumna + i * kier_k] = kolor;
	}
	else
		ile_zamian = 0;
	return ile_zamian;
}

/*
Zwraca liczbe zdobytych pionow we wszystkich mozliwych kierunkach.
Parametr akt_plansze okresla czy wykonywany ruch zostanie zapisany na planszy.
*/

int wykonaj_ruch(char kolor, char kolumna, int wiersz, bool akt_plansze) {
	int k = kolumna - 'a', ile_zamian = 0;

	if (0 <= k && k < KOLUMNY && 0 <= wiersz && wiersz < WIERSZE && plansza[wiersz][k] == '-') {
		// Liczymy ile zdobylismy pionkow w ruchu (kolumna, wiersz)
		for (int y = -1; y <= 1; y++)
			for (int x = -1; x <= 1; x++)
				if(y != 0 || x != 0)
					ile_zamian += licz_zdobyte_pionki(kolor, k, wiersz, y, x, akt_plansze);

		if (akt_plansze && ile_zamian > 0)
			plansza[wiersz][k] = kolor;
	}
	return ile_zamian;
}

/*
Dla danej planszy i koloru gra ruch, w ktorym zdobywa sie najwieksza liczbe pionkow.
Gdyby wiele ruchów to maksimum osiągało, wybiera ruch na pole o niższym numerze
wiersza a w ramach tego wiersza, pole we wcześniejszej kolumnie.
*/

void wygeneruj_naj_ruch(char kolor, char ruch[3]) {
	int l_zamian, wiersz, max_zamian = 0;
	char kolumna;

	// Szukamy ruchu dajacego najwieksza liczbe zamian.
	for (int w = 0; w < WIERSZE; w++) {
		for (char k = 'a'; k <= 'a' + (KOLUMNY - 1); k++) {
			l_zamian = wykonaj_ruch(kolor, k, w, false);
			if (l_zamian > max_zamian) {
				max_zamian = l_zamian;
				kolumna = k;
				wiersz = w;
			}
		}
	}
	// Jezeli mamy ruch to go wykonujemy i przekazujemy ruch przez argument.
	if (max_zamian > 0) {
		wykonaj_ruch(kolor, kolumna, wiersz, true);
		ruch[0] = kolumna;
		ruch[1] = '1' + wiersz;
		ruch[2] = '\0';
	}
	else {
		ruch[0] = '=';
		ruch[1] = '\0';
	}
}

void inicjalizuj_plansze() {
	for (int w = 0; w < WIERSZE; w++)
		for (int k = 0; k < KOLUMNY; k++)
			plansza[w][k] = '-';

	int srodek_w = WIERSZE / 2;
	int srodek_k = KOLUMNY / 2;
	plansza[srodek_w - 1][srodek_k - 1] = plansza[srodek_w][srodek_k] = 'B';
	plansza[srodek_w - 1][srodek_k] = plansza[srodek_w][srodek_k - 1] = 'C';
}

void rysuj_plansze() {
	for (int w = 0; w < WIERSZE; w++) {
		for (int k = 0; k < KOLUMNY; k++)
			printf("%c", plansza[w][k]);
		printf("%d\n", w+1);
	}
	for (char k = 'a'; k <= 'a' + (KOLUMNY - 1); k++)
		printf("%c", k);
	printf("\n");
}

int ocena_planszy() {
	int biale = 0, czarne = 0;

	for (int w = 0; w < WIERSZE; w++) {
		for (int k = 0; k < KOLUMNY; k++)
			if (plansza[w][k] == 'C')
				czarne++;
			else if (plansza[w][k] == 'B')
				biale++;
	}
	return czarne - biale;
}

int main(void) {
	int dl_linii;
	char *linia = NULL;
	size_t dl = 0;
	inicjalizuj_plansze();
	rysuj_plansze();

	while ((dl_linii = getline(&linia, &dl, stdin)) != -1) {
		dl_linii--;
		// Wygenerowany ruch
		char graczB_ruch[3];
		if (dl_linii == 1 && linia[0] == '=') { // Przeciwnik spasowal
			wygeneruj_naj_ruch('B', graczB_ruch);
			printf("= %s %d\n", graczB_ruch, ocena_planszy());
		}
		else {
			int zamiany = 0, graczC_wier = -1;
			char graczC_kol = '0';

			if(dl_linii == 2) {
				// Wczytany ruch
				graczC_kol = linia[0];
				graczC_wier = linia[1] - '1';
			}

			if ('a' <= graczC_kol && graczC_kol <= 'a' + (KOLUMNY - 1) && 0 <= graczC_wier && graczC_wier < WIERSZE)
				zamiany = wykonaj_ruch('C', graczC_kol, graczC_wier, true);

			if (zamiany) { // Legalny ruch
				wygeneruj_naj_ruch('B', graczB_ruch);
				printf("%c%d %s %d\n", graczC_kol, graczC_wier + 1, graczB_ruch, ocena_planszy());
			}
			else
				printf("? %d\n", ocena_planszy());
		}
		rysuj_plansze();
	}
	if (linia != NULL)
		free(linia);
	return 0;
}
