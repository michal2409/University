#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <stdbool.h>

int stos[4000];
int top = -1;

void dodaj_na_stos(int adr_powr) {
	stos[++top] = adr_powr;
}

int zdejmij_ze_stosu() {
	return (top > -1) ? stos[top--] : -1;
}

// struktura Etykieta zawiera pola z nazwa etykiety oraz
// pole z numerem instrukcji oznaczonej tej etykiety
struct Etykieta {
	char * nazwa;
	int nr;
} etykiety[1000];
int ile_etyk = 0;

// znajduje nr etykiety o danej nazwie, jesli takiej nie ma
// to dodaje ja na koncu tablicy etykiety i zwraca ile_etyk
int znajdz_idx_etyk(char * nazwa_etyk) {
	for (int i = 0; i < ile_etyk; i++)
		if (!strcmp(etykiety[i].nazwa, nazwa_etyk))
			return i;
	etykiety[ile_etyk].nazwa = nazwa_etyk;
	return ile_etyk++;
}

// struktura Insturkcja zawiera pole z informacja o jaka instrukcje chodzi
// oraz dwa pola z adresami, ktore sa wykorzystywane (byc moze tylko jeden z dwoch)
// jako parametry do wykonania instrukcji
enum Instr {odejmowanie, skok, pisanie, czytanie, wywolanie, powrot};
struct Instrukcja {
	enum Instr nazwa;
	int adres1;
	int adres2;
} instrukcje[3000];
int ile_instr = 0;

// wczytanie adresu gdy program napotka znak + - lub cyfre
// argumentem jest pierwszy wczytany znak, wynikiem wczytany adres
int wczyt_adr(int znak) {
	char adres[12];
	int i = 1;
	adres[0] = znak;
	while (isdigit(znak = getchar()))
		adres[i++] = znak;
	adres[i] = '\0';
	ungetc(znak, stdin);
	return atoi(adres);
}

// wczytanie etykiety gdy program napotka literę lub '_'
// arguementem jest pierwszy wczytany znak, wynikiem wczytana etykieta
char * wczyt_etyk(int znak) {
	char * ident = (char *) malloc(sizeof(char) * 2001);
	int i = 1;
	ident[0] = znak;
	znak = getchar();
	while(isalpha(znak) || isdigit(znak) || znak == '_') {
		ident[i++] = znak;
		znak = getchar();
	}
	ident[i] = '\0';
	ungetc(znak, stdin);
	return ident;
}

// dodaje instrukcje do tablicy instrukcje
void dodaj_instrukcje(enum Instr instr, int adres1, int adres2) {
	instrukcje[ile_instr].nazwa = instr;
	instrukcje[ile_instr].adres1 = adres1;
	instrukcje[ile_instr].adres2 = adres2;
	ile_instr++;
}

// sprawdza czy napotkany znak jest separatorem
bool jest_sep(int znak) {
	return (znak == ' ' || znak == '\t' || znak == '\n' || znak == '|');
}

// funkcja czyta program wejsciowy oraz rozpoznaje kolejne instrukcje i
// zapisuje je razem z argumentami do tablicy instrukcje, gdy napotka znak
// '&' oznaczajacy start danych wejsciowych przerywa czytanie programu
void wczytaj_program() {
	int znak;
	while ((znak = getchar()) != EOF) {
		if (!jest_sep(znak)) { // pomijamy separatory
			if (isdigit(znak) || znak == '-' || znak == '+') { // intrukcja typu (adres coś)
				int adres1 = wczyt_adr(znak);
				while (jest_sep(znak = getchar()));

				if (isdigit(znak) || znak == '-' || znak == '+') { // (adres adres) odejmowanie
					int adres2 = wczyt_adr(znak);
					dodaj_instrukcje(odejmowanie, adres1, adres2);
				}
				else if (znak == '^'){ // (adres ^) pisanie
					dodaj_instrukcje(pisanie, adres1, 0);
				}
				else if (isalpha(znak) || znak == '_') { // (adres, etykieta) skok
					char * nazwa_etyk = wczyt_etyk(znak);
					// ile_instr + 1 to adres powrotu, znajdz_idx_etyk(nazwa_etyk) to index etykiety w tablicy etykiety
					dodaj_instrukcje(skok, adres1, znajdz_idx_etyk(nazwa_etyk));
				}
			}
			else {
				if (isalpha(znak) || znak == '_') { // (etykieta) wywolanie
					char * nazwa_etyk = wczyt_etyk(znak);
					// ile_instr + 1 to adres powrotu, znajdz_idx_etyk(nazwa_etyk) to index etykiety w tablicy etykiety
					dodaj_instrukcje(wywolanie, ile_instr + 1, znajdz_idx_etyk(nazwa_etyk));
				}
				else if (znak == '^') { // (^ adres) czytanie
					while (jest_sep(znak = getchar()));
					int adres1 = wczyt_adr(znak);
					dodaj_instrukcje(czytanie, adres1, 0);
				}
				else if (znak == ':') { // definicja
					while (jest_sep(znak = getchar()));
					char * nazwa_etyk = wczyt_etyk(znak);
					etykiety[znajdz_idx_etyk(nazwa_etyk)].nr = ile_instr;
				}
				else if (znak == ';')  // powrot
					dodaj_instrukcje(powrot, 0, 0);

				else if (znak == '&')  // poczatek danych wejsciowych, konczymy czytac wejscie
					return;
			}
		}
	}
}
// komorki z adresami od - 5000 do 4999
int pamiec[10000];
// inicjalizacja pamieci o adresach -5000 do 4999
void init_pam(){
	for (int i = 0; i < 10000; i++)
		pamiec[i] = 4999 - i;
}

// odczytanie wartosci pamieci dla adresow mieszczacych sie w zakresie typu int
int odczyt_pam(int adres) {
	return (-5000 <= adres && adres <= 4999) ? pamiec[5000 + adres] : - 1 - adres;
}

// wykonuje instrukcje odejmowania dla podanych dwoch adresow
void odejmij(int adres1, int adres2) {
	int adres1_2 = odczyt_pam(odczyt_pam(adres1));
	int adres2_2 = odczyt_pam(odczyt_pam(adres2));
	int adres1_1 = 5000 + odczyt_pam(adres1);
	pamiec[adres1_1] = adres1_2 - adres2_2;
}

// wykonuje instrukcje czytania dla podanego adresu
void czytaj(int adres) {
	int adres_1 = 5000 + odczyt_pam(adres);
	pamiec[adres_1] = getchar();
}

// wykonuje instrukcje pisania dla podanego adresu
void pisz(int adres){
	putchar(odczyt_pam(odczyt_pam(adres)));
}
// funkcja wykonuje wczytany program, zaczyna od wykonania instrukcje[0]
// po czym wykonuje kolejne instrukcje
void wykonaj_program() {
	int idx = 0;
	while(idx < ile_instr) {
		switch (instrukcje[idx].nazwa) {
			case odejmowanie: {
				odejmij(instrukcje[idx].adres1, instrukcje[idx].adres2);
				idx++;
			}; break;
			case pisanie: {
				pisz(instrukcje[idx].adres1);
				idx++;
			}; break;
			case czytanie: {
				czytaj(instrukcje[idx].adres1);
				idx++;
			}; break;
			case skok: {
				if (odczyt_pam(odczyt_pam(instrukcje[idx].adres1)) > 0) // spr czy **adres > 0
					idx = etykiety[instrukcje[idx].adres2].nr; // skok do instr oznaczonej etykieta
				else
					idx++; // jesli **adres <= 0 to idziemy dalej
			}; break;
			case wywolanie: {
				dodaj_na_stos(instrukcje[idx].adres1);	// dodajemy na stos miejsce powrotu
				idx = etykiety[instrukcje[idx].adres2].nr; // skok do instr oznaczonej etykieta
			}; break;
			case powrot: {
				idx = zdejmij_ze_stosu(); // zdejmujemy adres powrotu ze stosu i skaczemy do niego
				if (idx == -1) // stos byl pusty konczymy program
					return;
			}; break;
		}
	}
}

int main(void) {
	init_pam();
	wczytaj_program();
	wykonaj_program();
    return 0;
}
