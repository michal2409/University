#ifndef STRSET_STRSET_H
#define STRSET_STRSET_H

#ifdef __cplusplus
#include <cstddef>
#include <iostream>
namespace jnp1 {
	extern "C" {
#else
#include <stddef.h>
#endif
	/*
	Tworzy nowy zbior i zwraca jego identyfikator.
	*/
	unsigned long strset_new();

	/*
	Jezeli istnieje zbior o identyfikatorze id, usuwa go, a w przeciwnym
	przypadku nie robi nic.
	*/
	void strset_delete(unsigned long id);

	/*
	Jezeli istnieje zbior o identyfikatorze id, zwraca liczbe jego elementow,
	a w przeciwnym przypadku zwraca 0.
	*/
	size_t strset_size(unsigned long id);

	/*
	Jezeli istnieje zbior o identyfikatorze id i element value nie nalezy do
	tego zbioru, to dodaje element do zbioru, a w przeciwnym przypadku nie
	robi nic.
	*/
	void strset_insert(unsigned long id, const char* value);

	/*
	Jezeli istnieje zbior o identyfikatorze id i element value nalezy do tego
	zbioru, to usuwa element ze zbioru, a w przeciwnym przypadku nie robi nic.
	*/
	void strset_remove(unsigned long id, const char* value);

	/*
	Jezeli istnieje zbior o identyfikatorze id i element value nalezy do tego
	zbioru, to zwraca 1, a w przeciwnym przypadku 0.
	*/
	int strset_test(unsigned long id, const char* value);

	/*
	Jezeli istnieje zbior o identyfikatorze id, usuwa wszystkie jego elementy,
	a w przeciwnym przypadku nie robi nic.
	*/
	void strset_clear(unsigned long id);

	/*
	Porownuje zbiory o identyfikatorach id1 i id2. Niech sorted(id) oznacza
	posortowany leksykograficznie zbior o identyfikatorze id. Takie ciagi juz
	porownujemy naturalnie: pierwsze miejsce, na ktorym sie roznia, decyduje
	o relacji wiekszosci. Jesli jeden ciag jest prefiksem drugiego, to ten
	bedacy prefiks jest mniejszy. Funkcja strset_comp(id1, id2) powinna zwrocic
	-1, gdy sorted(id1) < sorted(id2),
	0, gdy sorted(id1) = sorted(id2),
	1, gdy sorted(id1) > sorted(id2).
	Jezeli zbior o ktoryms z identyfikatorow nie istnieje, to jest traktowany
	jako rowny zbiorowi pustemu.
	*/
	int strset_comp(unsigned long id1, unsigned long id2);

#ifdef __cplusplus
	}
}
#endif

#endif //STRSET_STRSET_H