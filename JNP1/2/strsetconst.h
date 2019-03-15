#ifndef STRSET_STRSETCONST_H
#define STRSET_STRSETCONST_H

#ifdef __cplusplus
#include <iostream>
namespace jnp1 {
	extern "C" {
#endif
	/*
	Zwraca identyfikator zbioru, ktorego nie mozna modyfikowac i ktory zawiera
	jeden element: napis "42". Zbior jest tworzony przy pierwszym wywolaniu tej
	funkcji i wtedy zostaje ustalony jego numer.
	*/
	unsigned long strset42();

#ifdef __cplusplus
	}
}
#endif

#endif //STRSET_STRSETCONST_H.