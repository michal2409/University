### Opis

Biblioteka standardowa języka C++ udostępnia bardzo przydatne kontenery
(np. unordered_map i set), których nie ma w bibliotece C. Często też
potrzebujemy łączyć kod w C++ z kodem w C.

Celem tego zadania jest napisanie w C++ dwóch modułów obsługujących zbiory
ciągów znaków, tak aby można ich było używać w C. Każdy moduł składa się z pliku
nagłówkowego (z rozszerzeniem h) i pliku z implementacją (z rozszerzeniem cc).

Moduł strset (pliki strset.h i strset.cc) powinien udostępniać następujące
funkcje:

**unsigned long strset_new();**

      Tworzy nowy zbiór i zwraca jego identyfikator.

**void strset_delete(unsigned long id);**

      Jeżeli istnieje zbiór o identyfikatorze id, usuwa go, a w przeciwnym
      przypadku nie robi nic.

**size_t strset_size(unsigned long id);**

      Jeżeli istnieje zbiór o identyfikatorze id, zwraca liczbę jego elementów,
      a w przeciwnym przypadku zwraca 0.

**void strset_insert(unsigned long id, const char* value);**

      Jeżeli istnieje zbiór o identyfikatorze id i element value nie należy do
      tego zbioru, to dodaje element do zbioru, a w przeciwnym przypadku nie
      robi nic.

**void strset_remove(unsigned long id, const char* value);**

      Jeżeli istnieje zbiór o identyfikatorze id i element value należy do tego
      zbioru, to usuwa element ze zbioru, a w przeciwnym przypadku nie robi nic.

**int strset_test(unsigned long id, const char* value);**

      Jeżeli istnieje zbiór o identyfikatorze id i element value należy do tego
      zbioru, to zwraca 1, a w przeciwnym przypadku 0.

**void strset_clear(unsigned long id);**

      Jeżeli istnieje zbiór o identyfikatorze id, usuwa wszystkie jego elementy,
      a w przeciwnym przypadku nie robi nic.

**int strset_comp(unsigned long id1, unsigned long id2);**

      Porównuje zbiory o identyfikatorach id1 i id2. Niech sorted(id) oznacza
      posortowany leksykograficznie zbiór o identyfikatorze id. Takie ciągi już
      porównujemy naturalnie: pierwsze miejsce, na którym się różnią, decyduje
      o relacji większości. Jeśli jeden ciąg jest prefiksem drugiego, to ten
      będący prefiks jest mniejszy. Funkcja strset_comp(id1, id2) powinna zwrócić
      -1, gdy sorted(id1) < sorted(id2),
      0, gdy sorted(id1) = sorted(id2),
      1, gdy sorted(id1) > sorted(id2).
      Jeżeli zbiór o którymś z identyfikatorów nie istnieje, to jest traktowany
      jako równy zbiorowi pustemu.

Moduł strsetconst (pliki strsetconst.h i strsetconst.cc) powinien udostępniać
funkcję:

**unsigned long strset42();**

    Zwraca identyfikator zbioru, którego nie można modyfikować i który zawiera
    jeden element: napis "42". Zbiór jest tworzony przy pierwszym wywołaniu tej
    funkcji i wtedy zostaje ustalony jego numer.

Należy ukryć przed światem zewnętrznym wszystkie zmienne globalne i funkcje
pomocnicze nie należące do wyspecyfikowanych interfejsów modułów.

Moduły strset i strsetconst powinny wypisywać na standardowy strumień błędów
informacje diagnostyczne. Moduły te mogą sprawdzać poprawność wykonania funkcji
za pomocą asercji. Kompilowanie z parametrem -DNDEBUG powinno wyłączać
wypisywanie i asercje. Obsługa standardowego wyjścia diagnostycznego powinna być
realizowana z użyciem strumienia C++ (tzn. iostream).

Parametr value o wartości NULL jest niepoprawny.

Oczekiwane rozwiązanie powinno korzystać z kontenerów i metod udostępnianych
przez standardową bibliotekę C++. Nie należy definiować własnych struktur lub
klas. W szczególności nie należy przechowywać przekazanych przez użytkownika
wskaźników const char* bezpośrednio, bowiem użytkownik może po wykonaniu
operacji modyfikować dane pod uprzednio przekazanym wskaźnikiem lub zwolnić
pamięć. Na przykład poniższy kod nie powinien przerwać się z powodu
niespełnionej asercji:

    unsigned long s;
    char buf[4] = "foo";
    s = strset_new();
    strset_insert(s, buf);
    buf[0] = 'b';
    assert(strset_test(s, "foo"));
    assert(!strset_test(s, "boo"));

W rozwiązaniu nie należy nadużywać kompilacji warunkowej. Fragmenty tekstu
źródłowego realizujące wyspecyfikowane operacje na zbiorach nie powinny zależeć
od sposobu kompilowania – parametr -DNDEBUG lub jego brak (inaczej wersja
diagnostyczna nie miałaby sensu).

Przykład użycia znajduje się w pliku strset_test1.c. Przykład informacji
diagnostycznych wypisywanych przez powyższy przykład użycia znajduje się
w pliku strset_test1.err.

Aby umożliwić używanie modułów strset oraz strsetconst w języku C++, przy
kompilowaniu plików nagłówkowych strset.h i strsetconst.h w C++ interfejsy
modułów strset i strsetconst mają znaleźć się w przestrzeni nazw jnp1.
Przykłady użycia znajdują się w plikach strset_test2a.cc i strset_test2b.cc.
Przykład informacji diagnostycznych wypisywanych przez powyższe przykłady
użycia znajduje się w plikach strset_test2a.err i strset_test2b.err.

Przykłady można skompilować za pomocą poleceń:

gcc -Wall -Wextra -O2 -std=c11 -c strset_test1.c -o strset_test1.o
g++ -Wall -Wextra -O2 -std=c++17 -c strset.cc -o strset.o
g++ -Wall -Wextra -O2 -std=c++17 -c strsetconst.cc -o strsetconst.o
g++ strset_test1.o strsetconst.o strset.o -o strset1
g++ -Wall -Wextra -O2 -std=c++17 -c strset_test2a.cc -o strset_test2a.o
g++ -Wall -Wextra -O2 -std=c++17 -c strset_test2b.cc -o strset_test2b.o
g++ strset_test2a.o strsetconst.o strset.o -o strset2a
g++ strset_test2b.o strsetconst.o strset.o -o strset2b

Rozwiązanie powinno zawierać pliki strset.h, strset.cc, strsetconst.h,
strsetconst.cc, które należy umieścić w repozytorium w katalogu

grupaN/zadanie2/ab123456+cd123456

lub

grupaN/zadanie2/ab123456+cd123456+ef123456

gdzie N jest numerem grupy, a ab123456, cd123456, ef123456 są identyfikatorami
członków zespołu umieszczającego to rozwiązanie. Katalog z rozwiązaniem nie
powinien zawierać innych plików, ale może zawierać podkatalog prywatne, gdzie
można umieszczać różne pliki, np. swoje testy. Pliki umieszczone w tym
podkatalogu nie będą oceniane. Nie wolno umieszczać w repozytorium plików
dużych, binarnych, tymczasowych (np. *.o) ani innych zbędnych.