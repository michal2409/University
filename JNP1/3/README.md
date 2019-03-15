### Opis

BajtekCoin to cyfrowa waluta stworzona w 2018 roku przez Bajtka. Jej skrót to B.
„Bajtkomonety” mogą zostać zapisane w portfelu (klasie Wallet). Całkowita liczba
bajtkomonet wynosi co najwyżej 21 milionów. Bajtkomonety są jednak podzielne do
8 miejsc po przecinku, czyli 1 B = 1e8 jednostek (co razem daje 2,1e15
jednostek).

W ramach tego zadania należy zaimplementować portfel (klasa Wallet) oraz
historię operacji na nim.

Niech w0, w1, w2 będą obiektami klasy Wallet. Powinny być dostępne następujące
operacje:

**Wallet w0**

    Tworzy pusty portfel. Historia portfela ma jeden wpis.

**Wallet w0(n)**

    Tworzy portfel z n B, gdzie n jest liczbą naturalną. Historia portfela ma
    jeden wpis.

**Wallet w0(str)**

    Tworzy portfel na podstawie napisu str określającego ilość B. Napis może
    zawierać część ułamkową (do 8 miejsc po przecinku). Część ułamkowa jest
    oddzielona przecinkiem lub kropką. Białe znaki na początku i końcu napisu
    powinny być ignorowane. Historia portfela ma jeden wpis.

**Wallet w1(Wallet &&w2)**

    Konstruktor przenoszący. Historia operacji w1 to historia operacji w2
    i jeden nowy wpis.

**Wallet w0(Wallet &&w1, Wallet &&w2)**

    Tworzy portfel, którego historia operacji to suma historii operacji w1
    i w2 plus jeden wpis, całość uporządkowana wg czasów wpisów. Po operacji
    w w0 jest w1.getUnits() + w2.getUnits() jednostek, a portfele w1 i w2 są
    puste.

**Wallet::fromBinary(str)**

    Metoda klasowa tworząca portfel na podstawie napisu str, który jest zapisem
    ilości B w systemie binarnym. Kolejność bajtów jest grubokońcówkowa
    (ang. big endian).

Powyższe metody tworzą jeden wpis w historii operacji tworzonego obiektu.
Żeby nie można było kopiować monet, nie udostępniamy dla klasy Wallet
konstruktora kopiującego.

**w1 = w2**

    Przypisanie. Jeżeli oba obiekty są tym samym obiektem, to nic nie robi, wpp.
    historia operacji w1 to historia operacji w2 i jeden nowy wpis. Dostępne
    jest tylko przypisanie przenoszące, nie przypisanie kopiujące, np.

    Wallet w1, w2;
    w1 = Wallet(1); // OK
    w1 = w2; // błąd kompilacji

**w1 + w2**

    Dodawanie, np.

    Wallet w1(1), w2(2);
    Wallet suma1 = w1 + Wallet(1); // błąd kompilacji
    Wallet suma2 = Wallet(2) + w2; // OK, w w2 jest 0 B po operacji
                                   // i jeden nowy wpis w historii,
                                   // a w suma2 jest w2.getUnits() + 2 B.
                                   // Historia operacji powstałego obiektu
                                   // zależy od implementacji.
    Wallet suma3 = suma1 + suma2;  // błąd kompilacji
    Wallet suma4 = Wallet(1) + Wallet(2);  // OK, suma4 ma dwa wpisy
                                           // w historii i 3 B

**w1 - w2**

    Odejmowanie, analogicznie jak dodawanie, ale po odejmowaniu w w2 jest dwa
    razy więcej jednostek, niż było w w2 przed odejmowaniem.
    Historia operacji powstałego obiektu zależy od implementacji.

**w0 * n**
**n * w0**

    Pomnożenie przez liczbę naturalną.
    Historia operacji powstałego obiektu zależy od implementacji.

**w1 += w2**

    Po operacji w2 ma 0 B i dodatkowy wpis w historii, a w1 ma
    w1.getUnits() + w2.getUnits() jednostek i jeden dodatkowy wpis w historii.

**w1 -= w2**

    Analogicznie do dodawania.

**w1 *= n**

    Pomnożenie zawartości portfela przez liczbę naturalną.
    Dodaje jeden wpis w historii w1.

**w1 op w2**

    Operatory porównujące wartości portfeli w1 i w2, gdzie op to jeden z:
    ==, <, <=, != , >, >=.

**os << w0**

    Wypisuje "Wallet[b B]" na strumień os, gdzie b to zawartość portfela w B.
    Wypisywana liczba jest bez białych znaków, bez zer wiodących oraz zer na
    końcu w rozwinięciu dziesiętnym oraz z przecinkiem jako separatorem
    dziesiętnym.

**w0.getUnits()**

    Zwraca liczbę jednostek w portfelu.

**w0.opSize()**

    Zwraca liczbę operacji wykonanych na portfelu.

**w0[k]**

    Zwraca k-tą operację na portfelu. Pod indeksem 0 powinna być najstarsza
    operacja. Przypisanie do w[k] powinno być zabronione na etapie kompilacji.

Należy zaimplementować również globalną funkcję Empty(), która zwróci obiekt
reprezentujący pusty portfel. Modyfikowanie zwróconego obiektu powinno być
zabronione. W szczególności konstrukcja "Empty() += Wallet(1);" powinna zostać
zgłoszone jako błąd kompilacji.

Operatory powinny działać również z udziałem argumentów będących liczbami
naturalnymi, ale nie powinny działać z argumentami będącymi napisami i liczbami
zmiennoprzecinkowymi. W szczególności nie powinny się kompilować konstrukcje
typu:

    Wallet w1, w2;
    bool b
    float f;
    Wallet w3(true);
    Wallet w4('a');
    Wallet w5(f);
    w1 += "10";
    w1 = w2 + "10";
    b = "10" < w2;
    itp.

Natomiast poprawne są następujące konstrukcje:

    w1 += Wallet(3);
    w1 *= 3;
    b = 2 < w2;
    Wallet suma2 = 2 + w2
    itp.

Wartość portfela nie może być nigdy mniejsza od 0. Gdyby w wyniku którejś
operacji wartość portfela spadła poniżej 0, to należy rzucić wyjątek.

Podczas tworzenia nowych portfeli i operacji na nich należy zadbać, aby liczba
wszystkich B nie przekroczyła 21 milionów. W przypadku przekroczenia limitu,
należy rzucić wyjątek i uznać, że te B nie istnieją. Naturalnie nie należy
zliczać B, których nie ma już w portfelach.

Niech o0, o1, o2 reprezentują operacje na portfelu. Powinny być dostępne
następujące operacje:

o0.getUnits()

    Zwraca liczbę jednostek w portfelu po operacji.

o1 op o2

    Operatory porównujące czas utworzenia (z dokładnością do milisekund)
    operacji o1 i o2, gdzie op to jeden z: ==, <, <=, != , >, >=.

os << op

    Wypisuje na strumień os "Wallet balance is b B after operation made at day d".
    Liczba b jak przy wypisywaniu portfela. Czas d w formacie yyyy-mm-dd.

Wszystkie operatory, metody i funkcje powinny przyjmować argumenty oraz
generować wyniki, których typy są zgodne z ogólnie przyjętymi konwencjami
w zakresie używania referencji, wartości typu const i obiektów statycznych,
chyba że treść zadania wyraźnie wskazuje inaczej.

Przykładowy kod, demonstrujący użycie klasy Wallet i klasy reprezentującej
operację na portfelu, znajduje się w pliku wallet_example.cc.

Rozwiązanie będzie kompilowane poleceniem

g++ -Wall -Wextra -O2 -std=c++17 -c wallet.cc

Rozwiązanie powinno składać się z plików wallet.h oraz wallet.cc. Pliki te
należy umieścić w repozytorium w katalogu

grupaN/zadanie3/ab123456+cd123456

lub

grupaN/zadanie3/ab123456+cd123456+ef123456

gdzie N jest numerem grupy, a ab123456, cd123456, ef123456 są identyfikatorami
członków zespołu umieszczającego to rozwiązanie. Katalog z rozwiązaniem nie
powinien zawierać innych plików, ale może zawierać podkatalog prywatne, gdzie
można umieszczać różne pliki, np. swoje testy. Pliki umieszczone w tym
podkatalogu nie będą oceniane. Nie wolno umieszczać w repozytorium plików
dużych, binarnych, tymczasowych (np. *.o) ani innych zbędnych.