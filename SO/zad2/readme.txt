Zadanie 2

Termin oddania: 01.04.2019, godz. 20.00

Zaimplementuj w asemblerze x86_64 moduł symulujący działanie sieci euronowej.
Sieć składa się z N euronów, które są numerowane od 0 do N − 1. Moduł będzie
używany z języka C i ma udostępniać funkcję widzianą jako:

uint64_t euron(uint64_t n, char const *prog);

Eurony działają równolegle – każdy euron jest uruchamiany w osobnym wątku.
Parametr n zawiera numer euronu. Parametr prog jest wskaźnikiem na napis ASCIIZ
i opisuje obliczenie, jakie ma wykonać euron. Obliczenie składa się z operacji
wykonywanych na stosie, który na początku jest pusty. Znaki napisu
interpretujemy następująco ('X' to kod ASCII znaku X):

'+' – zdejmij dwie wartości ze stosu, oblicz ich sumę i wstaw wynik na stos;

'*' – zdejmij dwie wartości ze stosu, oblicz ich iloczyn i wstaw wynik na stos;

'-' – zaneguj arytmetycznie wartość na wierzchołku stosu;

'0' do '9' – wstaw na stos odpowiednio liczbę 0 do 9;

'n' – wstaw na stos numer euronu;

'B' – zdejmij wartość ze stosu, jeśli teraz na wierzchołku stosu jest wartość
      różna od zera, potraktuj zdjętą wartość jako liczbę w kodzie
      uzupełnieniowym do dwójki i przesuń się o tyle operacji;

'C' – zdejmij wartość ze stosu;

'D' – wstaw na stos wartość z wierzchołka stosu, czyli zduplikuj wartość na
      wierzchu stosu;

'E' – zamień miejscami dwie wartości na wierzchu stosu;

'G' – wstaw na stos wartość uzyskaną z wywołania (zaimplementowanej gdzieś
      indziej w języku C) funkcji

      uint64_t get_value(uint64_t n);

'P' – zdejmij wartość ze stosu (oznaczmy ją przez w) i wywołaj (zaimplementowaną
      gdzieś indziej w języku C) funkcję

      void put_value(uint64_t n, uint64_t w);

'S' – zdejmij wartość ze stosu, potraktuj ją jako numer euronu m, czekaj na
      operację 'S' euronu m ze zdjętym ze stosu numerem euronu n i zamień
      wartości na wierzchołkach stosów euronów m i n.

Po zakończeniu przez euron wykonywania obliczenia jego wynikiem, czyli wynikiem
funkcji euron, jest wartość z wierzchołka stosu. Wszystkie operacje wykonywane
są na liczbach 64-bitowych modulo 2^64. Zakładamy, że obliczenie jest poprawne,
tzn. zawiera tylko opisane wyżej znaki, kończy się zerowym bajtem, nie próbuje
sięgać po wartość z pustego stosu i nie doprowadza do zakleszczenia. Zachowanie
euronu dla niepoprawnego obliczenia jest niezdefiniowane.

Przykład poprawnego obliczenia dla N = 2 (jako napis w języku C):

"01234n+P56789E-+D+*G*1n-+S2ED+E1-+75+-BC"

Po wykonaniu tego obliczenia przez oba eurony, dla zdefiniowanych tak funkcji:

uint64_t get_value(uint64_t n) {
  assert(n < N);
  return n + 1;
}

void put_value(uint64_t n, uint64_t v) {
  assert(n < N);
  assert(v == n + 4);
}

wynikiem obliczenia euronu 0 jest 112, a wynikiem obliczenia euronu 1 jest 56.

Jako stosu, którego euron używa do opisanych wyżej obliczeń, należy użyć
sprzętowego stosu procesora. Nie wolno korzystać z żadnych bibliotek.
Synchronizację wątków należy zaimplementować za pomocą jakiegoś wariantu
wirującej blokady. Rozwiązanie będzie asemblowane na maszynie students
poleceniem:

nasm -DN=XXX -f elf64 -o euron.o euron.asm

gdzie XXX określa wartość parametru N.

Tekst źródłowy rozwiązania należy umieścić w pliku euron.asm w repozytorium SVN
w katalogu https://svn.mimuw.edu.pl/repos/SO/studenci/login/zadanie2, gdzie
login to identyfikator używany do logowania w laboratorium. W katalogu
z rozwiązaniem nie wolno umieszczać żadnych innych plików.

Zadanie nie wymaga napisania dużego kodu. Kod maszynowy w pliku euron.asm nie
powinien zajmować więcej niż ok. czterysta bajtów. Jednak rozwiązanie powinno
być przemyślane i dobrze przetestowane. Nie udostępniamy naszych testów, więc
przetestowanie rozwiązania jest częścią zadania, choć nie wymagamy pokazywania
tych testów. W szczególności na potrzeby testowania należy zaimplementować
własne funkcję get_value i put_value, ale nie należy ich implementacji dołączać
do rozwiązania.

Rozwiązanie zostanie poddane testom automatycznym. Będziemy sprawdzać poprawność
wykonywania obliczenia. Dokładnie będziemy też sprawdzać zgodność rozwiązania
z wymaganiami ABI, czyli prawidłowość użycia rejestrów i stosu procesora.
Oceniane będą poprawność i jakość tekstu źródłowego, w tym komentarzy, rozmiar
kodu maszynowego oraz spełnienie formalnych wymagań podanych w treści zadania,
np. poprawność nazwy pliku w repozytorium. Kod nieasemblujący się otrzyma 0
punktów.

Pytania do zadania można kierować na adres marpe@mimuw.edu.pl z [SOzad2]
w temacie, a odpowiedzi na często zadawane pytania szukać w pliku faq.txt.
