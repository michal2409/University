<h2>Zadanie maraton filmowy</h2><div id="intro" class="box generalbox boxaligncenter"><div class="no-overflow"><p>Celem zadania jest napisanie programu umożliwiającego grupom fanów kina
wybieranie filmów do maratonu filmowego.
Każdy kinomaniak (użytkownik) może wprowadzać preferowane filmy, które chciałby
obejrzeć w trakcie maratonu, może je także usuwać.
Użytkownik identyfikowany jest za pomocą identyfikatora liczbowego.
Identyfikatory użytkowników są unikalne.
Film identyfikowany jest za pomocą identyfikatora liczbowego.
Identyfikatory filmów są unikalne.
Użytkownicy tworzą drzewo.
Każdy już zapisany użytkownik może dopisywać nowych użytkowników, jako potomków
swojego wierzchołka drzewa.
Na początku istnieje użytkownik o identyfikatorze 0 (korzeń drzewa), niemający
żadnych preferencji filmowych.
Każdy użytkownik z wyjątkiem użytkownika o identyfikatorze 0 może siebie
wypisać.
Identyfikator filmu jest też jego arbitralnie ustaloną oceną.</p>

<p>Rozwiązanie powinno korzystać z dynamicznie alokowanych struktur danych.
Implementacja powinna być jak najbardziej efektywna.
Należy unikać zbędnego alokowania pamięci i kopiowania danych.
Przykładowe dane dla programu znajdują się w załączonym pliku.</p>

<h2>Operacje</h2>

<p>Program ma wykonywać następujące operacje:</p>

<ul>
<li><p><code>addUser &lt;parentUserId&gt; &lt;userId&gt;</code> – Użytkownik o identyfikatorze
<code>parentUserId</code> dodaje użytkownika o identyfikatorze <code>userId</code>.
Operacja ma się wykonywać w czasie stałym.</p></li>
<li><p><code>delUser &lt;userId&gt;</code> – Użytkownik o identyfikatorze <code>userId</code> wypisuje się.
Dodane przez niego preferencje filmowe są zapominane.
Użytkownicy uprzednio dodani przez użytkownika <code>userId</code> stają się potomkami
rodzica użytkownika <code>userId</code>.
Usunięcie użytkownika ma się wykonywać w czasie stałym.
Zapominanie preferencji filmowych ma się wykonywać w czasie liniowym
względem liczby preferencji usuwanego użytkownika.</p></li>
<li><p><code>addMovie &lt;userId&gt; &lt;movieRating&gt;</code> – Użytkownik o identyfikatorze <code>userId</code>
dodaje film o identyfikatorze <code>movieRating</code> do swoich preferencji filmowych.
Operacja ma się wykonywać w czasie co najwyżej liniowym względem liczby
preferencji użytkownika, który dodaje film.</p></li>
<li><p><code>delMovie &lt;userId&gt; &lt;movieRating&gt;</code> – Użytkownik o identyfikatorze <code>userId</code>
usuwa film o identyfikatorze <code>movieRating</code> ze swoich preferencji filmowych.
Operacja ma się wykonywać w czasie co najwyżej liniowym względem liczby
preferencji użytkownika, który usuwa film.</p></li>
<li><p><code>marathon &lt;userId&gt; &lt;k&gt;</code> – Wyznacza co najwyżej <code>k</code> identyfikatorów filmów o
najwyższych ocenach spośród:</p>

<ul>
<li><p>własnych preferencji filmowych użytkownika o identyfikatorze <code>userId</code>;</p></li>
<li><p>preferencji filmowych wyodrębnionych w wyniku przeprowadzenia operacji
<code>marathon</code> dla każdego z potomków użytkownika <code>userId</code>, przy czym
w wynikowej grupie <code>k</code> filmów znajdą się tylko takie, które mają ocenę
większą od maksymalnej oceny filmu spośród preferencji użytkownika
<code>userId</code>.</p></li>
</ul>

<p>Operacja ma się wykonywać w czasie liniowym względem iloczynu parametru <code>k</code>
i liczby użytkowników, dla których rekurencyjnie wyliczana jest ta operacja.</p></li>
</ul>

<h2>Dane wejściowe</h2>

<p>Program powinien czytać ze standardowego wejścia.
Można przyjąć następujące założenia o danych wejściowych:</p>

<ul>
<li>Parametry <code>&lt;userID&gt;</code> i <code>&lt;parentUserID&gt;</code> są liczbami całkowitymi z przedziału
od 0 do 65535.</li>
<li>Parametry <code>&lt;movieRating&gt;</code> oraz <code>&lt;k&gt;</code> są liczbami całkowitymi z przedziału od
0 do 2147483647.</li>
<li>Nazwa polecenia i liczby są oddzielone pojedynczą spacją, a każdy wiersz
wejścia kończy się linuksowym znakiem końca linii (znak <code>\n</code> w C, kod ASCII
10). Są to jedyne białe znaki występujące w danych wejściowych.</li>
</ul>

<h2>Informacje wypisywane przez program</h2>

<p>Program wypisuje na standardowe wyjście:</p>

<ul>
<li>Dla każdej operacji innej niż <code>marathon</code> wiersz ze słowem <code>OK</code>.</li>
<li>Dla operacji <code>marathon</code> wiersz zawierający posortowane
malejąco wyznaczone oceny, a w przypadku braku filmów spełniających warunki
maratonu wiersz zawierający jedynie słowo <code>NONE</code>.
Oceny powinny być oddzielone pojedynczą spacją.
Na końcu wiersza nie może być spacji.</li>
<li>Każdy wiersz wyjścia powinien kończyć się linuksowym znakiem końca linii
(znak <code>\n</code> w C, kod ASCII 10).</li>
</ul>

<h2>Obsługa błędów</h2>

<p>Program wypisuje informacje o błędach na standardowe wyjście diagnostyczne.</p>

<ul>
<li>Puste wiersze należy ignorować.</li>
<li>Wiersze rozpoczynające się znakiem <code>#</code> należy ignorować.</li>
<li>Dla każdego błędnego wiersza i dla każdej operacji, która nie może być
wykonana np. z powodu błędnej wartości parametru, należy wypisać wiersz ze
słowem <code>ERROR</code>, zakończony linuksowym znakiem końca linii (znak <code>\n</code> w C,
kod ASCII 10).</li>
</ul>

<h2>Zakończenie programu</h2>

<p>Program kończy się po przetworzeniu wszystkich operacji z wejścia i powinien
wtedy zakończyć się kodem 0.
Awaryjne zakończenie programu, np. na skutek niemożliwości zaalokowania
potrzebnej pamięci, powinno być sygnalizowane kodem 1.
Przed zakończeniem program powinien zwolnić całą zaalokowaną pamięć.</p>

<h1>Makefile</h1>

<p>Częścią zadania jest napisanie pliku <code>makefile</code>.
W wyniku wywołania polecenia <code>make</code> powinien powstać program wykonywalny <code>main</code>.
Jeśli któryś z plików źródłowych ulegnie zmianie, ponowne wpisanie <code>make</code>
powinno na nowo stworzyć plik wykonywalny.
Plik <code>makefile</code> powinien działać w następujący sposób:</p>

<ul>
<li>osobno kompiluje każdy plik <code>.c</code>,</li>
<li>osobno linkuje wszystkie pliki <code>.o</code>,</li>
<li>przy zmianie w pliku <code>.c</code> lub <code>.h</code> wykonuje tylko niezbędne akcje,</li>
<li>wywołanie <code>make clean</code> usuwa plik wykonywalny i dodatkowe pliki powstałe
podczas kompilowania.</li>
</ul>

<p>Pliki należy kompilować z opcjami:</p>

<pre><code>-Wall -Wextra -std=c11 -O2
</code></pre>

<h2>Skrypt testujący</h2>

<p>Osobną częścią zadania jest napisanie skryptu <code>test.sh</code>.
Po wywołaniu</p>

<pre><code class="sh">./test.sh &lt;prog&gt; &lt;dir&gt;
</code></pre>

<p>skrypt powinien uruchomić program <code>&lt;prog&gt;</code> dla wszystkich plików wejściowych
postaci <code>&lt;dir&gt;/*.in</code>, porównać wyniki z odpowiadającymi im plikami
<code>&lt;dir&gt;/*.out</code> i <code>&lt;dir&gt;/*.err</code>, a następnie wypisać, które testy zakończyły się
powodzeniem, a które niepowodzeniem.
Do wykrywania problemów z zarządzaniem pamięcią należy użyć programu <code>valgrind</code>.</p>

<h2>Podział na pliki</h2>

<p>Rozwiązanie powinno zawierać następujące pliki:</p>

<ul>
<li><code>main.c</code> – Główny plik programu, w którym wczytuje się dane wejście,
wywołuje operacje na strukturach danych i wypisuje wyniki działania
programu.
Plik ten nie powinien znać szczegółów definicji i implementacji użytych
struktur danych.</li>
<li><code>x.c</code>, <code>x.h</code> – Implementacja modułu (struktury danych) <code>x</code>. Plik <code>x.h</code>
zawiera deklaracje operacji modułu <code>x</code>, a plik <code>x.c</code> – ich implementację.
Jako <code>x</code> należy użyć stosownej nazwy modułu (struktury danych), np. <code>tree</code>
itp.
Powinno być tyle par plików <code>x.c</code>, <code>x.h</code>, ile jest w rozwiązaniu modułów
(struktur danych).</li>
<li><code>makefile</code> – Patrz punkt „makefile”.</li>
<li><code>test.sh</code> – Patrz punkt „skrypt testujący”.</li>
</ul>

<p>Rozwiązanie należy oddać jako archiwum skompresowane programem <code>zip</code> lub parą
programów <code>tar</code> i <code>gz</code>.</p>

<h2>Punktacja</h2>

<p>Za w pełni poprawne rozwiązanie zadania implementujące wszystkie funkcjonalności
można zdobyć maksymalnie 20 punktów.
Rozwiązanie niekompilujące się lub nie oparte na dynamicznie alokowanych
strukturach danych będzie ocenione na 0 punktów.
Możliwe są punkty karne za poniższe uchybienia:</p>

<ul>
<li>Za każdy test, którego program nie przejdzie, traci się 1 punkt.</li>
<li>Za problemy z zarządzaniem pamięcią można stracić do 6 punktów.</li>
<li>Za niezgodną ze specyfikacją strukturę plików w rozwiązaniu można stracić do
4 punktów.</li>
<li>Za złą złożoność operacji można stracić do 4 punktów.</li>
<li>Za błędy stylu kodowania można stracić do 4 punktów.</li>
<li>Za brak lub źle działający <code>makefile</code> można stracić do 2 punktów.</li>
<li>Za brak skryptu testującego lub błędy w skrypcie testującym można stracić do
2 punktów.</li>
</ul>
