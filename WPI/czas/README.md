<h1 id="wprowadzenie">Wprowadzenie</h1>
<p><em>Czas</em> to język programowania abstrakcyjnej maszyny <em>Czas</em>.</p>
<h2 id="składnia-języka">Składnia języka</h2>
<p>Składnię języka Czas opisuje gramatyka, której symbolami końcowymi są</p>
<ul>
<li><p><code>identyfikator</code>, zapisany jak w języku C,</p></li>
<li><p><code>liczba</code>, reprezentująca liczbę całkowitą w zapisie dziesiętnym, poprzedzoną opcjonalnym znakiem <code>+</code> lub <code>-</code>,</p></li>
<li><p>znaki ujęte w apostrofy.</p></li>
</ul>
<p>W kodzie źródłowym programu, przed lub po, ale nie wewnątrz, tekstowych reprezentacji symboli końcowych gramatyki języka, mogą wystąpić, w dowolnej liczbie, spacje, tabulacje, końce wiersza i znaki <code>|</code>. Nazywane są separatorami i nie mają wpływu na interpretację programu.</p>
<p>Symbolem początkowym gramatyki języka Czas jest <code>Program</code>.</p>
<pre><code>

Program → ε | Element Program

Element → Definicja | Instrukcja

Definicja → ':' Etykieta

Instrukcja → Odejmowanie | Skok | Wywołanie | Powrót | Czytanie | Pisanie

Odejmowanie → Adres Adres

Skok → Adres Etykieta

Wywołanie → Etykieta

Powrót → ';'

Czytanie → '^' Adres

Pisanie → Adres '^'

Etykieta → identyfikator

Adres → liczba</code></pre>
<p>Etykiety reprezentują miejsca w kodzie. Każda etykieta w programie musi wystąpić w <code>Definicji</code>. O instrukcji bezpośrednio za definicją etykiety powiemy, że jest tą etykietą oznaczona. Program, w którym są dwie definicje etykiety o tym samym identyfikatorze, jest błędny.</p>
<p>Adresy reprezentują komórki pamięci.</p>
<h2 id="pamięć-maszyny">Pamięć maszyny</h2>
<p>Pamięć maszyny jest nieograniczona. Adresami i wartościami jej komórek mogą być dowolne liczby całkowite ze znakiem. Początkowa wartość komórki o adresie <code>a</code> jest równa <code>-1 - a</code>. W opisie semantyki instrukcji wartość komórki o adresie <code>adres</code> oznaczamy przez <code>*adres</code>.</p>
<p>Maszyna ma też <em>stos powrotu</em>, który przechowuje informacje o miejscach w kodzie. Udostępnia on operacje włożenia na stos nowej informacji oraz zdjęcia ze stosu ostatniej położonej informacji, która nie została jeszcze zdjęta.</p>
<h2 id="wykonanie-programu">Wykonanie programu</h2>
<p>Pusty program nie ma żadnego efektu.</p>
<p>Wykonanie programu niepustego zaczyna się od pierwszej instrukcji. Kończy się albo po przejściu za ostatnią instrukcję albo po wykonaniu instrukcji kończącej program.</p>
<p>Po instrukcji aktualnej wykonywana jest albo instrukcja wskazana przez instrukcję aktualną albo instrukcja następna po niej w kodzie.</p>
<h2 id="instrukcje-maszyny">Instrukcje maszyny</h2>
<p>Maszyna Czas ma sześć instrukcji.</p>
<ul>
<li><p>Instrukcja odejmowania <code>adres1 adres2</code> zapisuje w komórce o adresie <code>*adres1</code> wynik odejmowania <code>**adres2</code> od <code>**adres1</code>.</p>
<p>Jeśli np. przed wykonaniem instrukcji <code>3 1</code> było <code>*1 == 10</code>, <code>*3 == 30</code>, <code>*10 == 100</code> i <code>*30 == 300</code> to po jej wykonaniu będzie <code>*30 == 200</code>.</p></li>
<li><p>Instrukcja skoku <code>adres etykieta</code> sprawdza, czy <code>**adres</code> jest większe od zera. Jeśli tak, to następną wykonaną instrukcją będzie instrukcja oznaczona <code>etykietą</code>.</p></li>
<li><p>Instrukcja wywołania <code>etykieta</code> wskazuje, że następną wykonaną instrukcją ma być instrukcja oznaczona <code>etykietą</code>. Jednocześnie na stos powrotu wkładane jest miejsce w kodzie bezpośrednio za instrukcją aktualną.</p></li>
<li><p>Instrukcja powrotu <code>;</code> wskazuje, że następną wykonaną instrukcją ma być instrukcja w miejscu, które zdejmujemy ze stosu powrotu. Jeśli stos jest pusty, instrukcja powrotu kończy program.</p></li>
<li><p>Instrukcja czytania <code>^ adres</code> zapisuje w komórce o adresie <code>*adres</code> kod znaku wczytanego za pomocą funkcji <code>getchar()</code> języka C lub -1, jeśli wynikiem <code>getchar()</code> było <code>EOF</code>.</p></li>
<li><p>Instrukcja pisania <code>adres ^</code> pisze, za pomocą funkcji <code>putchar()</code> języka C, znak o kodzie <code>**adres</code>.</p></li>
</ul>
<h1 id="polecenie">Polecenie</h1>
<p>Napisz interpreter, czyli program, który wykona program w języku Czas.</p>
<p>Na wejściu interpretera jest kod źródłowy programu, po którym, opcjonalnie, może być znak <code>&amp;</code> i dane dla interpretowanego programu.</p>
<p>Wynikiem pracy interpretera jest wynik interpretowanego programu.</p>
<h1 id="przykłady">Przykłady</h1>
<p>Do treści zadania dołączone są pliki <code>.czs</code> z przykładowymi danymi i pliki <code>.out</code> z wynikami wzorcowymi.</p>
<ul>
<li><p>Dla danych <a href="hello.czs">hello.czs</a> polecenie</p>
<pre class="sourceCode bash"><code class="sourceCode bash"><span class="kw">&lt;</span> <span class="kw">hello.czs</span> ./czas <span class="kw">&gt;</span> hello.out</code></pre>
<p>utworzy plik <a href="hello.out">hello.out</a>.</p></li>
<li><p>Dla danych <a href="hal.czs">hal.czs</a> polecenie</p>
<pre class="sourceCode bash"><code class="sourceCode bash"><span class="kw">echo</span> <span class="st">"&amp;HAL"</span> <span class="kw">|</span> <span class="kw">cat</span> hal.czs - <span class="kw">|</span> <span class="kw">./czas</span> <span class="kw">&gt;</span> hal.out</code></pre>
<p>utworzy plik <a href="hal.out">hal.out</a>.</p></li>
<li><p>Dla danych <a href="rekursja.czs">rekursja.czs</a> polecenie</p>
<pre class="sourceCode bash"><code class="sourceCode bash"><span class="kw">&lt;</span> <span class="kw">rekursja.czs</span> ./czas <span class="kw">&gt;</span> rekursja.out</code></pre>
<p>utworzy plik <a href="rekursja.out">rekursja.out</a>.</p></li>
</ul>
<h1 id="walidacja-i-testy">Walidacja i testy</h1>
<ul>
<li><p>Rozwiązania zostaną poddane walidacji, wstępnie sprawdzającej zgodność ze specyfikacją. Pomyślne przejście walidacji jest warunkiem dopuszczenia programu do testów poprawności.</p></li>
<li><p>Programy będą kompilowane poleceniem</p>
<pre><code>gcc -std=c11 -pedantic -Wall -Wextra -Werror nazwa.c -o nazwa</code></pre>
<p>Wymagane są wszystkie wymienione opcje kompilatora. Nie będą do nich dodawane żadne inne.</p></li>
<li><p>Przyjmujemy, że wynik funkcji <code>main()</code> inny niż <code>0</code> informuje o błędzie wykonania programu.</p></li>
<li><p>Poprawność wyniku sprawdzamy, przekierowując na wejście programu zawartość pliku z danymi i porównując rezultat, za pomocą programu <code>diff</code>, z wynikiem wzorcowym, np.</p>
<pre><code>&lt; przyklad.in ./nazwa | diff - przyklad.out</code></pre>
<p>Ocena poprawności wyniku jest binarna. Uznajemy go za poprawny, jeżeli program <code>diff</code> nie wskaże żadnej różnicy względem wyniku wzorcowego.</p></li>
</ul>
<h1 id="założenia">Założenia</h1>
<p>Wolno założyć, że</p>
<ul>
<li><p>koszt algorytmu wyszukiwania etykiety nie będzie miał wpływu na ocenę rozwiązania,</p></li>
<li><p>dane, czyli kod źródłowy programu w języku Czas, są poprawne,</p></li>
<li><p>liczby w kodzie programu, oraz wyniki obliczeń tego programu, mieszczą się w zakresie typu <code>int</code>,</p></li>
<li><p>znaki na wejściu interpretowanego programu będą miały kody od 0 do 127,</p></li>
<li><p>liczba różnych etykiet nie przekracza 1000,</p></li>
<li><p>suma długości wszystkich etykiet, bez powtórzeń, nie przekracza 2000,</p></li>
<li><p>program ma nie więcej niż 3000 instrukcji,</p></li>
<li><p>liczba wartości na stosie powrotu nie przekroczy 4000,</p></li>
<li><p>podczas wykonania interpretowanego programu, wartości będą zapisywane tylko do komórek pamięci o adresach z przedziału od -5000 do 4999.</p>
<p>Odczytanie wartości komórki powinno być jednak możliwe dla dowolnych adresów mieszczących się w zakresie typu <code>int</code>.</p></li>
