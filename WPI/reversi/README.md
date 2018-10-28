<h1 id="wprowadzenie">Wprowadzenie</h1>
<p>Gra <a href="https://en.wikipedia.org/wiki/Reversi"><em>Reversi</em></a>, znana też pod nazwą <em>Othello</em>, jest rozgrywana na <code>64</code>-polowej planszy podzielonej na wiersze od <code>1</code> do <code>8</code> i kolumny od <code>a</code> do <code>h</code>. Pola nazywamy, wskazując najpierw kolumnę a następnie wiersz.</p>
<p>W Reversi gra się dwustronnymi czarno-białymi pionami. Na każdym polu może być co najwyżej jeden pion. Jeśli pion w danej chwili jest odwrócony do góry stroną czarną, nazywamy go pionem czarnym, jeśli białą - jest pionem białym.</p>
<p>Partię gry w Reversi zaczyna się na planszy z pionami białymi na polach <code>d4</code> i <code>e5</code> oraz czarnymi na polach <code>d5</code> i <code>e4</code>. Pozostałe pola są puste.</p>
<p>W grze bierze udział dwóch graczy, nazywanych <em>Czarnym</em> i <em>Białym</em>, od koloru pionów, którym się posługują. Grę rozpoczyna gracz Czarny.</p>
<p>Gracze wykonują, na przemian, po jednym ruchu polegającym na umieszczeniu na planszy piona swojego koloru. Jeśli na wszystkich polach w <em>linii</em>, czyli w wierszu, kolumnie lub przekątnej, między położonym właśnie pionem a innym pionem tego samego koloru, są piony w kolorze przeciwnym, zmieniają one kolor, czyli są odwracane. Położenie na planszy jednego piona może spowodować jednoczesną zmianę kilku linii pionów.</p>
<p>Ruch jest legalny tylko, gdy powoduje zmianę koloru co najmniej jednego piona na planszy. Jeśli w danej chwili gracz nie może wykonać legalnego ruchu, nie kładzie na planszy piona.</p>
<p>Choć nie jest to zgodne z regułami gry w Reversi, w tym zadaniu pozwalamy graczowi zrezygnować z ruchu nawet, gdy może wykonać ruch legalny.</p>
<p>Gra kończy się, gdy żaden z graczy nie może wykonać legalnego ruchu. Zwycięzcą zostaje gracz, który ma na planszy więcej pionów swojego koloru.</p>
<h1 id="polecenie">Polecenie</h1>
<p>Napisz program grający w Reversi jako gracz Biały.</p>
<p>Program pisze diagram początkowego stanu planszy. Następnie, w pętli, czyta kolejne polecenia gracza Czarnego, odpowiada na nie jako gracz Biały i pisze nowy diagram planszy.</p>
<p>Spośród ruchów gracza Białego program wybiera ten, po którym liczba białych pionów na planszy będzie największa. Gdyby wiele ruchów to maksimum osiągało, wybiera ruch na pole o niższym numerze wiersza a w ramach tego wiersza, pole we wcześniejszej kolumnie.</p>
<h1 id="postać-danych">Postać danych</h1>
<p>W kolejnych wierszach wejścia są polecenia użytkownika grającego jako gracz Czarny. Polecenie może być poprawne lub niepoprawne. Poprawne polecenie może być poleceniem rezygnacji z ruchu lub jego wykonania.</p>
<p>Wiersz o treści <code>=</code> to poprawne polecenie rezygnacji z ruchu. Wiersz z nazwą pola jest poprawnym poleceniem wykonania ruchu jeżeli, zgodnie z regułami gry w Reversi, ruch na to pole jest legalny.</p>
<h1 id="postać-wyniku">Postać wyniku</h1>
<p>Na wyjściu programu jest ciąg diagramów planszy, przedzielonych komunikatami z odpowiedzią na polecenia użytkownika.</p>
<p>Diagram opisuje każde pole za pomocą znaku</p>
<ul>
<li><p><code>-</code> gdy pole jest puste,</p></li>
<li><p><code>C</code> gdy na polu jest pion czarny,</p></li>
<li><p><code>B</code> gdy na polu jest pion biały.</p></li>
</ul>
<p>Znaki te są pogrupowane w wiersze i uporządkowane w kolejności rosnącego numeru wiersza a w wierszu, według rosnącej kolumny. W <em>lewym górnym rogu</em> diagramu jest więc pole <code>a1</code>. Na końcu każdego wiersza i pod każdą kolumną jest ich oznaczenie. Np. diagram stanu początkowego planszy ma postać</p>
<pre><code>--------1
--------2
--------3
---BC---4
---CB---5
--------6
--------7
--------8
abcdefgh</code></pre>
<p>Odpowiedź programu na polecenie użytkownika mieści się w jednym wierszu. Kończy się on, poprzedzoną spacją, oceną planszy. Ocena jest równa różnicy między liczbą czarnych i białych pionów.</p>
<p>Jeśli polecenie jest błędne, na początku odpowiedzi programu, przed oceną planszy, jest znak zapytania. W przeciwnym przypadku zapisany jest tam ruch gracza Czarnego i, po spacji, ruch gracza Białego. Rezygnację z ruchu zapisujemy jako <code>=</code> a wykonanie ruchu, wskazując pole, na które położono pion.</p>
<p>Odpowiedź na polecenie jest więc słowem języka opisanego poniższą gramatyką bezkontekstową z symbolem początkowym <code>S</code>. Symbolami końcowymi tej gramatyki są znaki ujęte w apostrofy oraz <code>n</code>, które reprezentuje liczbę całkowitą zapisaną dziesiętnie.</p>
<pre><code>S → R ' ' n
R → '?' | P ' ' P
P → '=' | K W
K → 'a' | 'b' | 'c' | 'd' | 'e' | 'f' | 'g' | 'h'
W → '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8'</code></pre>
<p>W tekście wynikowym programu nie ma żadnych znaków, które nie zostały wymienione powyżej. Każdy wypisywany wiersz, także ostatni, jest zakończony końcem wiersza <code>\n</code>.</p>
<h1 id="przykłady">Przykłady</h1>
<p>Do treści zadania dołączone są pliki <code>.in</code> z przykładowymi danymi i pliki <code>.out</code> z wynikami wzorcowymi.</p>
<ul>
<li><p>Dla danych <a href="przyklad1.in">przyklad1.in</a> poprawny wynik to <a href="przyklad1.out">przyklad1.out</a>.</p></li>
<li><p>Dla danych <a href="przyklad2.in">przyklad2.in</a> poprawny wynik to <a href="przyklad2.out">przyklad2.out</a>.</p></li>
<li><p>Dla danych <a href="przyklad3.in">przyklad3.in</a> poprawny wynik to <a href="przyklad3.out">przyklad3.out</a>.</p></li>
</ul>
<h1 id="walidacja-i-testy">Walidacja i testy</h1>
<ul>
<li><p>Rozwiązania zostaną poddane walidacji, wstępnie sprawdzającej zgodność ze specyfikacją. Pomyślne przejście walidacji jest warunkiem dopuszczenia programu do testów poprawności.</p></li>
<li><p>Programy będą kompilowane poleceniem</p>
<pre><code>gcc -std=c11 -pedantic -Wall -Wextra -Werror nazwa.c -o nazwa</code></pre>
<p>Wymagane są wszystkie wymienione opcje kompilatora. Nie będą do nich dodawane żadne inne.</p></li>
<li><p>Przyjmujemy, że wynik funkcji <code>main()</code> inny niż <code>0</code> informuje o błędzie wykonania programu.</p></li>
<li><p>Poprawność wyniku sprawdzamy, przekierowując na wejście programu zawartość pliku <code>.in</code> i porównując rezultat, za pomocą programu <code>diff</code>, z plikiem <code>.out</code>, np.</p>
<pre><code>&lt; przyklad.in ./nazwa | diff - przyklad.out</code></pre>
<p>Ocena poprawności wyniku jest binarna. Uznajemy go za poprawny, jeżeli program <code>diff</code> nie wskaże żadnej różnicy względem wyniku wzorcowego.</p></li>
</ul>
<h1 id="wskazówki">Wskazówki</h1>
<ul>
<li><p>Pod Linuxem, pracując z programem interakcyjnie na konsoli, koniec danych sygnalizujemy, naciskając klawisze <code>Ctrl</code>-<code>D</code>.</p></li>
<li><p>W przygotowaniu danych testowych może pomóc polecenie <code>tee</code>. Przesyła ono dane z wejścia na wyjście, jednocześnie zapisując ich kopię w pliku, którego nazwa jest argumentem polecenia. Wykonanie</p>
<pre class="sourceCode bash"><code class="sourceCode bash"><span class="kw">tee</span> test.in <span class="kw">|</span> <span class="kw">./reversi</span></code></pre>
<p>uruchomi program <code>reversi</code> w trybie interaktywnym, tworząc kopię danych testowych w pliku <code>test.in</code>. Dzięki temu test na tych samych danych będzie można powtórzyć, wykonując polecenie</p>
<pre><code>&lt; test.in ./reversi &gt; test.out</code></pre></li>
</ul>
<h1 id="wyjaśnienia">Wyjaśnienia</h1>
<ul>
<li><p>Wolno założyć, że każde polecenie użytkownika, także ostatnie, jest w wierszu poprawnie zakończonym reprezentacją końca wiersza <code>\n</code>.</p></li>
<li><p>Program czyta z wejścia i wykonuje wszystkie polecenia użytkownika. Nie przerywa pracy przed dojściem do końca danych nawet, gdyby stwierdził, że dalsza gra nie ma sensu. Nie uznaje też za błąd sytuacji, w której użytkownik zrezygnował z gry, choć jeszcze nie przegrał.</p></li>
