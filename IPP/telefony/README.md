</span><h2>Zadanie telefony, część 1</h2><div id="intro" class="box generalbox boxaligncenter"><div class="no-overflow"><p>Tegoroczne duże zadanie polega na zaimplementowaniu operacji na numerach
telefonów. Na potrzeby tego zadania przyjmujemy, że numer telefonu jest to
niepusty ciąg składający się z cyfr 0, 1, 2, 3, 4, 5, 6, 7, 8, 9.</p>

<p>Jako pierwszą część zadania należy zaimplementować moduł operacji
przekierowywania numerów telefonów. Opis interfejsu modułu znajduje się w pliku
<code>phone_forward.h</code> w formacie komentarzy dla programu <code>doxygen</code>. Przykład użycia
znajduje się w pliku <code>phone_forward_example.c</code>.</p>

</span><h2>Zadanie telefony, część 2</h2><div id="intro" class="box generalbox boxaligncenter"><div class="no-overflow"><span class="filter_mathjaxloader_equation"><span class="nolink"><p>Jako drugą część dużego zadania należy zaimplementować program, który,
korzystając z modułu zaimplementowanego w części pierwszej, udostępnia operacje
na numerach telefonów przez interfejs tekstowy. Ponadto należy zaimplementować
skrypt w bashu.</p>

<h2>Interfejs tekstowy</h2>

<p>Program czyta dane ze standardowego wejścia, wyniki wypisuje na standardowe
wyjście, a informacje o błędach na standardowe wyjście diagnostyczne.</p>

<h3>Poprawne dane wejściowe</h3>

<p>Dane wejściowe wyraża się w pewnym języku programowania.
W języku tym są trzy rodzaje leksemów:</p>

<ul>
<li><code>numer</code> – niepusty ciąg cyfr 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;</li>
<li><code>identyfikator</code> – niepusty ciąg składający się z małych lub dużych liter
alfabetu angielskiego i cyfr dziesiętnych, zaczynający się od litery – jest
nazwą bazy przekierowań;</li>
<li><code>operator</code>.</li>
</ul>

<p>W języku tym są cztery operatory:</p>

<ul>
<li><code>NEW</code></li>
<li><code>DEL</code></li>
<li><code>&gt;</code></li>
<li><code>?</code></li>
</ul>

<p>Słowa <code>NEW</code> i <code>DEL</code> są zastrzeżone – nie może być takich identyfikatorów.</p>

<p>Język udostępnia następujące operacje tworzenia, przełączania i usuwania bazy
przekierowań:</p>

<ul>
<li><code>NEW identyfikator</code> – jeśli baza przekierowań o podanej nazwie nie istnieje,
to tworzy nową bazę o tej nazwie i ustawia ją jako aktualną, a jeśli baza o
takiej nazwie już istniej, to tylko ustawia ją jako aktualną;</li>
<li><code>DEL identyfikator</code> – usuwa bazę przekierowań o podanej nazwie;</li>
</ul>

<p>Język udostępnia następujące operacje, dotyczące aktualnej bazy przekierowań,
wykonywane na numerach:</p>

<ul>
<li><code>numer &gt; numer</code> – dodaje przekierowanie numerów;</li>
<li><code>numer ?</code> – wypisuje przekierowanie z podanego numeru;</li>
<li><code>? numer</code> – wypisuje przekierowania na podany numer;</li>
<li><code>DEL numer</code> – usuwa wszystkie przekierowania, których numer jest prefiksem.</li>
</ul>

<p>Między leksemami może nie być odstępu albo może być dowolna liczba białych
znaków (spacja, tabulator, znak nowej linii, znak powrotu karetki). Między
leksemami musi być co najmniej jeden biały znak, jeśli jego brak powodowałby
błędną interpretację.</p>

<p>W języku mogą pojawić się komentarze.
Komentarz rozpoczyna i kończy się sekwencją <code>$$</code>.</p>

<p>Program powinien obsługiwać co najmniej 100 różnych baz przekierowań.</p>

<h3>Obsługa błędów składniowych</h3>

<p>Powinna zostać wypisana jedna linia z komunikatem:</p>

<pre><code>ERROR n
</code></pre>

<p>gdzie <code>n</code> to numer pierwszego znaku, który nie daje się zinterpretować jako
poprawne wejście. Numer znaku jest to kolejny numer bajtu wczytany przez
program, licząc od jedynki.</p>

<p>Jeśli dane wejściowe kończą się niespodziewanie, powinna zostać wypisana jedna
linia z komunikatem:</p>

<pre><code>ERROR EOF
</code></pre>

<h3>Obsługa błędów wykonania</h3>

<p>Powinna zostać wypisana jedna linia z komunikatem:</p>

<pre><code>ERROR operator n
</code></pre>

<p>gdzie <code>operator</code> to nazwa operatora, którego wykonanie spowodowało błąd, a <code>n</code>
numer pierwszego znaku tego operatora. Numer znaku jest to kolejny numer bajtu
wczytany przez program, licząc od jedynki.</p>

<p>Przy poprawnym wykonaniu operator <code>?</code> powinien wypisać co najmniej jeden numer.
Brak numeru do wypisania należy traktować jako błąd.</p>

<p>Na początku nie ma ustawionej aktualnej bazy danych i wszelkie operacje na
numerach należy traktować jako błędne.</p>

<h2>Zakończenia działania programu</h2>

<p>Poprawne zakończenie programu po przetworzeniu wszystkich danych wejściowych
powinno być sygnalizowane kodem wyjścia (ang. <em>exit code</em>) 0.
Zakończenie programu z błędem powinno być sygnalizowane kodem wyjścia 1.
W powyższym opisie stwierdzenie „powinna zostać wypisana jedna linia
z komunikatem” oznacza, że po wypisaniu tej linii program kończy działanie
z błędem.
Niezależnie od tego, czy program zakończył się poprawnie czy z błędem, powinien
zwolnić zaalokowaną pamięć.</p>

<h2>Zadanie telefony, część 3</h2><div id="intro" class="box generalbox boxaligncenter"><div class="no-overflow"><p>W trzeciej części zadania oczekujemy poprawienia ewentualnych błędów
z poprzednich części oraz zmodyfikowania programu.</p>

<p>Modyfikujemy definicję numeru telefonu. Teraz operujemy na 12 cyfrach. Numer
telefonu jest to nadal niepusty ciąg, którego elementami są te cyfry. Jedenastą
cyfrę, czyli cyfrę dziesięć, reprezentujemy jako znak <code>:</code>, czyli dwukropek,
a dwunastą, czyli jedenaście – jako znak <code>;</code>, czyli średnik.</p>

<h2>Moduł operacji na numerach telefonów</h2>

<p>Mówimy, że numer telefonu <code>X</code> jest nietrywialny, jeśli wynik wykonania funkcji
<code>phfwdReverse</code> dla numeru <code>X</code> zawiera jakiś numer różny od <code>X</code>.
Do modułu <code>phone_forward</code> należy doimplementować funkcję</p>

<pre><code>size_t phfwdNonTrivialCount(struct PhoneForward *pf, char const *set, size_t len);
</code></pre>

<p>Funkcja oblicza liczbę nietrywialnych numerów długości <code>len</code> zawierających tylko
cyfry, które znajdują się w napisie <code>set</code>. Jeśli wskaźnik <code>pf</code> ma wartość
<code>NULL</code>, <code>set</code> ma wartość <code>NULL</code>, <code>set</code> jest pusty, <code>set</code> nie zawiera żadnej
cyfry lub parametr <code>len</code> jest równy zeru, wynikiem jest zero. Obliczenia należy
wykonywać modulo dwa do potęgi liczba bitów reprezentacji typu <code>size_t</code>.
Napis <code>set</code> może zawierać dowolne znaki.</p>

<h2>Interfejs tekstowy</h2>

<p>Należy doimplementować operator</p>

<ul>
<li><code>@</code></li>
</ul>

<p>oraz operację</p>

<ul>
<li><code>@ numer</code> – wypisuje jedną linię (zakończoną znakiem przejścia no nowej linii)
zawierającą liczbę dziesiętną (bez zer wiodących i innych dodatkowych znaków)
będącą wynikiem działania funkcji <code>phfwdNonTrivialCount</code> dla aktualnej bazy
przekierowań, gdzie jako parametr <code>set</code> podano napis reprezentujący <code>numer</code>,
a wartość <code>len</code> to <code>max(0, |numer| - 12)</code>, przy czym <code>|numer|</code> oznacza liczbę
cyfr numeru <code>numer</code>.</li>
</ul>

