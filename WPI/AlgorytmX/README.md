<h1 id="wprowadzenie">Wprowadzenie</h1>
<p>Rozwiązaniem problemu <a href="https://en.wikipedia.org/wiki/Exact_cover">dokładnego pokrycia</a>, dla rodziny <code>P</code> podzbiorów dziedziny <code>S</code>, jest zbiór <code>Q \in P</code> taki, że każdy element dziedziny <code>S</code> należy do dokładnie jednego elementu <code>Q</code> Problem dokładnego pokrycia jest <a href="https://en.wikipedia.org/wiki/NP-completeness">NP-zupełny</a>. Nie jest znany deterministyczny algorytm rozwiązujący go w czasie wielomianowym. Algorytm o koszcie wykładniczym, stosujący <em>metodę prób i błędów</em>, jest przez Donalda Knutha nazywany <a href="https://en.wikipedia.org/wiki/Knuth%27s_Algorithm_X">Algorytmem X</a>.</p>
<p>Wiele problemów, np. łamigłówki, można sprowadzić do problemu dokładnego pokrycia. Pomimo wykładniczej złożoności, efektywność rozwiązania problemu za pomocą Algorytmu X może być dla użytkownika akceptowalna.</p>
<h1 id="polecenie">Polecenie</h1>
<p>Napisz program czytający ciąg wierszy <code>F, W1, ..., Wn</code>, w którym <code>F</code> to <em>filtr</em> a <code>W1, ..., Wn</code> jest instancją problemu dokładnego pokrycia.</p>
<p>Wszystkie czytane wiersze są tej samej długości <code>d</code>. Wiersz <code>F</code> składa się ze znaków minus <code>-</code> i plus <code>+</code>. W każdym wierszu <code>W1, ..., Wn</code> jest co najmniej jeden znak różny od podkreślenia <code>_</code>.</p>
<p>Wynikiem programu jest ciąg wierszy reprezentujących rozwiązania instancji <code>W1, ..., Wn</code>, przekształconych następnie za pomoca filtru <code>F</code> przez usunięcie znaków z pozycji, na których w filtrze jest <code>-</code>.</p>
<p>Rozwiązaniem instancji<code>W1, ..., Wn</code> jest zbiór <code>Q \in {1, ..., n}</code>, reprezentowany przez tekst <code>R</code> długości<code>d</code>. Tekst ten nie zawiera znaku podkreślenia <code>_</code> i spełnia warunek <code>\forall i \in {1,...,d} \exists j \in Q : (Wj)_i = R_i oraz \forall k \in Q \setminus j (Wk)_i = '_'</code>

<p>Rozwiązania budujemy ze znaków wierszy <code>W1,...,Wn</code>, zgodnie z ich kolejnością. W danym rozwiązaniu, w kolejności od początku, wybieramy metodą prób i błędów znaki nie kolidujące z wyborami dokonanymi wcześniej.</p>
<h1 id="przykłady">Przykłady</h1>
<p>Do treści zadania dołączone są pliki <code>.in</code> z przykładowymi danymi i pliki <code>.out</code> z wynikami wzorcowymi.</p>
<ul>
<li><p>Dla danych <a href="https://moodle.mimuw.edu.pl/pluginfile.php?file=%2F24600%2Fmod_assign%2Fintroattachment%2F0%2Fprzyklad1.in&amp;amp;forcedownload=1">przyklad1.in</a> poprawny wynik to <a href="https://moodle.mimuw.edu.pl/pluginfile.php?file=%2F24600%2Fmod_assign%2Fintroattachment%2F0%2Fprzyklad1.out&amp;amp;forcedownload=1">przyklad1.out</a>.</p></li>
<li><p>Dla danych <a href="https://moodle.mimuw.edu.pl/pluginfile.php?file=%2F24600%2Fmod_assign%2Fintroattachment%2F0%2Fprzyklad2.in&amp;amp;forcedownload=1">przyklad2.in</a> poprawny wynik to <a href="https://moodle.mimuw.edu.pl/pluginfile.php?file=%2F24600%2Fmod_assign%2Fintroattachment%2F0%2Fprzyklad2.out&amp;amp;forcedownload=1">przyklad2.out</a>.</p></li>
<li><p>Dla danych <a href="https://moodle.mimuw.edu.pl/pluginfile.php?file=%2F24600%2Fmod_assign%2Fintroattachment%2F0%2Fprzyklad3.in&amp;amp;forcedownload=1">przyklad3.in</a> poprawny wynik to <a href="https://moodle.mimuw.edu.pl/pluginfile.php?file=%2F24600%2Fmod_assign%2Fintroattachment%2F0%2Fprzyklad3.out&amp;amp;forcedownload=1">przyklad3.out</a>.</p></li>
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
<h1 id="uwagi-i-wskazówki">Uwagi i wskazówki</h1>
<ul>
<li><p>Wolno założyć, że dane są poprawne.</p></li>
<li><p>Wolno założyć, że wszystkie wiersze danych są zakończone reprezentacją końca wiersza.</p></li>
<li><p>Program nie powinien nakładać ograniczeń na liczbę wierszy danych i ich długość. Wolno tylko założyć, że dane zmieszczą się w pamięci.</p></li>
<li><p>Naturalnym sposobem implementacji metody prób i błędów jest rekursja.</p></li>
<li><p>Donald Knuth opisuje implementację Algorytmu X za pomocą dość złożonej struktury danych, opartej na liście dwukierunkowej, z zastosowaniem techniki <a href="https://en.wikipedia.org/wiki/Dancing_Links">tańczących wskaźników</a>.</p>
<p>Nie wymagamy od Państwa takiej implementacji. Oczywiście nie zabraniamy używania list. Za całkowicie wystarczające uznamy jednak rozwiązanie, w którym jedynymi dynamicznymi strukturami danych będą powiększane tablice.</p></li>
<li><p>Prosimy pamiętać, że Państwa praca ma być w pełni samodzielna.</p></li>
