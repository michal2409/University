<!DOCTYPE html>
<html lang="pl">
<head>
  <meta http-equiv="content-type" content="text/html; charset=UTF-8">
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
<body>
<div class="container">

<h3>Wprowadzenie</h3>
<p>
Jak wskazują badania naukowe i nasze własne doświadczenia, percepcja czasu zmienia się wraz z wiekiem (<a href="https://en.wikipedia.org/wiki/Time_perception#Changes_with_age">https://en.wikipedia.org/wiki/Time_perception#Changes_with_age)</a>. Im jesteśmy starsi, tym szybciej zdaje się płynąć czas. Można podejrzewać, że to zjawisko jest jednym z powodów, dla których międzypokoleniowy dialog bywa niezwykle trudny – gdyby tylko starsze pokolenie mogło przekonać swoje dzieci i wnuki, że należy „chwytać dzień”, bo czas płynie tak szybko, a sami zrozumieli, że minuty mogą dłużyć się niczym godziny, gdy odliczamy czas do weekendu…
</p>

<h2>Zadanie 3</h2>
<p>
Naszym zadaniem jest umożliwienie procesom MINIX-a uproszczoną manipulację upływem czasu innych procesów (potomków i przodków w linii prostej). W tym celu należy m.in. zaimplementować wywołanie systemowe <code>PM_DISTORT_TIME</code>, funkcję biblioteczną <code>int distort_time(pid_t pid, uint8_t scale)</code> (dostępną przez <code>unistd.h</code>) oraz dokonać modyfikacji istniejących wywołań systemowych: <code>PM_GETTIMEOFDAY</code> i <code>PM_CLOCK_SETTIME</code>.
</p>
<p>
Efekt wywołania dodanej funkcji systemowej powinien być następujący:
<ul>
    <li>Jeśli proces wywołujący, C, jest <strong>przodkiem</strong> procesu P o podanym numerze <code>pid</code>, upływ czasu odczytywanego przez P za pomocą <code>PM_GETTIMEOFDAY</code> jest <strong>przyspieszany</strong> (skalowany) zgodnie z wartością <code>scale</code>.</li>
    <li>Jeśli proces wywołujący, C, jest <strong>potomkiem</strong> procesu P o podanym numerze <code>pid</code>, upływ czasu odczytywanego przez P za pomocą <code>PM_GETTIMEOFDAY</code> jest <strong>opóźniany</strong> (skalowany) zgodnie z wartością <code>scale</code>.</li>
    <li>Ponowne wywołanie funkcji <code>distort_time</code> dla tego samego procesu C, jeśli zakończy się sukcesem, powinno nadpisać poprzednio ustalony współczynnik przeskalowania upływu czasu. </li>
</ul>
<p>
W przypadku pomyślnego przeskalowania czasu, funkcja <code>distort_time</code> powinna zwrócić kod <code>OK</code> (wartość 0). Wszelkie błędy występujące na drodze do przeskalowania czasu powinny skutkować zwróceniem odpowiedniej wartości innej niż <code>OK</code> (niektóre z takich sytuacji zostały opisane w punkcie <a href="#specific_cases">Przypadki szczególne</a>).
</p>
<p>
Jak wspomniano, proces P mierzy upływ czasu za pomocą istniejącego wywołania systemowego <code>PM_GETTIMEOFDAY</code>. Gdy upływ czasu procesu P zostanie przeskalowany o wartość różną od 1, jedynie pierwszy odczyt <code>PM_GETTIMEOFDAY</code> będzie zgodny z rzeczywistością i stanie się punktem odniesienia w czasie. Każdy kolejny odczyt zwróci przeskalowaną wartość względem ustalonego punktu odniesienia. Na przykład, jeśli przyspieszenie upływu czasu jest dwukrotne, P odczyta najpierw prawdziwy czas, <code>X</code>, za pomocą <code>PM_GETTIMEOFDAY</code>, a po 5 sekundach odczyt przy użyciu <code>PM_GETTIMEOFDAY</code> wskaże czas (<code>X</code> + 10 sekund). Zgodnie z obecną implementacją <code>PM_GETTIMEOFDAY</code>, czas po przeskalowaniu powinien być zwracany z dokładnością do mikrosekund.
</p>
<p>
Wywołanie funkcji systemowej <code>PM_CLOCK_SETTIME</code> przez którykolwiek z działających procesów powinno zresetować punkty odniesienia w czasie dla wszystkich procesów, tzn.: <code>PM_GETTIMEOFDAY</code> powinno ponownie zwrócić procesowi P rzeczywisty czas (nowy punkt odniesienia) i dopiero kolejne odczyty zostaną przeskalowane.
</p>

<h3 id="specific_cases">Przypadki szczególne</h3>
<ul>
    <li>Jeśli proces o podanym <code>pid</code> nie istnieje: <code>return EINVAL</code>.</li>
    <li>Jeśli <code>pid</code> procesu C i podany <code>pid</code> procesu P są takie same: <code>return EPERM</code>.</li>
    <li>Jeśli procesy P i C nie są spokrewnione w linii prostej: <code>return EPERM</code>.</li>
    <li>Jeśli <code>scale == 0</code>: odczytywany czas dla procesu P powinien stać w miejscu.</li>
</ul>

<h3>Przykład</h3>
<pre>
P1: proces
P2: proces powstały w wyniku wywołania fork() przez P1
P3: proces-obserwator

/* Procesy P1, P2 zakłócają sobie wzajemnie poczucie upływu czasu. */
P1.distort_time(P2, 3)
P2.distort_time(P1, 2)

/* Procesy odczytują aktualny czas - odczytane wartości są prawdziwe, podobne do wartości odczytanej przez P3. */
T1_start = P1.gettimeofday()
T2_start = P2.gettimeofday()
T3_start = P3.gettimeofday()

/* Procesy zasypiają na 5 sekund czasu rzeczywistego - działanie funkcji sleep nie podlega zaburzeniom czasu. */
P1.sleep(5)
P2.sleep(5)
P3.sleep(5)

/* Procesy odczytują ponownie czas. */
T1_end = P1.gettimeofday()
T2_end = P2.gettimeofday()
T3_end = P3.gettimeofday()

/* Różnice podane są w przybliżeniu. */
T1_end - T1_start = 2.5 s
T2_end - T2_start = 15 s
T3_end - T3_start = 5 s

/* Jeden z procesów ustala nowy czas. */
P3.settimeofday(T3_end)
T1_start2 = P1.gettimeofday()
assert(T1_start2 prawie równy T3_end)
P1.sleep(3)
T1_end2 = P1.gettimeofday()
assert((T1_end2 - T1_start2) prawie równe 1.5 s)

/* P2 nadpisuje ustalone wcześniej przeskalowanie dla P1. */
P2.distort_time(P1, 4)
T1_start3 = P1.gettimeofday()
P1.sleep(10)
T1_end3 = P1.gettimeofday()
assert((T1_end3 - T1_start3) prawie równe 2.5 s)
</pre>

<h3>Postać rozwiązania</h3>
<p>
Poniżej przyjmujemy, że ab123456 oznacza identyfikator studenta rozwiązującego
zadanie.
</p>
<p>
Należy przygotować łatkę (ang. <i>patch</i>) ze zmianami w katalogu <code>/usr</code>.
Plik zawierający łatkę o nazwie <code>ab123456.patch</code> uzyskujemy za pomocą polecenia
</p>
<pre>
diff -rupN oryginalne-źródła/usr/ moje-rozwiązanie/usr/ > ab123456.patch
</pre>
<p>
gdzie <code>oryginalne-źródła</code> to ścieżka do niezmienionych źródeł MINIX-a, natomiast
<code>moje-rozwiązanie</code> to ścieżka do źródeł MINIX-a zawierających rozwiązanie. Tak
użyte polecenie <code>diff</code> rekurencyjnie przeskanuje pliki ze ścieżki
<code>oryginalne-źródła/usr</code>, porówna je z plikami ze ścieżki <code>moje-rozwiązanie/usr</code>
i wygeneruje plik <code>ab123456.patch</code>, który podsumowuje różnice. Tego pliku będziemy
używać, aby automatycznie nanieść zmiany na nową kopię MINIX-a, gdzie będą
przeprowadzane testy rozwiązania.
</p>
<p>
Umieszczenie łatki w katalogu / nowej kopii MINIX-a i wykonanie polecenia <code>patch -p1 < ab123456.patch</code>
powinno skutkować naniesieniem wszystkich oczekiwanych zmian wymaganych przez
rozwiązanie. Należy zadbać, aby łatka zawierała tylko niezbędne różnice.
</p>
<p>
Po naniesieniu łatki domyślnie zostaną wykonane polecenia:
<ul>
<li><code>make && make install</code> w katalogach <code>/usr/src/minix/servers/pm</code> oraz <code>/usr/src/lib/libc</code>,</li>
<li><code>make do-hdboot</code>w katalogu <code>/usr/src/releasetools</code>,</li>
<li><code>reboot</code>.</li>
</ul>
Jeżeli rozwiązanie wymaga podjęcia innych kroków instalacji, autor jest proszony o dołączenie pliku <code>setup.sh</code>, który przeprowadzi instalację rozwiązania po uruchomieniu z katalogu <code>/</code>. Ostatnim poleceniem w pliku <code>setup.sh</code> powinno być polecenie <code>reboot</code>.
</p>
<h3>Oddawanie rozwiązań</h3>
<p>
Łatkę o nazwie <code>ab123456.patch</code> należy umieścić w repozytorium, w katalogu <code>studenci/ab123456/zadanie3</code> (<code>ab123456</code> zastępujemy <strong>swoim</strong> loginem). Poza łatką i, opcjonalnie, plikiem <code>setup.sh</code> w katalogu nie powinny znaleźć się żadne inne pliki.
Ostateczny termin umieszczenia rozwiązania w repozytorium to <strong>30 kwietnia 2019, godz. 20.00</strong>.
</p>

<h3>Uwagi różne</h3>
<ul>
    <li>Wskazówka: serwer pm przechowuje informacje o procesach w tablicy <code>mproc</code> zadeklarowanej w pliku <code>mproc.h</code>.
    </li>
    <li>Nie przyznajemy punktów za rozwiązanie, w którym łatka nie nakłada się
poprawnie, które nie kompiluje się lub które powoduje kernel panic podczas
uruchamiania.</li>
    <li>Pytania dotyczące zadania należy kierować na adres mailowy:
        inga.rub@mimuw.edu.pl.</li>
</ul>

<h3>Pytania i odpowiedzi</h3>
<p>P: <b>Czy samo wywołanie <code>distort_time</code> wpływa na punkt odniesienia?</b>
</br>
O: Nie, punkt odniesienia jest ustalany jedynie przez pierwsze wywołanie <code>gettimeofday</code>, które nastąpiło po przeskalowaniu czasu za pomocą <code>distort_time</code> oraz przez wywołanie <code>settimeofday</code>.</p>

<p>P: <b>Czy <code>gettimeofday</code> wywołany po <code>distort_time</code> przy współczynniku skalowania równym 1 ustala punkt odniesienia?</b>
</br>
O: Tak.</p>

<p>P: <b>Czy można założyć, że liczba nanosekund, która upłynęła od 1970 roku mieści się w zmiennej 64-bitowej w momencie korzystania z <code>gettimeofday</code>?</b>
</br>
O: Nie.</p>

<p>P: <b>Czy wywołanie syscalla <code>PM_CLOCK_SETTIME</code> resetuje punkty odniesienia tylko jak się powiedzie, czy zawsze?</b>
</br>
O: <code>PM_CLOCK_SETTIME</code> resetuje punkty odniesienia tylko wtedy, gdy się powiedzie.</p>

<p>P: <b>Czy wywołanie funkcji <code>distort_time</code> powinno zwracać kod błędu poprzez <code>errno</code> czy bezpośrednio?</b>
</br>
O: Bezpośrednio.</p>

<p>P: <b>Czy zakończenie pracy przez proces P, który dokonał przeskalowania czasu dla procesu Q anuluje przeskalowanie czasu dla Q?</b>
</br>
O: Nie.</p>
<p>P: <b>Czy informacje o współczynniku przeskalowania i ustawionym punkcie odniesienia powinny być dziedziczone przy wykonaniu <code>fork()</code>?</b>
</br>
O: Wszystkie warianty dziedziczenia informacji o zniekształceniu upływu czasu są dopuszczalne.</p>

<p><b>P: Kiedy ustawiany jest punkt odniesienia?</b>
</br>
O: Punkt odniesienia ustawiany jest przy okazji wywołania <code>gettimeofday</code>, którego pseudokod wygląda następująco:
<ul>
    <li>Jeśli skala zniekształcenia czasu nie została ustalona, zwróć wynik jak zwykle.</li>
    <li>Jeśli skala zniekształcenia czasu została ustalona, ale nie ma punktu odniesienia, ustaw punkt odniesienia (aktualny czas) i zwróć wynik jak zwykle.</li>
    <li>Jeśli skala zniekształcenia czasu została ustalona i jest punkt odniesienia, zwróć zniekształcony wynik.</li>
</ul>
</p>
</div>
</body>
</html>
