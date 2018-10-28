<h3>Wprowadzenie</h3>

<p>W jednoprocesorowym systemie operacyjnym zadaniem modułu
szeregującego (zwanego planistą) jest wybieranie z kolejki procesów
oczekujących kolejnego procesu, któremu zostanie przydzielony procesor.
Proces, któremu zostanie przydzielony procesor usuwany jest z kolejki
procesów oczekujących. O tym, czy proces wraca do kolejki procesów
oczekujących decyduje to, czy całe zadeklarowane przez proces zapotrzebowanie
na procesor zostało wykorzystane. Jeśli tak, proces opuszcza system, jeśli
nie wraca do kolejki procesów oczekujących. Miejsce, na które wraca zależy
od stosowanej strategii szeregowania. </p>

<p>Typowe strategie stosowane przez planistę to: </p>
<ul>
  <li>FCFS (ang. First Come, First
    Served) — procesor przydzielany jest procesom w kolejności ich
    pojawienia się w kolejce procesów oczekujących. W przypadku pojawienia
    się więcej niż jednego procesu w tej samej chwili czasu o kolejności
    decyduje identyfikator procesu (dla potrzeb tego zadania przyjmujemy, że
    identyfikatorem procesu jest liczba całkowita) - procesy o mniejszej
    wartości identyfikatora są umieszczane w kolejce wcześniej. Procesor
    pozostaje przydzielony procesowi do chwili wykorzystania przez proces
    całego deklarowanego zapotrzebowania na czas procesora. </li>
  <li>SJF (ang. Shortest Job
    First) — procesor przydzielany jest procesowi deklarującemu
    najmniejsze spośród procesów oczekujących zapotrzebowanie na procesor
    (liczone w jednostkach czasu). W przypadku, gdy więcej niż jeden proces
    deklaruje najmniejsze zapotrzebowanie, decyduje identyfikator procesu -
    proces o mniejszej wartości identyfikatora wybierany jest jako pierwszy.
    Procesor pozostaje przydzielony procesowi do chwili wykorzystania przez
    proces całego deklarowanego zapotrzebowania na czas procesora. </li>
  <li>SRT (ang. Shortest Remaining
    Time) — wybór następuje na zasadach analogicznych jak wyżej. W
    przypadku pojawienia się w kolejce procesów oczekujących procesu, który
    deklaruje zapotrzebowanie na procesor mniejsze niż pozostały do
    wykorzystania przez proces wykonujacy się odcinek czasu procesor jest
    odbierany procesowi wykonującemu się i przydzielany procesowi, który
    się pojawił. W przypadku, gdy pojawiający się proces deklaruje
    zapotrzebowanie identyczne z pozostałym do wykorzystania odcinkiem czasu
    procesu wykonującego się, procesor nie jest odbierany. </li>
  <li>PS (ang. Processor
    Sharing) — wszystkie procesy zgłaszajace zapotrzebowanie na
    procesor otrzymują do niego dostęp natychmiast, przy czym przy n procesach ich faktyczne wykorzystanie procesora w
    ciągu jednostki czasu wynosi 1∕n. Oznacza to w szczególności, że tempo
    wykonywania się procesu może się zmieniać w zależności od liczby
    procesów wykonujących się. Każdy z procesów korzysta z procesora do
    czasu wykorzystania całego deklarowanego zapotrzebowania na procesor. </li>
  <li>RR (ang. Round Robin)-
    procesy ustawiane są w kolejce cyklicznej zgodnie z kolejnością
    pojawiania się ich w systemie. Jeśli w systemie pojawia się więcej niż
    jeden proces w tej samej chwili czasu o miejscu w kolejce decyduje
    identyfikator procesu — proces o mniejszej wartości identyfikatora
    umieszczany jest w kolejce wcześniej. Procesor przydzielany jest procesom
    kolejno na q jednostek czasu. Po upływie czasu q proces, który się
    wykonywał umieszczany jest na końcu kolejki procesów oczekujących.
    Jeśli w chwili kończenia się odcinka q pojawi się nowy proces,
    umieszczany jest on w kolejce za procesem, któremu właśnie odebrano
    procesor. </li>
  <li>
    <p>Możliwe są również inne strategie. </p>
  </li>
</ul>


<p>Dla porównania efektywności poszczególnych strategii
szeregowania stosowane są różne kryteria. Typowe to: </p>
<ul>
  <li>średni czas obrotu zadania — średni czas liczony od
    momentu pojawienia się zadania w kolejce zadań oczekujących do momentu
    zakończenia jego wykonywania. </li>
  <li>średni czas oczekiwania — średni czas przebywania
    zadania w kolejce zadań oczekujących.</li>
</ul>

<p>Istnieją również inne kryteria, ale dla zastosowania
większości z nich, poza informacją początkową o kolejce zadań, wystarczy
informacja o momencie zakończenia poszczególnych zadań w serii. </p>

<p></p>

<h3>Zadanie</h3>

<p>Należy napisać program <code>Planista,</code> który
pozwoli na przeprowadzenie eksperymentów polegających na porównaniu
efektywności wybranych strategii zgodnie z określonym kryterium.
</p>

<p>Należy zaimplementować następujące strategie
szeregowania: FCFS, SJF, SRF, PS, RR z różną wielkością <code>q</code>
oraz oba wymienione kryteria porównywania efektywności. </p>

<p>Każdy z eksperymentów powinien polegać na symulacji
wykonania zadanego ciągu zadań z wykorzystaniem wszystkich wymienionych
strategii i ocenie ich efektywności zgodnie z wymienionymi kryteriami. </p>

<p></p>

<h3>Dane</h3>

<p>Dane do eksperymentu mają być wczytane z pliku
tekstowego, którego nazwa jest parametrem wywołania programu albo ze
standardowego wejścia, o ile program został wywołany bez parametru. Kolejne
wiersze danych mają zawierać: </p>
<ul>
  <li>liczbę całkowitą określającą liczbę zadań
    (procesów) w serii, </li>
  <li>dla każdego zadania w oddzielnym wierszu dwie liczby
    całkowite oddzielone spacją, określające moment pojawienia się zadania
    w kolejce (liczony od 0) oraz zapotrzebowanie zadania na procesor (nie
    mniej niż 1), Kolejne wiersze ułożone są w kolejności odpowiadającej
    identyfikatorom procesów - pierwszy wiersz dotyczy procesu o
    identyfikatorze 1, następny o identyfikatorze 2, itd. Można założyć,
    że przynajmniej jeden proces pojawi się w chwili 0. </li>
  <li>liczbę całkowitą określającą liczbę wariantów
    strategii RR, które należy uwzględnić, </li>
  <li>odpowiednią liczbę liczb całkowitych oddzielonych
    spacją, określających wielkość parametru q dla kolejnych wariantów
    strategii RR.</li>
</ul>

<p>Należy sprawdzać istnienie i możliwość otwarcia pliku z
danymi. Informacja o błędzie powinna mieć postać: <code>Plik z danymi nie
jest dostępny. </code></p>

<p>Nie należy zakładać istnienia pliku z danymi w tym samym
katalogu, w którym znajduje się program. </p>

<p>Należy sprawdzać poprawność danych wejściowych.
Informacja o błędzie powinna mieć postać: <code>Błąd w wierszu &lt;numer
wiersza&gt; : &lt;informacja dodatkowa&gt;.</code> Wiersze z danymi numerowane
są od jedynki. W przypadku wykrycia pierwszego błędu w danych program
powinien zakończyć pracę. Informacja dodatkowa powinna, w miarę
możliwości, określać dokładniej rodzaj błędu. Jej konkretna postać
zależy od decyzji programisty. </p>

<p>Wyniki działania programu mają zostać wypisane na
standardowym wyjściu w następującej postaci (patrz przykład):
</p>

<div>
Strategia: &lt;strategia&gt;</div>

<div>
Ciąg trójek liczb oddzielonych spacjami, ujętych w nawiasy [...]. Dwie
pierwsze liczby powinny być liczbami całkowitymi, a trzecia wypisaną z
dokładnością do dwóch miejsc po kropce liczbą dziesietną. Kolejne liczby
oznaczają: identyfikator procesu, moment przybycia procesu do kolejki zadań,
moment zakończenia wykonywania się procesu. Trójki liczb powinny być
posortowane wg. czasu zkończenia zadania. W przypadku jednakowych czasów
zakończenia, wg. identyfikatorów procesu. </div>

<div>
Średni czas obrotu: &lt;liczba z dokładnością do dwóch miejsc po
kropce&gt; </div>

<div>
Średni czas oczekiwania: &lt;liczba, jak wyżej&gt; </div>

<div>
&lt;pusty wiersz&gt; </div>

<p>i tak analogicznie dla wszystkich wymienionych strategii i
wielkości q. </p>

<p>Napis &lt;strategia&gt; ma mieć postać skrótu definiującego strategię w
treści zadania, przy czym skrót opisujacy strategię RR ma zostać
uzupełniony (po znaku minus) liczbą określającą wielkość kwantu czasu q,
np. RR-1.</p>

<h3>Przykłady</h3>

<p>Dla danych:</p>
<pre>5
0 10
0 29
0 3
0 7
0 12
2
1 10</pre>

<p>wynik powinien miec postać (proszę zwrócić uwagę na kropki):</p>

<p></p>
<pre>Strategia: FCFS
[1 0 10.00][2 0 39.00][3 0 42.00][4 0 49.00][5 0 61.00]
Średni czas obrotu: 40.20
Średni czas oczekiwania: 28.00

Strategia: SJF
[3 0 3.00][4 0 10.00][1 0 20.00][5 0 32.00][2 0 61.00]
Średni czas obrotu: 25.20
Średni czas oczekiwania: 13.00

Strategia: SRT
[3 0 3.00][4 0 10.00][1 0 20.00][5 0 32.00][2 0 61.00]
Średni czas obrotu: 25.20
Średni czas oczekiwania: 13.00

Strategia: PS
[3 0 15.00][4 0 31.00][1 0 40.00][5 0 44.00][2 0 61.00]
Średni czas obrotu: 38.20
Średni czas oczekiwania: 0.00

Strategia: RR-1
[3 0 13.00][4 0 30.00][1 0 38.00][5 0 44.00][2 0 61.00]
Średni czas obrotu: 37.20
Średni czas oczekiwania: 25.00

Strategia: RR-10
[1 0 10.00][3 0 23.00][4 0 30.00][5 0 52.00][2 0 61.00]
Średni czas obrotu: 35.20
Średni czas oczekiwania: 23.00</pre>

<p>natomiast dla danych:</p>
<pre>5
0 10
0 29
0 3
0
0 12
2
1 10</pre>
program powinien wypisać np.: 
<pre>Błąd w wierszu 5: za mało danych.</pre>
