<h2>Zadanie 1</h2><div id="intro" class="box generalbox boxaligncenter"><div class="no-overflow"><p></p><p>Zadanie polega na napisania serwera i klienta do pobierania 
fragmentów plików z innego komputera. Zakładamy, że dostępne pliki są 
krótsze niż 4 GiB. Używać będziemy protokołu TCP i adresowania IPv4.</p>
<h2 id="klient">1. Klient</h2>
<p>Klient wysyła do serwera polecenia. Są dwa rodzaje poleceń:</p>
<ul>
<li>prośba o przesłanie listy dostępnych plików;</li>
<li>żądanie przysłania fragmentu pliku o podanej nazwie.</li>
</ul>
<p>Rodzaj polecenia podany jest w jego dwóch początkowych bajtach jako 2-bajtowa liczba całkowita bez znaku:</p>
<ul>
<li>1 oznacza prośbę o przysłanie listy plików,</li>
<li>2 oznacza żądanie przysłania fragment pliku.</li>
</ul>
<p>Prośba o przysłanie listy plików nie zawiera nic więcej.</p>
<p>Żądanie przysłania fragmentu pliku zawiera kolejne pola:</p>
<ul>
<li>adres początku fragmentu w bajtach, 4-bajtowa liczba całkowita bez znaku;</li>
<li>liczba bajtów do przesłania, typ jw.;</li>
<li>długość nazwy pliku w bajtach, 2-bajtowa liczba całkowita bez znaku;</li>
<li>nazwa pliku <em>niezakończona</em> bajtem zerowym.</li>
</ul>
<p>Program klienta uruchamia się następująco:</p>
<pre><code>netstore_client &lt;nazwa-lub-adres-IP4-serwera&gt; [&lt;numer-portu-serwera&gt;]</code></pre>
<p>Domyślny numer portu to 6543.</p>
<p>Klient łączy się z serwerem po TCP, wysyła prośbę o listę plików i oczekuje odpowiedzi zawierającej listę dostępnych plików.</p>
<p>Po otrzymaniu listy plików klient powinien wyświetlić ją 
użytkownikowi na standardowe wyjście, każdy nazwa w nowym wierszu. Nazwy
 należy poprzedzić kolejnym numerem i znakiem kropki. Następnie ze 
standardowego wejścia należy pobrać numer pliku, adres początku 
fragmentu i adres końca fragmentu, każda wartość w osobnym wierszu.</p>
<p>Otrzymane wartości należy obudować żądaniem przysłania fragmentu 
pliku o formacie podanym powyżej i wysłać do serwera. Po wysłaniu klient
 oczekuje na odpowiedź.</p>
<p>Odpowiedź może być:</p>
<ul>
<li>odmową: któraś z podanych wartości jest błędna;</li>
<li>ciągiem danych z zawartością fragmentu, otrzymane dane należy 
zapisać do pliku o tej samej nazwie co źródło, ale w podkatalogu tmp 
bieżącego katalogu. Dane powinny trafić w to samo miejsce w docelowym 
pliku co w pliku, z którego były pobrane.</li>
</ul>
<p>Jeśli plik istnieje, to nie czyścimy go, tylko nadpisujemy wskazane bajty. Jeśli plik nie istnieje, to oczywiście go tworzymy.</p>
<p>Może się zdarzyć, że zapisanie fragmentu w jego miejscu spowoduje 
powstanie „dziury”. Jest to sytuacja poprawna. W kolejnym pobraniu być 
może załatamy tę dziurę albo wypełnimy ją w inny sposób.</p>
<p>Po pozytywnym odebraniu wszystkich bajtów fragmentu należy zakończyć pracę.</p>
<p>W przypadku odmowy przyczynę odmowy należy wypisać użytkownikowi, po czym zakończyć pracę.</p>
<h2 id="serwer">2. Serwer</h2>
<p>Program serwera uruchamia się następująco:</p>
<pre><code>netstore-server &lt;nazwa-katalogu-z-plikami&gt; [&lt;numer-portu-serwera]</code></pre>
<p>Domyślny numer portu to 6543.</p>
<p>Serwer po wystartowaniu oczekuje na polecenia klientów.</p>
<p>Po nawiązaniu połączenia serwer oczekuje na prośbę o przysłanie listy nazw dostępnych plików.</p>
<p>Odpowiedź z listą dostępnych plików zawiera na początku dwubajtową liczbę całkowitą 1.</p>
<p>Kolejne pola to: - długość pola z nazwami plików: 4-bajtowa liczba 
całkowita bez znaku; - nazwy plików, rozdzielane znakiem | (kreska 
pionowa).</p>
<p>Zakładamy, że nazwy wszystkich plików są w ASCII i nie zawierają znaków o kodach mniejszych niż 32 ani znaku '|'.</p>
<p>Jeśli zamiast prośby o listę plików serwer od razu otrzyma żadanie 
fragmentu pliku, to traktuje to jako sytuację poprawną i przechodzi do 
części opisanej poniżej. Taka sytuacja nie jest możliwa w naszym 
kliencie (bo użytkownik nie ma tam jak wprowadzić nazwy pliku), ale 
zgodna z ogólnym protokołem.</p>
<p>Po wysłaniu listy plików serwer oczekuje na żądania klienta.</p>
<p>Dla każdego żądania możliwe są dwie reakcje.</p>
<ol>
<li>Wykonanie żądania nie jest możliwe:</li>
</ol>
<ul>
<li>zła nazwa pliku (być może plik w międzyczasie zniknął),</li>
<li>nieprawidłowy (w danym momencie) adres początku fragmentu: większy niż (rozmiar-pliku<rozmiar pliku=""> - 1),</rozmiar></li>
<li>podano zerowy rozmiar fragmentu.</li>
</ul>
<p>Odmowa zaczyna się dwubajtowym polem z liczbą 2, powód odmowy jest 
podany w kolejnym polu 4-bajtowym zawierającym podtyp (wartość 
odpowiednio 1, 2 lub 3 dla powyższych błędów).</p>
<p>Po wysłaniu odmowy serwer łagodnie zamyka połączenie.</p>
<ol start="2">
<li>Żądanie jest wykonalne. Jeśli rozmiar fragmentu jest za duży, to 
wysłane będą wszystkie bajty do końca pliku (czyli rozmiar zostanie 
,,obcięty'').</li>
</ol>
<p>Serwer zaczyna wysyłać podany fragment. Na początku wysyła dwubajtowe
 pole z liczbą 3, następnie (być może zmodyfikowaną, patrz wyżej) 
długość fragmentu na 4 bajtach. Potem idą już bajty fragmentu.</p>
<p>Ponieważ pliki (i fragmenty) bywają spore i nie ma sensu trzymanie 
bufora, który pomieściłby cały wysyłany plik lub jego fragment, to 
wstawianie danych do wysłania powinno odbywać się porcjami po około 512 
KiB.</p>
<p>Po udanym wysłaniu wszystkich bajtów serwer czeka na zamknięcie połączenia, po czym obsługuje następnego klienta.</p>
<h2 id="dodatkowe-wymagania">3. Dodatkowe wymagania</h2>
<ol>
<li><p>Liczby całkowite identyfikujące rodzaj polecenia lub odpowiedzi, 
adres początku fragmentu czy jego rozmiar, przesyłamy w sieciowej 
kolejności bajtów.</p></li>
<li><p>Katalog z rozwiązaniem powinien zawierać pliki źródłowe klient.c i
 serwer.c oraz plik Makefile zapewniający automatyczną kompilację i 
linkowanie. Można też umieścić tam inne pliki potrzebne do skompilowania
 i uruchomienia programu, jeśli to jest konieczne.</p></li>
</ol>
<p>Nie wolno używać dodatkowych bibliotek, czyli innych niż standardowa biblioteka C.</p>
