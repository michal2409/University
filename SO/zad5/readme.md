<!DOCTYPE html>
<html lang='pl'>
  <head>
    <title>Systemy operacyjne, zadanie 5</title>
    <meta content='text/html; charset=UTF-8' http-equiv='content-type'>
    <link href='https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css' rel='stylesheet'>
  </head>
  <body>
    <div class='container'>


      <h2>Zadanie 5</h2>
      <p>
				Najprawdopodobniej każdej i każdemu z nas zdarzyło się choć raz usunąć
				przez przypadek jakiś ważny plik. Problem ten wydaje się być powszechny,
				skoro	wiele systemów operacyjnych i menedżerów plików wprowadza
				różnorodne zabezpieczenia (pytania typu „Czy na pewno chcesz usunąć ten
				plik?”, kosz systemowy, …) mające zapobiegać takim sytuacjom.
			</p>
			<p>
        Celem tego zadania jest rozbudowanie serwera <code>mfs</code>,
        obsługującego system plików MINIX (MINIX File System), o prototypowe
        mechanizmy utrudniające przypadkowe usunięcie pliku w systemie MINIX.
      </p>
      <p>
				W poniższym opisie słowo <i>plik</i> jest używane w znaczeniu zwykłego
				pliku, nie katalogu.
      </p>


      <h3>Usuwanie plików</h3>
      <p>
        Zmodyfikowany serwer działa tak samo jak normalny <code>mfs</code> za
        wyjątkiem operacji usuwania. Jeśli użytkownik będzie próbował usunąć plik
        (nie katalog), to działanie serwera zależy od aktywnego trybu:
      </p>

      <h4>Tryb A</h4>
      <p>
				W tym trybie próba usunięcia pliku powinna się po prostu nie powieść.
				Operacja usuwania kończy się błędem informującym, że takie działanie nie
				jest dozwolone (<code>EPERM</code>). Przykład:
      </p>
<pre>
# ls
A.mode plik
# rm ./plik
rm: ./plik: Operation not permitted
# echo $?
1
# ls
A.mode plik
</pre>
			<p>
				Polecenie <code>echo $?</code> wyświetla status zakończenia (ang.
				<i>exit status</i>) poprzedniego polecenia.
			</p>

			<h4>Tryb B</h4>
			<p>
				Gdy aktywny jest ten tryb, aby skutecznie usunąć plik należy wywołać
				operację usuwania dwukrotnie. Pierwsza próba usunięcia kończy się błędem
				informującym, że operacja jest w toku (<code>EINPROGRESS</code>), a plik
				jest nadal dostępny. Można go odczytywać, edytować, a nawet odmontować
				i zamontować w innej maszynie dysk, na którym jest on zapisany. Dopiero
				ponowna próba usunięcia tego pliku rzeczywiście usuwa go z dysku. Jeśli
				użytkownik zorientuje się, że to nie ten plik chce usunąć, po pierwszym
				wywołaniu może wykonać operację zastąpienia tego pliku samym sobą
				(<code>mv ./plik ./plik</code>). Wtedy serwer „zapomina” o poprzedniej
				próbie usunięcia pliku i aby go teraz usunąć, należy znów wywołać
				operację usuwania dwukrotnie. Przykład:
			</p>
<pre>
# ls
B.mode plik
# rm ./plik
rm: ./plik: Operation now in progress
# echo $?
1
# ls
B.mode plik
# mv ./plik ./plik
# rm ./plik
rm: ./plik: Operation now in progress
# ls
B.mode plik
# rm ./plik
# echo $?
0
# ls
B.mode
</pre>

			<h4>Tryb C</h4>
			<p>
				Działając w tym trybie, serwer zamiast usunąć plik, zmienia mu nazwę,
				dodając do aktualnej sufiks <code>.bak</code>. Zatem, z punktu widzenia
				użytkownika, plik znika, ale powstaje jego kopia zapasowa. Usuwanie
				pliku będącego już kopią zapasową (pliku o nazwie kończącej się
				<code>.bak</code>) działa standardowo (tzn. bez tworzenia kopii
				zapasowej kopii zapasowej). Jeśli zmiana nazwy nie jest możliwa (np. po
				dodaniu sufiksu <code>.bak</code> nowa nazwa pliku przekroczy maksymalną
				dozwoloną	długość), cała operacja usuwania powinna zakończyć się błędem
				informującym o przyczynie niepowodzenia, a plik nie powinien zostać
				usunięty. Przykład:
			</p>
<pre>
# ls
C.mode plik
# rm ./plik
# echo $?
0
# ls
C.mode plik.bak
# rm ./plik.bak
# echo $?
0
# ls
C.mode
</pre>


			<h3>Sterowanie trybami</h3>
			<p>
				Aby serwer wykonywał usuwanie plików w sposób niestandardowy, użytkownik
				musi stworzyć plik o nazwie <code>A.mode</code>, <code>B.mode</code> lub
				<code>C.mode</code>, w zależności od tego, z którego trybu chce korzystać.
				Interfejs ten aktywuje wybrany tryb tylko dla usuwania plików leżących
				w tym samym katalogu co plik <code>{A,B,C}.mode</code>. Zatem
				w podkatalogu lub nadkatalogu użytkownik może korzystać z innego bądź
				żadnego trybu.
			</p>
			<p>
				Aby dezaktywować wybrany tryb, należy usunąć plik
				<code>{A,B,C}.mode</code>. Usuwanie plików <code>{A,B,C}.mode</code>
				powinno działać zawsze w standardowy sposób, niezależnie od aktywnego
				trybu. Nie podlegają one specjalnemu usuwaniu! Jeśli użytkownik stworzy
				w danym miejscu więcej niż jeden plik <code>{A,B,C}.mode</code>, serwer
				może realizować operację usuwania w dowolny sposób, ale nie może ona
				uszkodzić systemu plików lub samego serwera.
			</p>


      <h3>Wymagania i niewymagania</h3>
      <ol>
        <li>
					Poprzez usuwanie pliku rozumiemy operację, jaką wykonuje serwer
					obsługujący system plików podczas użycia polecenia <code>rm</code>
					w powłoce lub jego odpowiednika w jakimś języku programowania.
				</li>
				<li>
					Usuwanie innych obiektów niż zwykłe pliki (np. katalogów) ma działać
					jak w niezmodyfikowanym serwerze, niezależnie od aktywnego trybu.
				</li>
				<li>
					Gdy żaden tryb specjalny nie jest aktywny (nie ma w danym katalogu
					żadnego z plików <code>{A,B,C}.mode</code>), usuwanie powinno działać
					jak w niezmodyfikowanym serwerze.
				</li>
				<li>
					Wszystkie pozostałe operacje i działania na plikach inne niż usuwanie
					powinny działać jak w niezmodyfikowanym serwerze.
				</li>
        <li>
					Modyfikacje serwera nie mogą powodować błędów w systemie plików: ma
					być on zawsze poprawny i spójny.
				</li>
				<li>
					Dyski przygotowane i używane przez niezmodyfikowany serwer
					<code>mfs</code> powinny być poprawnymi dyskami także dla
					zmodyfikowanego serwera. Nie wymagamy natomiast odwrotnej
					kompatybilności, tzn. dyski używane poprzez zmodyfikowany serwer nie
					muszą działać poprawnie z niezmodyfikowanym serwerem.
				</li>
        <li>
					Modyfikacje mogą dotyczyć tylko serwera <code>mfs</code> (czyli mogą
					dotyczyć tylko plików w katalogu <code>/usr/src/minix/fs/mfs</code>).
				</li>
        <li>
					Podczas działania zmodyfikowany serwer nie może wypisywać żadnych
					dodatkowych informacji na konsolę ani do rejestru zdarzeń
					(ang. <em>log</em>).
				</li>
        <li>
					Można założyć, że w testowanych przypadkach użytkownik będzie miał
					wystarczające uprawnienia do wykonania usuwania i wszystkich innych
					operacji, które potrzebuje wykonać zmodyfikowane usuwanie.
				</li>
        <li>
					Można założyć, że w testowanych przypadkach w systemie plików będą
					tylko zwykłe pliki (nie łącza, nie pseudo-urządzenia itp.) i katalogi.
				</li>
				<li>
					Można założyć, że na MINIX-e będzie zawsze ustawiona prawidłowa data
					i godzina, a rozwiązanie nie musi działać poprawnie przed 2019 i po
					2037 roku.
				</li>
			</ol>

      <h3>Wskazówki</h3>
      <ol>
        <li>
					Aby skompilować i zainstalować zmodyfikowany serwer <code>mfs</code>,
					należy wykonać <code>make; make install</code> w katalogu
					<code>/usr/src/minix/fs/mfs</code>. Takimi poleceniami będzie budowane
					i instalowane oddanie rozwiązanie.
				</li>
        <li>
					Każde zamontowane położenie (ich listę wyświetli polecenie
					<code>mount</code>) obsługiwane jest przez nową instancję serwera
					<code>mfs</code>. Położenia zamontowane przed instalacją nowego
					serwera będą obsługiwane nadal przez jego starą wersję, więc, aby
					przetestować na nich zmodyfikowany serwer, należy je odmontować
					i zamontować ponownie lub zrestartować system.
				</li>
        <li>
					Aby zmodyfikowany serwer obsługiwał też korzeń systemu plików
					<code>/</code> należy wykonać dodatkowe kroki, ale radzimy nie
					testować na nim (i nie wymagamy tego) zmodyfikowanego usuwania.
				</li>
        <li>
          Do MINIX-a uruchomionego pod <code>qemu</code> można dołączać
          dodatkowe dyski twarde (i na nich testować swoje modyfikacje). Aby
          z tego skorzystać, należy:
          <ol type="a">
            <li>
							Na komputerze-gospodarzu stworzyć plik będący nowym dyskiem, np.:
							<code>qemu-img create -f raw extra.img 1M</code>.
						</li>
            <li>
							Podłączyć ten dysk do maszyny wirtualnej, dodając do parametrów,
							z jakimi uruchamiane jest <code>qemu</code>, parametry
							<code>-drive file=extra.img,format=raw,index=1,media=disk</code>,
							gdzie <code>index</code> to numer kolejny dysku (0 to główny dysk
							– główny obraz naszej maszyny).
						</li>
            <li>
							Za pierwszym razem stworzyć na nowym dysku system plików mfs:
							<code>/sbin/mkfs.mfs /dev/c0d1</code>, gdzie
							<code>/dev/c0d&lt;numer kolejny dodanego dysku></code>.
						</li>
            <li>
							Stworzyć pusty katalog (np. <code>mkdir /root/nowy</code>)
							i zamontować do niego podłączony dysk
							<code>mount /dev/c0d1 /root/nowy</code>.
						</li>
            <li>
							Wszystkie operacje wewnątrz tego katalogu będą realizowane na
							zamontowanym w tym położeniu dysku.
						</li>
            <li>
							Aby odmontować dysk, należy użyć polecenia
							<code>umount /root/nowy</code>.
						</li>
          </ol>
        </li>
        <li>
					Szukając miejsca do przechowywania na dysku dodatkowych informacji o
					zapisanych na nim plikach można skorzystać z wymagania poprawnego
					działania systemu plików tylko od 2019 do 2037 roku.
        </li>
      </ol>


      <h3>Rozwiązanie</h3>
      <p>
				Poniżej przyjmujemy, że ab123456 oznacza identyfikator studenta
				rozwiązującego zadanie. Należy przygotować łatkę (ang. <i>patch</i>) ze
				zmianami. Plik o nazwie <code>ab123456.patch</code> uzyskujemy za pomocą
				polecenia <code>diff -rupN</code>, tak jak w zadaniu 3. Łatka będzie
				aplikowana przez umieszczenie jej w katalogu <code>/</code> nowej kopii
				MINIX-a i wykonanie polecenia <code>patch -p1 < ab123456.patch</code>.
				Należy zadbać, aby łatka zawierała tylko niezbędne różnice.
				W repozytorium, w katalogu <code>studenci/ab123456/zadanie5</code>
				należy umieścić tylko patch ze zmianami. Ostateczny termin umieszczenia
				rozwiązania w repozytorium to <b><del>4 czerwca 2019, godz. 20.00</del>
				9 czerwca 2019, godz. 23:59</b>.
			</p>


      <h3>Ocenianie</h3>
      <p>
				Oceniana będą zarówno poprawność, jak i styl rozwiązania. Podstawą do
				oceny rozwiązania będą testy automatyczne i przejrzenie kodu przez
				sprawdzającego. Za poprawną i w dobrym stylu implementację obsługi
				usuwania plików w trybach A, B i C można otrzymać odpowiednio 1, 2 i 2 pkt.
				Rozwiązanie, w którym łatka nie nakłada się poprawnie, które nie
				kompiluje się lub powoduje <code>kernel panic</code> podczas
				uruchamiania otrzyma 0 pkt.
			</p>


      <h3>Pytania</h3>
      <p>
				Pytania do zadania należy kierować na adres
				<i>BanaszekMateusz@mimuw.edu.pl</i> z <code>[SOzad5]</code> w temacie,
				a odpowiedzi na często zadawane pytania szukać w pliku
				<a href="faq.html">faq.html</a>.
			</p>


    </div>
  </body>
</html>
