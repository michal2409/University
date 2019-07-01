<!DOCTYPE html>
<html lang="pl"><head>
  <title>Systemy operacyjne, zadanie 3</title>
  <meta http-equiv="content-type" content="text/html; charset=UTF-8">
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
</head><body>
<div class="container">

<h2>Zadanie 6</h2>

<p>
Celem zadania jest stworzenie urządzenia znakowego <code>/dev/adler</code>, które będzie liczyło
sumę kontrolną <a href="https://pl.wikipedia.org/wiki/Adler-32">Adler32</a>.
</p>

<p>
W algorytmie Adler-32 trzymamy dwie zmienne: A i B, początkowo inicjowane
odpowiednio na 1 i 0. Taki stan zmiennych nazywamy stanem początkowym.
</p>

<p>
Operacja odczytu z urządzenia daje wartość sumy kontrolnej w formacie
szesnastkowym (cyfry powyżej 9 pisne małymi literami) wszystkich danych, które
się do urządzenia zapisało od ostatniego odczytu (lub uruchomienia urządzenia,
jeśli nie było jeszcze żadnego odczytu).
Stan urządzenia jest resetowany do stanu początkowego po każdym odczycie.
Suma kontrolna ma 8 znaków i poprawna operacja odczytu daje w wyniku zawsze 8 bajtów.
Operacja odczytu nie powinna się udać (nic nie powinno być przeczytane), jeśli
rozmiar czytanych danych jest mniejszy od 8 znaków, i powinna zakończyć się
wtedy błędem EINVAL – invalid argument.
</p>

<p>
Operacja zapisu do urządzenia pozwala na wpisanie danych, których sumę kontrolną
chcemy policzyć. <strong>Dane potencjalnie mogą być bardzo duże.</strong>
</p>

<p>
Urządzenie powinno zachowywać aktualny stan w przypadku uruchomienia
<code>service update</code>.
</p>

<p>
W <code>/etc/system.conf</code> zostanie umieszczony poniższy wpis:
</p>
<pre>
service adler
{
        system
                IRQCTL          # 19
                DEVIO           # 21
        ;
        ipc
                SYSTEM pm rs tty ds vm vfs
                pci inet lwip amddev
                ;
    uid 0;
};
</pre>

<p>
Sterownik będzie kompilowany za pomocą dostarczonego przez nas Makefile,
analogicznego jak dla przykładowego sterownika hello. Rozwiązanie powinno składać
się z jednego pliku <code>adler.c</code>, który będzie zawierał implementację urządzenia
znakowego. Plik razem z Makefile zostanie umieszczony w katalogu:
<code> /usr/src/minix/drivers/adler</code>,
gdzie zostaną wykonane polecenia:
</p>
<pre>
make clean
make
make install

service up /service/adler
service update /service/adler
service down adler
</pre>

<h3> Oddawanie zadania</h3>

<p>
Termin oddania: 10 czerwca 2019, godz. 20.
</p>

<p>
Rozwiązanie należy umieścić w pliku <code>adler.c</code> w repozytorium SVN w katalogu
</p>
<pre>
studenci/ab123456/zadanie6
</pre>
<p>
gdzie ab123456 jest identyfikatorem studenta używanym do logowania
w laboratorium komputerowym. Katalog ten nie powinien zawierać innych plików.
</p>

<p>
Z uwagi na automatyzację sprawdzania, rozwiązania niespełniające wymogów
formalnych (termin oddania, nazwa pliku, nazwa katalogu w repozytorium itp.)
otrzymają 0 punktów.
</p>

<p>
Pytania należy kierować do jks@mimuw.edu.pl
</p>

<h3>Przykład</h3>
<pre>
# mknod /dev/adler c 20 0
# service up /service/adler -dev /dev/adler
# head -c 8 /dev/adler | xargs echo
00000001
# echo "Hello" &#62; /dev/adler
# head -c 8 /dev/adler | xargs echo
078b01ff
# head -c 8 /dev/adler | xargs echo
00000001
</pre>
</div>

</body></html>
