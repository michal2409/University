Zadanie 1

Termin oddania: 22.03.2019, godz. 20.00

Krzemową dolinkę zamieszkują dwie nacje: cienkokońcówkowce i grubokońcówkowce.
Głównym zajęciem końcówkowców jest wymienianie się plikami. Jednak obie nacje
nie przepadają wzajemnie za sobą i od czasu do czasu wszczynają wojny
nieatomowe. Dla grubokońcówkowca sygnałem do ataku jest otrzymanie pliku
z utajnioną wiadomością. Cienkokońcówkowce odkryły sposób kodowania tej
wiadomości. Napisz program, który pomoże cienkokońcówkowcom sprawdzać, czy
przechwycony plik zawiera sygnał do ataku. Cienkokońcówkowce odkryły, że
grubokońcówkowce wykorzystują do tego celu magiczną stałą 68020, na cześć
legendarnego grubokońcówkowego wojownika poległego w jednej z wojen jabłecznych.
Niestety cienkokońcówkowce nie potrafią odkodować dokładnego czasu ataku,
dlatego istotne jest opracowanie programu, który, zużywając możliwie mało
zasobów (czas i pamięć), pozwoli nawet małemu cienkokońcówkowcowi w porę
przygotować się do nadchodzącego ataku. Na szczęście cienkokońcówkowce mają
instrukcję bswap, która umożliwia im sprawne czytanie liczb grubokońcówkowych.

Formalna specyfikacja zadania

Napisz w asemblerze x86_64 program, który jako argument przyjmuje nazwę pliku
i sprawdza, czy ten plik zawiera utajniony sygnał do ataku. Plik traktujemy jako
binarny zawierający sekwencję 32-bitowych liczb (oczywiście kodowanych
w formacie grubokońcowkowym). Plik zawiera sygnał do ataku, jeśli jednocześnie
spełnione są wszystkie następujące warunki:
– plik nie zawiera liczby 68020,
– plik zawiera liczbę większą od 68020 i mniejszą od 2 do potęgi 31,
– w pliku istnieje pięć kolejnych liczb o wartościach odpowiednio 6, 8, 0, 2, 0,
– suma wszystkich liczb w pliku modulo 2 do potęgi 32 jest równa 68020.

Program niczego nie wypisuje. Program kończy się kodem (ang. exit code) 0, jeśli
podany plik zawiera sygnał do ataku, a 1 w przeciwnym przypadku. Program kończy
się kodem 1 również wtedy, gdy wystąpił jakiś błąd, np. podano złą liczbę
argumentów, podany argument jest niepoprawny, plik o podanej nazwie nie
istnieje, plik ma złą długość, wystąpił błąd operacji odczytu pliku itp.

Tekst źródłowy programu należy umieścić w pliku attack.asm w repozytorium SVN
w katalogu https://svn.mimuw.edu.pl/repos/SO/studenci/login/zadanie1, gdzie
login to identyfikator używany do logowania w laboratorium. W katalogu
z rozwiązaniem nie wolno umieszczać żadnych innych plików.

Nie wolno korzystać z żadnych bibliotek. Rozwiązanie będzie kompilowane na
maszynie students poleceniami:

nasm -f elf64 -o attack.o attack.asm
ld --fatal-warnings -o attack attack.o

Oceniane będą poprawność i czas działania programu, zapotrzebowanie na pamięć,
rozmiar kodu maszynowego (sumaryczny rozmiar sekcji ładowanych z pliku
wykonywalnego do pamięci), jakość kodu źródłowego i spełnienie formalnych
wymagań podanych w treści zadania. Program niekompilujący się otrzyma 0 punktów.
Testy, limity czasowe i pamięciowe zostaną ustalone po terminie oddania
rozwiązania.

Do zadania dołączone są dwa przykłady: dla pliku _test_1.0 program powinien
zakończyć się kodem 0, a dla pliku _test_1.1 – kodem 1.

Pytania do zadania można kierować na adres marpe@mimuw.edu.pl z [SOzad1]
w temacie, a odpowiedzi na często zadawane pytania szukać w pliku faq.txt.
