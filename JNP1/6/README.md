### Opis

Na fali sukcesu Star Wars postanowiono wyprodukować sequel Star Wars 2.

Znowu musisz stworzyć program, który symuluje bitwy gwiezdne.
Jednak tym razem masz użyć innego paradygmatu do realizacji swojego programu.

Celem zadania jest stworzenie programu symulującego bitwy gwiezdne. Bitwa
rozgrywa się w przestrzeni międzygwiezdnej pomiędzy siłami Rebelii a Imperium.
Ponieważ chcemy symulować różne bitwy, należy przygotować rozwiązanie ogólne.

Stwórz klasy opisane poniżej oraz odpowiednie klasy pomocnicze, które ułatwią
implementację oraz umożliwią łatwą rozbudowę programu.

### Plik rebelfleet.h oraz rebelfleet.cc

Klasy Explorer, StarCruiser i XWing, w których używane są typy pomocnicze:
* **ShieldPoints** będący typem przechowującym wytrzymałość tarczy,
* **Speed** będący typem reprezentującym prędkość,
* **AttackPower** reprezentujący siłę ataku.

#### Konstruktor
**Klasa Explorer** przyjmuje w konstruktorze parametry **shield** typu **ShieldPoints** oraz
**speed** typu **Speed**, a **StarCruiser** oraz **XWing** dodatkowo parametr **power** typu **AttackPower**.
Klasa **StarCruiser** może przyjmować speed w zakresie od **99999 do 299795** włącznie,
a klasy **Explorer** oraz **XWing** w zakresie od **299796 do 2997960** włącznie. Poprawność
wartości parametru **speed** należy sprawdzać za pomocą **asercji**.

#### Metody publiczne
Klasy **Explorer, StarCruiser i XWing** udostępniają metody publiczne:
* **ShieldPoints getShield()** – zwraca wytrzymałość tarczy,
* **Speed getSpeed()** – zwraca prędkość statku,
* **void takeDamage(AttackPower damage)** – zmniejsza wytrzymałości tarczy o damage, ale nie więcej niż statek ma aktualnie.

Klasy **StarCruiser oraz XWing** mają dodatkowo metodę publiczną:
* **AttackPower getAttackPower()** – zwraca siłę ataku statku.

**Klasy Explorer, StarCruiser i XWing powinny być podklasami ogólniejszego typu
RebelStarship.**

Dodatkowo powinny istnieć funkcje fabrykujące dla Explorer, StarCruiser i XWing
z sygnaturami pasującymi do przykładu.

### Plik imperialfleet.h oraz imperialfleet.cc

Klasy DeathStar, ImperialDestroyer i TIEFighter, w których używane są typy
pomocnicze:
* **ShieldPoints** będący typem przechowującym wytrzymałość tarczy,
* **AttackPower** reprezentujący siłę ataku.

#### Konstruktor
Klasy **DeathStar, ImperialDestroyer i TIEFighter** przyjmują w konstruktorze
parametry **shield** typu **ShieldPoints** oraz **power** typu **AttackPower**.

#### Metody publiczne
Klasy DeathStar, ImperialDestroyer i TIEFighter mają metody publiczne:
* **ShieldPoints getShield()** – zwraca wytrzymałość tarczy,
* **AttackPower getAttackPower()** – zwraca siłę ataku statku,
* **void takeDamage(AttackPower damage)** – zmniejsza wytrzymałości tarczy o damage, ale nie więcej niż statek ma aktualnie.

**Klasy DeathStar, ImperialDestroyer i TIEFighter powinny być podklasami
ogólniejszego typu ImperialStarship.**

W sequelu statki Imperium mogą również formować grupę, która działa wspólnie.
Należy zaimplementować odpowiednią klasę reprezentującą eskadrę **Squadron**,
która w konstruktorze przyjmuje** wektor (std::vector)** albo **listę
inicjującą (std::initializer_list)** zawierającą statki, które są w eskadrze.
Squadron ma metody publiczne:
* **ShieldPoints getShield()** – zwraca wytrzymałość tarczy eskadry (suma
  wytrzymałości niezniszczonych statków w eskadrze),
* **AttackPower getAttackPower()** – zwraca siłę ataku eskadry (suma sił ataku
  niezniszczonych statków w eskadrze),
* **void takeDamage(AttackPower damage)** – zmniejsza wytrzymałości tarczy o damage
  dla każdego statku w eskadrze, ale nie więcej niż statek ma aktualnie.

Dodatkowo powinny istnieć funkcje fabrykujące dla DeathStar, ImperialDestroyer,
TIEFighter i Squadron z sygnaturami pasującymi do przykładu.

### Plik battle.h oraz battle.cc

Klasa **SpaceBattle** tworzona za pomocą klasy **Builder**, dla której można ustawić conajmniej:
* statki, które biorą udział w bitwie,
* czas startowy **t0** typu **Time**,
* czas maksymalny **t1** typu **Time**.

Klasa SpaceBattle ma metody publiczne:
* **size_t countImperialFleet()** – zwraca liczbę niezniszczonych statków Imperium,
* **size_t countRebelFleet()** – zwraca liczbę niezniszczonych statków Rebelii,
* **void tick(T timeStep)** – na początku sprawdza aktualny czas; jeśli jest to
  czas ataku, to następuje atak statków Imperium na statki Rebelii; na koniec
  czas przesuwa się o timeStep.

SpaceBattle rozgrywa się w czasie międzygwiezdnym. Czas liczony jest
w sekundach, od sekundy 0 do sekundy t1 i potem znów od sekundy 0, i tak
cyklicznie. Pierwsze odliczanie zaczyna się od sekundy t0. **Ze względu na
zakłócenia magnetyczne statki mogą atakować tylko w sekundach podzielnych
przez 2 lub 3, a niepodzielnych przez 5**.

Ataki podczas bitwy odbywają się sekwencyjnie. W sekundzie ataku każdy
niezniszczony statek imperialny po kolei atakuje wszystkie niezniszczone statki
rebelianckie, czyli ma miejsce następująca sekwencja zdarzeń:

dla każdego statku Imperium
  dla każdego statku Rebelii
    jeśli oba statki nie nie zostały jeszcze zniszczone,
      statek Imperium atakuje statek Rebelii.

Kolejność atakowania (iterowania) jest zgodna z kolejnością, w jakiej statki
zostały przekazane podczas konstrukcji. Jeśli zaatakowana jednostka rebeliancka
może się bronić (ma parametr power), to wtedy obrażenia zadawane są
„jednocześnie” i oba statki odnoszą odpowiednie obrażenia zgodnie z siłami ataku.
Statek zostaje zniszczony, jeśli wytrzymałość jego tarczy spadnie do zera.

Wywołanie tick() na bitwie, podczas gdy wszystkie statki Imperium zostały
zniszczone, powoduje wypisanie na standardowe wyjście napisu "REBELLION WON\n".
Wywołanie tick() na bitwie, podczas gdy wszystkie statki Rebelii zostały
zniszczone, powoduje wypisanie na standardowe wyjście napisu "IMPERIUM WON\n".
Jeśli wszystkie statki zarówno Imperium jak i Rebelii są zniszczone, to zostaje
wypisany napis "DRAW\n".

### Inne wymagania

Bardzo istotną częścią zadania jest zaprojektowanie odpowiedniej hierarchii
klas. W szczególności nie wszystkie klasy, jakie są wymagane w rozwiązaniu,
zostały jawnie wyspecyfikowane w treści zadania.

Należy zaprojektować metodę lub metody ataku na ofiarę tak, aby spełnić
wymagania funkcjonalne. Miejsce umieszczenia tej metody zależy od projektanta.
Statek imperialny atakuje statek rebeliancki, obniżając wytrzymałość jego tarczy.
Należy też uwzględnić specjalny przypadek, gdy atakowane są XWing lub StarCruiser
– wtedy atak następuje w dwie strony – wytrzymałość tarczy obniżana jest zarówno
statkowi rebelianckiemu, jak i imperialnemu.

Strategia czasów ataków powinna być łatwo podmienialna na inną, zgodnie z zasadą
Open-Closed Principle z zasad SOLID.

Przykład użycia znajduje się w pliku starwars2_example.cc.

Rozwiązanie będzie kompilowane poleceniem

g++ -Wall -Wextra -O2 -std=c++17 *.cc
