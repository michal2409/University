### Opis
Celem zadania jest stworzenie programu symulującego bitwy gwiezdne. Bitwa rozgrywa się w przestrzeni międzygwiezdnej pomiędzy siłami Rebelii a Imperium. Ponieważ chcemy symulować różne bitwy, należy przygotować rozwiązanie ogólne. Stwórz następujące typy szablonowe:

### rebelfleet.h

* Klasy **Explorer**, **StarCruiser** i **XWing** reprezentujące statki *Rebelii*, gdzie **U** jest typem pól przechowujących **wytrzymałość tarczy** oraz **prędkość**, a w przypadku *XWing* oraz *StarCruiser* również **siłę ataku** statku.

* Klasa **Explorer** przyjmuje w *konstruktorze* parametry **shield** oraz **speed** typu **U**, a **StarCruiser** oraz **XWing** dodatkowo parametr **attackPower** też typu **U**. Klasa **StarCruiser** może przyjmować **speed** w zakresie od **99999** do **299795** włącznie, a klasy **Explorer** oraz **XWing** w zakresie od **299796** do **2997960** włącznie. Poprawność wartości parametru speed należy sprawdzać za pomocą **asercji**.

* Klasy **Explorer**, **StarCruiser** i **XWing** udostępniają metody publiczne: 
    * **U getShield()** – zwraca wytrzymałości tarczy,
    * **U getSpeed()** – zwraca prędkość, 
    * **void takeDamage(U damage)** – zmniejsza wytrzymałości tarczy o damage, ale nie więcej niż statek ma aktualnie.

* Klasy **StarCruiser** oraz **XWing** mają dodatkowo metodę publiczną:
    * **U getAttackPower()** – zwraca siłę ataku statku.

* Klasy **Explorer**, **StarCruiser** i **XWing** mają publiczną składową **valueType** reprezentującą typ **U**, którym zostały sparametryzowane.

* Szablony klas **Explorer**, **StarCruiser** i **XWing** powinny być specjalizacją ogólniejszego szablonu **RebelStarship<typename U, ...>**.

### imperialfleet.h

* Klasy **DeathStar**, **ImperialDestroyer** i **TIEFighter** reprezentują statki imperium, gdzie **U** jest typem pól przechowujących **wytrzymałość tarczy** i **siłę ataku** statku.

* Klasy **DeathStar**, **ImperialDestroyer** i **TIEFighter** przyjmują w konstruktorze parametry **shield** oraz **attackPower** typu **U**.

* Klasy **DeathStar**, **ImperialDestroyer** i **TIEFighter** udostępniają metody publiczne: 
    * **U getShield()** – zwraca wytrzymałości tarczy,
    * **U getAttackPower()** – zwraca siłę ataku statku,
    * **void takeDamage(U damage)** – zmniejsza wytrzymałości tarczy o damage, ale nie więcej niż statek ma aktualnie.

* Klasy **DeathStar**, **ImperialDestroyer** i **TIEFighter** mają publiczną składową **valueType** reprezentującą typ **U**, którym zostały sparametryzowane.

* Dodatkowo trzeba stworzyć funkcję, która umożliwi statkowi **Imperium** zaatakowanie wybranego statku **Rebelii**:
    * **void attack<I, R>(I imperialShip, R rebelShip)** – wykonuje atak na statek Rebelii, obniżając wytrzymałość jego tarczy. Należy też uwzględnić specjalny przypadek, gdy **R = StarCruiser** lub **R = XWing** – wtedy atak następuje **w dwie strony** – wytrzymałość tarczy jest obniżana zarówno statkowi Rebelii, jak i statkowi Imperium.

* Szablony klas **DeathStar**, **ImperialDestroyer** i **TIEFighter** powinny być specjalizacją ogólniejszego szablonu **ImperialStarship<typename U, ...>**.

### battle.h

* Klasa **SpaceBattle<typename T, t0, T t1, typename... S>**, gdzie: **T** – typ przechowujący czas, **t0** – czas startowy, **t1** – czas końcowy, po którego przekroczeniu następuje **wyzerowanie licznika**, **S...** – typy statków. Należy sprawdzać poprawność parametrów **t0 i t1 w czasie kompilacji**.

* Klasa **SpaceBattle** przyjmuje w konstruktorze kolejno **obiekty reprezentujące statki biorące udział w bitwie**.

* Klasa **SpaceBattle** udostępnia metody publiczne:
    * **size_t countImperialFleet()** – zwraca liczbę niezniszczonych statków Imperium;
    * **size_t countRebelFleet()** - zwraca liczbę niezniszczonych statków Rebelii; 
    * **void tick(T timeStep)** – na początku sprawdza aktualny czas; jeśli jest to czas ataku, to następuje atak statków Imperium na statki Rebelii; na koniec czas przesuwa się o timeStep.

* **SpaceBattle** rozgrywa się w **czasie międzygwiezdnym**. Czas liczony jest w **sekundach**, od **sekundy 0** do **sekundy t1** i potem znów od **sekundy 0**, i tak **cyklicznie**. Pierwsze odliczanie zaczyna się od **sekundy t0**. Ze względu na zakłócenia magnetyczne statki mogą atakować tylko w sekundach będących **kwadratami liczb całkowitych**. Obliczenie wszystkich potrzebnych liczb **reprezentujących sekundy ataku** i ich **zapamiętanie w odpowiedniej kolekcji** musi odbyć się w **czasie kompilacji**.

* Ataki podczas bitwy odbywają się **sekwencyjnie**. W sekundzie ataku każdy niezniszczony statek imperialny po kolei atakuje wszystkie niezniszczone statki rebelianckie, czyli ma miejsce następująca sekwencja zdarzeń:
    * Dla każdego statku **Imperium** dla każdego statku **Rebelii** jeśli oba statki nie zostały jeszcze zniszczone, statek **Imperium atakuje statek Rebelii**.

    * Kolejność atakowania (iterowania) jest **zgodna z kolejnością, w jakiej statki zostały przekazane w konstruktorze**. Jeśli zaatakowana jednostka **rebeliancka** może się bronić (**ma parametr attackPower**), to wtedy obrażenia zadawane są **„jednocześnie”** i oba statki odnoszą odpowiednie obrażenia zgodnie z siłami ataku. Statek zostaje **zniszczony**, jeśli wytrzymałość jego tarczy spadnie do **zera**.

* Wywołanie **tick()** na bitwie, podczas gdy wszystkie statki **Imperium** zostały **zniszczone**, powoduje wypisanie na standardowe wyjście napisu **"REBELLION WON\n"**. Wywołanie **tick()** na bitwie, podczas gdy wszystkie statki **Rebelii** zostały **zniszczone**, powoduje wypisanie na standardowe wyjście napisu **"IMPERIUM WON\n"**. Jeśli wszystkie statki zarówno **Imperium jak i Rebelii są zniszczone**, to zostaje wypisany napis **"DRAW\n"**.
