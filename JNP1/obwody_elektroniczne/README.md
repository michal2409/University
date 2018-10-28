# Obwody elektroniczne

Należy napisać program, który na podstawie tekstowego opisu obwodu
elektronicznego wytworzy listę typów elementów (ang. bill of materials)
potrzebnych do zmontowania odpowiedniego układu. Dla uproszczenia zakładamy, że
obwody elektroniczne składają się z:
- tranzystorów **T**
- diod **D**
- rezystorów **R**
- kondensatorów **C**
- źródeł napięcia **E**

## Wejście
Program czyta opis obwodu ze standardowego wejścia. Każdy wiersz zawiera opis jednego elementu obwodu. 
##### Wiersz zawiera: 
- oznaczenie elementu 
- typ elementu
- listę węzłów, do których podłączone są końcówki tego elementu.
##### Oznaczenie elementu:
- litera T, D, R, C lub E
- numer elementu (liczba całkowita z zakresu od 0 do 999999999, bez zer wiodących)

##### Typ elementu:
- napis rozpoczynający się wielką literą alfabetu ang lub cyfrą dziesiętną, po czym mogą wystąpić wielkie i małe litery alfabetu ang, cyfry dziesiętne, przecinek, łącznik lub ukośnik

##### Numer węzła:
- liczba całkowita z zakresu od 0 do 999999999 (bez zer wiodących)

##### Ponadto:
- Tranzystor ma trzy końcówki, a pozostałe elementy – dwie końcówki.
- Składowe opisu elementu odseparowane są białymi znakami, które mogą też znajdować się na początku i końcu wiersza.
- Oznaczenia elementów w obwodzie nie mogą się powtarzać.
- Wszystkie końcówki danego elementu nie mogą być podłączone do jednego węzła.

## Wyjście
##### Program:
- analizuje dane wejściowe wiersz po wierszu
- ignoruje puste wiersze
- ocenia poprawność każdego niepustego wiersza

##### Jeśli wiersz nie jest poprawny:
- informacje w nim zawarte nie są brane pod uwagę
- program wypisuje na standardowe wyjście diagnostyczne wiersz z informacją o błędzie (numer błędnego wiersza, **licząc od 1**, i dokładną, oryginalną jego postać)

Po przeanalizowaniu wejścia program wypisuje na standardowe wyjście listę typów elementów.
Każdy typ elementu ma być opisany w 1 wierszu, który ma zawierać listę oznaczeń elementów tego typu oraz typ elementu.

Lista typów elementów powinna być posortowana w następującej kolejności:
- tranzystory,
- diody
- rezystory
- kondensatory
- źródła napięcia

W obrębie tego samego rodzaju elementów lista oznaczeń elementów i wiersze powinny być posortowane rosnąco według numerów elementów.

Na koniec, jeśli w obwodzie występuje węzeł, do którego nie są podłączone co najmniej 2 elementy, program wypisuje na standardowe wyjście diagnostyczne 1 wiersz z ostrzeżeniem zawierającym posortowaną rosnąco listę numerów takich węzłów.

W obwodzie zawsze występuje węzeł o numerze 0 (masa), nawet gdy nie został jawnie wyspecyfikowany w danych wejściowych.
