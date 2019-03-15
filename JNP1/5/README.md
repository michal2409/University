### Opis
Bajtek, po krachu jego cyfrowej waluty, został zmuszony wrócić do kariery
akademickiej. Ponieważ nie jest zadowolony ze swojej nowej pensji, pisze obecnie
wniosek o awans na stanowisko profesora. We wniosku musi podkreślić „wpływowość”
swoich publikacji. W tym celu potrzebuje przeprowadzić analizę cytowań.

Strukturą, która umożliwia przeprowadzenie takiej analizy, jest graf cytowań.
**Graf cytowań** jest grafem **skierowanym acyklicznym**. Każdy wierzchołek grafu
odpowiada **jednej publikacji**. Graf ma jedno **źródło**, które reprezentuje **pierwotną
publikację**, której wpływ chcemy analizować. Krawędź łączy publikację **cytowaną**
z publikacją **cytującą**.

### Zadanie
Należy zaimplementować szablon klasy, która reprezentuje taki graf.

**template \<class Publication> CitationGraph;**

Klasa **Publication** reprezentuje informacje o publikacji. Jej implementacja
zostanie dostarczona w stosownym czasie.

* Klasa **CitationGraph** powinna udostępniać następujący interfejs.

    * Tworzy nowy graf. Tworzy także węzeł publikacji o identyfikatorze **stem_id**. <br/>
    **CitationGraph(Publication::id_type const &stem_id);**

    *  Konstruktor przenoszący i przenoszący operator przypisania. Powinny być **noexcept**. <br/>
    **CitationGraph(CitationGraph\<P> &&other);** <br/>
    **CitationGraph\<P>& operator=(CitationGraph\<P> &&other);**

   * Zwraca identyfikator źródła. Metoda ta powinna być **noexcept wtedy i tylko
     wtedy, gdy metoda Publication::get_id jest noexcept**. Zamiast pytajnika należy
     wpisać stosowne wyrażenie. <br/>
     **Publication::id_type get_root_id() const noexcept(?);**

   * Zwraca listę identyfikatorów publikacji cytujących publikację o podanym
     identyfikatorze. Zgłasza **wyjątek PublicationNotFound**, jeśli dana publikacja
     nie istnieje. <br/>
     **std::vector\<Publication::id_type> get_children(Publication::id_type const &id) const;**

    * Zwraca listę identyfikatorów publikacji cytowanych przez publikację o podanym
      identyfikatorze. Zgłasza wyjątek **PublicationNotFound**, jeśli dana publikacja
      nie istnieje. <br/>
      **std::vector\<Publication::id_type> get_parents(Publication::id_type const &id) const;**

    * Sprawdza, czy publikacja o podanym identyfikatorze istnieje. <br/>
      **bool exists(Publication::id_type const &id) const;**

    * Zwraca referencję do obiektu reprezentującego publikację o podanym
    identyfikatorze. Zgłasza wyjątek **PublicationNotFound**, jeśli żądana publikacja
    nie istnieje. <br/>
    **Publication& operator[](Publication::id_type const &id) const;**

    * Tworzy węzeł reprezentujący nową publikację o identyfikatorze **id** cytującą
     publikacje o podanym identyfikatorze **parent_id** lub podanych identyfikatorach
    **parent_ids**. Zgłasza wyjątek **PublicationAlreadyCreated**, jeśli publikacja
    o identyfikatorze **id** już istnieje. Zgłasza wyjątek **PublicationNotFound**, jeśli
    któryś z wyspecyfikowanych poprzedników nie istniejealbo lista poprzedników jest pusta. <br/>
    **void create(Publication::id_type const &id, Publication::id_type const &parent_id);** <br/>
    **void create(Publication::id_type const &id, std::vector\<Publication::id_type> const &parent_ids);**

    * Dodaje nową krawędź w grafie cytowań. Zgłasza wyjątek **PublicationNotFound**,
    jeśli któraś z podanych publikacji nie istnieje. <br/>
    **void add_citation(Publication::id_type const &child_id, Publication::id_type const &parent_id);**

    * Usuwa publikację o podanym identyfikatorze. Zgłasza wyjątek
    **PublicationNotFound**, jeśli żądana publikacja nie istnieje. Zgłasza wyjątek
    **TriedToRemoveRoot** przy próbie usunięcia pierwotnej publikacji. <br/>
    **void remove(Publication::id_type const &id);**

* Zakładamy, że:
    * klasa **Publication** ma konstruktor przyjmujący argument typu
  **Publication::id_type;**
    * klasa **Publication** ma metodę **Publication::id_type get_id() const**, która
  dodatkowo może, ale nie musi być noexcept;
    * typ **Publication::id_type** ma **konstruktor bezargumentowy, konstruktor
  kopiujący, konstruktor przenoszący i operatory przypisania**;
    * wartości typu **Publication::id_type** tworzą porządek liniowy i można je
  porównywać za pomocą operatorów **==, !=, <=, >=, <, >**.

* Ponadto:
    * wszystkie metody klasy **CitationGraph** powinny gwarantować **silną odporność
  na wyjątki**, a tam, gdzie to jest możliwe i pożądane, powinny być **noexcept**;
    * wszystkie metody powinny być **przezroczyste dla wyjątków**, czyli powinny
  przepuszczać wszelkie wyjątki zgłaszane przez wywoływane przez nie funkcje,
  a obserwowalny stan obiektu **nie powinien się zmienić**;
    * próba użycia **konstruktora kopiującego lub kopiującego operatora** przypisania
  dla obiektów klasy **CitationGraph** powinna zakończyć się **błędem kompilacji**;
    * zachowanie obiektu typu **CitationGraph** po utworzeniu cyklu pozostaje
  **niezdefiniowane – nie trzeba wykrywać takiej sytuacji**;
    * wyjątki **PublicationAlreadyCreated, PublicationNotFound oraz TriedToRemoveRoot**
  powinny być zdefiniowane **poza klasą CitationGraph** i powinny **dziedziczyć
  z std::exception;**
    * wyszukiwanie publikacji powinno być **szybsze niż liniowe**.

* Zarządzanie pamięcią powinno być zrealizowane za pomocą sprytnych wskaźników
z biblioteki standardowej.

Przykład użycia znajduje się w pliku citation_graph_example.cc i wypisuje na
standardowe wyjście:

PublicationNotFound
PublicationAlreadyCreated
TriedToRemoveRoot

**=== Ustalenia techniczne ===**

Rozwiązanie będzie kompilowane na maszynie students poleceniem

g++ -Wall -Wextra -Og -g -std=c++17

Jako rozwiązanie należy dostarczyć plik **citation_graph.h**, który należy umieścić
w repozytorium w katalogu
