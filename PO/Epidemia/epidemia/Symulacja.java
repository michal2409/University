import agent.*;
import raport.Raport;
import ustawienia.Ustawienia;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Properties;
import java.util.Random;


public class Symulacja {
    private final Random rand = new Random();   
    private Agent[] agenci;
    private int lZdrowych;
    private int lOdpornych;
    private int lChorych;
    
    private long seed;
    private int liczbaAgentów;
    private double prawdTowarzyski;
    private double prawdSpotkania;
    private double prawdZarażenia;
    private double prawdWyzdrowienia;
    private double śmiertelność;
    private int liczbaDni;
    private int śrZnajomych;
    private Properties mergedProp; 
    private Raport raport;
    
    // Generuje tablice Agentów rozmiaru liczbaAgentów gdzie z prawd. prawdTowarzyski losowany 
    // jest AgentTowarzyski. Dokładnie jeden z wylosowanych Agentów jest chory, pozostali są zdrowi.
    public void generujAgentow(int liczbaAgentów, double prawdTowarzyski) {
        agenci = new Agent[liczbaAgentów];
        for (int i = 0; i < liczbaAgentów; i++) 
            agenci[i] = (rand.nextDouble() < prawdTowarzyski) ? new AgentTowarzyski(i) : new AgentZwykly(i); 
        agenci[rand.nextInt(liczbaAgentów)].ustawStan(Stan.CHORY);
    }
    
    // Generuje graf o liczbaKraw krawedziach miedzy agentami.
    public void generujGraf(int liczbaAgentów, int liczbaKraw) {
        int aktLiczbaKraw = 0;
        while (aktLiczbaKraw < liczbaKraw) {
            int w1 = rand.nextInt(liczbaAgentów); // Losujemy pierwszy wierzchołek w grafie.
            int w2 = rand.nextInt(liczbaAgentów - 1); // w2 losujemy z rozkładu jednostajnego {0, ..., w1-1, w1+1, ..., liczbaAgentów - 1}.
            if (w2 >= w1) // Zapewnia, że w1 != w2 oraz rozkład losowania jest jednostajny.
                w2++;
            if (agenci[w1].dajSasiadow().contains(agenci[w2])) // Krawędź już istnieje.
                continue;
            agenci[w1].dodajSasiada(agenci[w2]);
            agenci[w2].dodajSasiada(agenci[w1]);
            aktLiczbaKraw++;
        }
    }
    
    // Na początku każdego dnia każdy zarażony agent może umrzeć 
    // (z prawd. śmiertelność) lub wyzdrowieć (z prawd. prawdWyzdrowienia).
    public void rozpocznijDzien(int liczbaAgentów, double śmiertelność, double prawdWyzdrowienia) {
        for (Agent agent : agenci) {
            if (!agent.jestChory()) 
                continue;
            if (rand.nextDouble() < śmiertelność) { // Agent umiera.
                agent.ustawStan(Stan.MARTWY);
                lChorych--;
                for (int i = 0; i < liczbaAgentów; i++) { // Usunięcie martwego agenta z listy sąsiadów.
                    if (agenci[i].jestMartwy())
                        continue;
                    agenci[i].dajSasiadow().remove(agent);
                }
                continue;
            }
            if (rand.nextDouble() < prawdWyzdrowienia) { // Agent zyskuje odporność.
                agent.ustawStan(Stan.ODPORNY);
                lOdpornych++;
                lChorych--;
            }
        }
    }
    
    // Agent z prawd. prawdSpotkania decyduje czy chce się spotkać.
    // Agent powtarza planowanie spotkań dopóki nie wylosuje, że nie chce się spotykać.
    public void zaplanujSpotkania(int dzien, int liczbaDni, double prawdSpotkania) {
        if (liczbaDni - dzien - 1 == 0) // Ostatniego dnia nie planujemy spotkan.
            return;
        for (Agent agent: agenci) {
            if (agent.jestMartwy()) // Matrwy agent nie uczestniczy w symulacji.
                continue;
            while (agent.czyIdzieNaSpotkanie(prawdSpotkania, rand))
                // Agent losuje jeden z pozostałych dni symulacji kiedy dojdzie do spotkania.
                agent.umówSpotkanie(rand.nextInt(liczbaDni - dzien - 1) + dzien + 1, rand);
        }
    }
    
    public void przeprowadzSpotkania(int dzien, double prawdZarażenia) {
        for (Agent agent: agenci) {
            if (agent.jestMartwy()) // Matrwy agent nie uczestniczy w symulacji.
                continue;
            if (agent.dajSpotkania().peek() == null) // Agent nie ma zaplanowanych spotkań.
                continue;
            if (agent.dajSpotkania().peek().dajDzien() != dzien) // Agent nie ma zaplanowanych spotkań na dany dzień.
                continue;
            int liczbaZarażen = agent.spotkajSię(dzien, prawdZarażenia, rand);
            lChorych += liczbaZarażen;
            lZdrowych -= liczbaZarażen;
        }
    }
    
    public void wczytajUstawienia() throws IOException {
        Ustawienia ustawienia = new Ustawienia();
        ustawienia.wczytajPlikiProperties();
        mergedProp = ustawienia.dajProperties();
        seed = ustawienia.dajSeed(mergedProp);
        liczbaAgentów = ustawienia.dajLiczbęAgentów();
        prawdTowarzyski = ustawienia.dajPrawdTowarzyski();
        prawdSpotkania = ustawienia.dajPrawdSpotkania();
        prawdZarażenia = ustawienia.dajPrawdZarażenia();
        prawdWyzdrowienia = ustawienia.dajPrawdWyzdrowienia();
        śmiertelność = ustawienia.dajŚmiertelność();
        liczbaDni = ustawienia.dajLiczbęDni();
        śrZnajomych = ustawienia.dajLiczbęŚrZnajomych();
        raport = new Raport(new BufferedWriter(new FileWriter(new File(ustawienia.dajPlikZRaportem(mergedProp)))));
        lZdrowych = liczbaAgentów - 1;
        lChorych = 1;
        lOdpornych = 0;
    }
    
    public void przygotujSymulacje() throws IOException {
        rand.setSeed(seed);
        generujAgentow(liczbaAgentów, prawdTowarzyski);
        generujGraf(liczbaAgentów, liczbaAgentów * śrZnajomych / 2);
        raport.wypiszParametry(mergedProp);
        raport.wypiszAgentow(agenci);
        raport.wypiszGraf(agenci);
    }
    
    public void przeprowadźSymulacje() throws IOException {
        for (int i = 0; i < liczbaDni; i++) {
            raport.wypiszStatystykiZdrowotne(lZdrowych, lChorych, lOdpornych);
            rozpocznijDzien(liczbaAgentów, śmiertelność, prawdWyzdrowienia);
            zaplanujSpotkania(i, liczbaDni, prawdSpotkania);
            przeprowadzSpotkania(i, prawdZarażenia);
        }
        raport.wypiszStatystykiZdrowotne(lZdrowych, lChorych, lOdpornych);
        raport.dajWriter().close();
    }

    public static void main(String[] args) throws IOException {
        Symulacja sym = new Symulacja();
        sym.wczytajUstawienia();
        sym.przygotujSymulacje();
        sym.przeprowadźSymulacje();
    }
}
