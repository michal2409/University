package agent;

import spotkanie.Spotkanie;
import java.util.Random;
import java.util.PriorityQueue;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;

class SpotkanieComparator implements Comparator<Spotkanie> {
    @Override
    public int compare(Spotkanie s1, Spotkanie s2) {
        return s1.dajDzien() - s2.dajDzien();
    }
}

public abstract class Agent {
    protected final int idAgenta;
    protected Stan stan; 
    protected HashSet<Agent> sasiedzi;
    protected PriorityQueue<Spotkanie> spotkania;
    
    public Agent(int id) {
        idAgenta = id;
        sasiedzi = new HashSet<>(); 
        spotkania = new PriorityQueue<>(50, new SpotkanieComparator());
        this.stan = Stan.ZDROWY;
    }
    
    // Zwraca true jeśli agent chce iść na spotkanie.
    // Agent zwykły przeciąża tę funkcję.
    public boolean czyIdzieNaSpotkanie(double prawdSpotkania, Random rand) {
        return rand.nextDouble() < prawdSpotkania;
    }
    
    // Umawia spotkanie na dany dzień
    // Agent towarzyski przeciąża tę funkcję.
    public void umówSpotkanie(int dzieńSpotkania, Random rand) {
        int index = rand.nextInt(sasiedzi.size());
        Iterator<Agent> iter = sasiedzi.iterator();
        for (int i = 0; i < index; i++) 
            iter.next();
        spotkania.add(new Spotkanie(dzieńSpotkania, this, iter.next()));
    }
    
    // Przeprowadza spotkania zaplanowane na dzień spotkania. Zwraca liczbę zarażeń 
    // do których doszło na przeprowadzonych spotkaniach.
    public int spotkajSię(int dzieńSpotkania, double prawdZarazenia, Random rand) {
        int liczbaZarażeń = 0;
        while (spotkania.peek() != null && spotkania.peek().dajDzien() == dzieńSpotkania)
            if (spotkania.poll().przeprowadzSpotkanie(prawdZarazenia, rand))
                liczbaZarażeń++;
        return liczbaZarażeń;
    }

    public boolean jestZdrowy() {
        return stan == Stan.ZDROWY;
    }
    
    public boolean jestChory() {
        return stan == Stan.CHORY;
    }
    
    public boolean jestMartwy() {
        return stan == Stan.MARTWY;
    }
    
    public boolean jestOdporny() {
        return stan == Stan.ODPORNY;
    }
    
    public void ustawStan(Stan stan) {
        this.stan = stan;
    }
    
    public void dodajSasiada(Agent sasiad) {
        sasiedzi.add(sasiad);
    }
    
    public HashSet<Agent> dajSasiadow() {
        return sasiedzi;
    }
     
    public PriorityQueue<Spotkanie> dajSpotkania() {
        return spotkania;
    }
    
    public int dajId() {
        return idAgenta;
    }
    
    @Override
    public String toString() {
        String s = (this.jestChory()) ? "* " : " ";
        return (idAgenta+1) + s;
    }
}
