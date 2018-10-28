package spotkanie;

import agent.*;
import java.util.Random;

public class Spotkanie {
    private final int dzienSpotkania;
    private final Agent agent1;
    private final Agent agent2;
    
    public Spotkanie(int dzien, Agent a1, Agent a2) {
        this.dzienSpotkania = dzien;
        agent1 = a1;
        agent2 = a2;
    }
    
    // Jeżeli któryś ze spotykających się agentów jest zarażony a drugi nie ma odporności,
    // to z prawd. prawdZarażenia może dojść do zarażenia, wpp. takie spotkanie nie ma żadnego efektu. 
    // Zwraca true jeśli doszło do zarażenia, wpp false.
    public boolean przeprowadzSpotkanie(double prawdZarazenia, Random rand) {
        if (agent1.jestMartwy() || agent2.jestMartwy()) // Do czasu spotkania zmarł jeden z agentow.
            return false;
        if ((agent1.jestChory() || agent2.jestChory()) && (agent1.jestZdrowy() || agent2.jestZdrowy())) {
            if (rand.nextDouble() > prawdZarazenia)
                return false;
            if (agent1.jestZdrowy())
                agent1.ustawStan(Stan.CHORY);
            else 
                agent2.ustawStan(Stan.CHORY);
            return true;
        }
        return false;
    }
    
    public int dajDzien() {
        return dzienSpotkania;
    }
}
