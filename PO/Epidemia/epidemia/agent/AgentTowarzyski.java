package agent;

import spotkanie.Spotkanie;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;

public class AgentTowarzyski extends Agent {
    public AgentTowarzyski(int id) {
        super(id);
    }
    
    // Losuje jednego ze swoich znajomych i znajomych swoich znajomych i umawia spotkanie.
    @Override
    public void umówSpotkanie(int dzieńSpotkania, Random rand) {
        if (this.jestChory()) { // Dopóki nie wyzdrowieje planuje się spotykać tylko ze swoimi bezpośrednimi znajomymi.
            super.umówSpotkanie(dzieńSpotkania, rand);
            return;
        }
        HashSet<Agent> kandydaciNaSpotkanie = new HashSet<>(sasiedzi);
        for (Agent a: sasiedzi)
            kandydaciNaSpotkanie.addAll(a.dajSasiadow());
                int index = rand.nextInt(sasiedzi.size());
        Iterator<Agent> iter = kandydaciNaSpotkanie.iterator();
        for (int i = 0; i < index; i++) 
            iter.next();
        spotkania.add(new Spotkanie(dzieńSpotkania, this, iter.next()));
    }
    
    @Override
    public String toString() {
        return super.toString() + "towarzyski";
    }
}
