package agent;

import java.util.Random;

public class AgentZwykly extends Agent {
    
    public AgentZwykly(int id) {
        super(id);
    }
    
    @Override
    public boolean czyIdzieNaSpotkanie(double prawdSpotkania, Random rand) {
        if (this.jestChory()) // Dopóki nie wyzdrowieje planuje nowe spotkania z dwa razy mniejszym prawdopodobieństwem.
            prawdSpotkania /= 2;
        return rand.nextDouble() < prawdSpotkania;
    }
    
    @Override
    public String toString() {
        return super.toString() + "zwykly";
    }
}
