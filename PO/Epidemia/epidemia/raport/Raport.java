package raport;

import agent.Agent;
import java.io.IOException;
import java.io.Writer;
import java.util.Properties;
import java.util.Set;

public class Raport {
    private final Writer raport;
    
    public Raport(Writer raport) {
        this.raport = raport;
    }
    
    public void wypiszParametry(Properties mergedProp) throws IOException {
        raport.write("# twoje wyniki powinny zawierać te komentarze\n");
        Set<String> keys = mergedProp.stringPropertyNames();
        for(String k:keys)
            raport.write(k+"="+mergedProp.getProperty(k) + "\n");
        raport.write("\n");
    }
    
    public void wypiszAgentow(Agent[] agenci) throws IOException {
        raport.write("# agenci jako: id typ lub id* typ dla chorego\n");
        for (Agent agent: agenci)
            raport.write(agent + "\n");
        raport.write("\n");
    }
    
    public void wypiszGraf(Agent[] agenci) throws IOException {
        raport.write("# graf\n");
        for (Agent agent : agenci) {
            raport.write("" + (agent.dajId()+1) );
            for (Agent sasiadAgenta : agent.dajSasiadow())
                raport.write(" " + (sasiadAgenta.dajId()+1));
            raport.write("\n");
        }
        raport.write("\n");
    }
    
    public void wypiszStatystykiZdrowotne(int lZdrowych, int lChorych, int lOdpornych) throws IOException {
        raport.write("zdrowi" + lZdrowych + " chorzy" + lChorych + " uodp" + lOdpornych + "\n");
    }
    
    public Writer dajWriter() {
        return raport;
    }
}
