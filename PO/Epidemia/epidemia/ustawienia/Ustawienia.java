package ustawienia;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.Reader;
import static java.lang.System.exit;
import java.nio.channels.Channels;
import java.nio.charset.MalformedInputException;
import java.nio.charset.StandardCharsets;
import java.util.Properties;

class DaneExcept extends Exception {
    private final String klucz;
    private final String wiadomosc;
    
    public DaneExcept(String wiadomosc, String klucz) {
        this.klucz = klucz;
        this.wiadomosc = wiadomosc;
    }
    
    @Override
    public String toString() {
        return wiadomosc + " dla klucza " + klucz;
    }
}

public class Ustawienia {
    private Properties mergedProp; 
    private final String SEED = "seed";
    private final String LICZBA_AGENTÓW = "liczbaAgentów";
    private final String PRAWD_TOWARZYSKI = "prawdTowarzyski";
    private final String PRAWD_SPOTKANIA = "prawdSpotkania";
    private final String PRAWD_ZARAŻENIA = "prawdZarażenia";
    private final String PRAWD_WYZDROWIENIA = "prawdWyzdrowienia";
    private final String ŚMIERTELNOŚĆ = "śmiertelność";
    private final String LICZBA_DNI = "liczbaDni";
    private final String ŚR_ZNAJOMYCH = "śrZnajomych";
    private final String PLIK_Z_RAPORTEM = "plikZRaportem";
    private final String DEFAULT_PROPERTIES = "default.properties";
    private final String SIMULATION_CONFIG = "simulation-conf.xml";
    
    private FileInputStream stworzFileInputStream(String sciezka, String nazwaPliku) {
        FileInputStream file = null;
        try {
            file = new FileInputStream(sciezka + "/" + nazwaPliku);
        } catch(FileNotFoundException e) {
            System.out.println("Brak pliku " + nazwaPliku);
            exit(0);
        }
        return file;
    }
    
    // Zwraca inta odpowiadajacego podanemu kluczowi z pliku Properties. 
    // Jeżeli wartość nie należy do do podanego w argumencie zakresu zgłaszany jest błąd.
    private int dajLiczbę(String klucz, int dolnyLimit, int gornyLimit, Properties prop) {
        String str = null;
        int liczba = -1; 
        try {
            str = prop.getProperty(klucz);
            if (str == null)
                throw new DaneExcept("Brak wartości", klucz);
            liczba = Integer.parseInt(str);
            if (liczba < dolnyLimit || liczba > gornyLimit)
                throw new DaneExcept("Niedozwolona wartość \"" + liczba + "\"", klucz);
        } catch (NumberFormatException e) {
            System.out.println("Niedozwolona wartość \"" + str + "\" dla klucza " + klucz);
            exit(0);
        } catch (DaneExcept e) {
            System.out.println(e);
            exit(0);
        }
        return liczba;
    }

    public int dajLiczbęAgentów() {
        return dajLiczbę(LICZBA_AGENTÓW, 0, 1000000, mergedProp);
    }
    
    public int dajLiczbęDni() {
        return dajLiczbę(LICZBA_DNI, 1, 1000, mergedProp);
    }
    
    public int dajLiczbęŚrZnajomych() {
        return dajLiczbę(ŚR_ZNAJOMYCH, 0, dajLiczbęAgentów() - 1, mergedProp);
    }
    
    // Zwraca wartość prawdopodobieństwa dla danego klucza z pliku Properties.
    // Jeżeli wartość nie jest prawdopodobieństwem zgłasz błąd.
    private double dajPrawd(String klucz, Properties prop) {
        String str = null;
        double prawd = -1; 
        try {
            str = prop.getProperty(klucz);
            if (str == null)
                throw new DaneExcept("Brak wartości", klucz);
            prawd = Double.parseDouble(str);
            if (prawd < 0 || prawd > 1)
                throw new DaneExcept("Niedozwolona wartość \"" + prawd + "\"", klucz);
        } catch (NumberFormatException e) {
            System.out.println("Niedozwolona wartość \"" + str + "\" dla klucza " + klucz);
            exit(0);
        } catch (DaneExcept e) {
            System.out.println(e);
            exit(0);
        }
        return prawd;
    }
    
    public double dajPrawdTowarzyski() {
        return dajPrawd(PRAWD_TOWARZYSKI, mergedProp);
    }
    
    public double dajPrawdSpotkania() {
        return dajPrawd(PRAWD_SPOTKANIA, mergedProp);
    }
    
    public double dajPrawdZarażenia() {
        return dajPrawd(PRAWD_ZARAŻENIA, mergedProp);
    }
    
    public double dajPrawdWyzdrowienia() {
        return dajPrawd(PRAWD_WYZDROWIENIA, mergedProp);
    }
    
    public double dajŚmiertelność() {
        return dajPrawd(ŚMIERTELNOŚĆ, mergedProp);
    }
    
    public String dajPlikZRaportem(Properties prop) {
        String plikZRaportem = null;
        try {
            plikZRaportem = prop.getProperty(PLIK_Z_RAPORTEM);
            if (plikZRaportem == null || plikZRaportem.isEmpty())
                throw new DaneExcept("Brak wartości", PLIK_Z_RAPORTEM);
        } catch (DaneExcept e) {
            System.out.println(e);
            exit(0);
        }
      return plikZRaportem;  
    }
    
    public long dajSeed(Properties prop) {
        long seed = -1;
        String klucz = SEED, str = null;
        try {
            str = prop.getProperty(klucz);
            if (str == null)
                throw new DaneExcept("Brak wartości", klucz);
            seed = Long.parseLong(str);
        } catch (NumberFormatException e) {
            System.out.println("Niedozwolona wartość \"" + str + "\" dla klucza " + klucz);
            exit(0);
        } catch (DaneExcept e) {
            System.out.println(e);
            exit(0);
        }
        return seed;
    }
    
    // Sprawdza poprawność wartości kluczy pliku Properties.
    // W przypadku braku klucza nie jest zgłaszany bład.
    public void sprawdzProperties(Properties prop, boolean defaultProp) {
        if (defaultProp || prop.getProperty(SEED) != null)
            dajSeed(prop);
        if (defaultProp || prop.getProperty(PRAWD_TOWARZYSKI) != null)
            dajPrawd(PRAWD_TOWARZYSKI, prop);
        if (defaultProp || prop.getProperty(PRAWD_SPOTKANIA) != null)
            dajPrawd(PRAWD_SPOTKANIA, prop);
        if (defaultProp || prop.getProperty(PRAWD_ZARAŻENIA) != null)
            dajPrawd(PRAWD_ZARAŻENIA, prop);
        if (defaultProp || prop.getProperty(PRAWD_WYZDROWIENIA) != null)
            dajPrawd(PRAWD_WYZDROWIENIA, prop);
        if (defaultProp || prop.getProperty(ŚMIERTELNOŚĆ) != null)
            dajPrawd(ŚMIERTELNOŚĆ, prop);
        int liczbaAgentow = Integer.MAX_VALUE;
        if (defaultProp || prop.getProperty(LICZBA_AGENTÓW) != null)
            liczbaAgentow = dajLiczbę(LICZBA_AGENTÓW, 0, 1000000, prop);
        if (defaultProp || prop.getProperty(LICZBA_DNI) != null)
            dajLiczbę(LICZBA_DNI, 1, 1000, prop);
        if (defaultProp || prop.getProperty(ŚR_ZNAJOMYCH) != null)
            dajLiczbę(ŚR_ZNAJOMYCH, 0, liczbaAgentow, prop); 
        if (defaultProp || prop.getProperty(PLIK_Z_RAPORTEM) != null)
            dajPlikZRaportem(prop);
    } 
    
    public void wczytajPlikiProperties() throws FileNotFoundException, IOException {
        String rootPath = System.getProperty("user.dir");
        Properties defaultProp = new Properties();
        Properties simulConfProp = new Properties();
        FileInputStream streamDefault = null, streamSimulationConfig = null;
        Reader readerDef = null;
        try {
            streamDefault = stworzFileInputStream(rootPath, DEFAULT_PROPERTIES);
            streamSimulationConfig = stworzFileInputStream(rootPath, SIMULATION_CONFIG);
            readerDef = Channels.newReader(streamDefault.getChannel(), StandardCharsets.UTF_8.name());
            defaultProp.load(readerDef);
            simulConfProp.loadFromXML(streamSimulationConfig);
        } catch (MalformedInputException e) {
            System.out.println("default.properties nie jest tekstowy");
            exit(0);
        } catch (IOException e) { 
            System.out.println("simulation-conf.xml nie jest XML");
            exit(0);
        } finally {
            if (streamDefault != null)
                streamDefault.close();
            if (streamSimulationConfig != null)
                streamSimulationConfig.close();
            if (readerDef != null)
                readerDef.close();
        }
        sprawdzProperties(defaultProp, true);
        sprawdzProperties(simulConfProp, false);
        mergedProp = new Properties();
        mergedProp.putAll(defaultProp);
        mergedProp.putAll(simulConfProp);
    }
    
    public Properties dajProperties() {
        return mergedProp;
    }
}