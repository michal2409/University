package planista;
import java.util.LinkedList;

public abstract class Strategia {
    protected LinkedList<Proces> kolejkaProcesow;
    protected Proces[] procesy;
    protected int idxNastProc;
    protected double czas;
    
    public abstract void przydzielProcesor();
    
    // Dodaje do kolejki procesów procesy przybywające do aktualnego czasu.
    protected void uzupelnijKolejke() {
        while (idxNastProc < procesy.length && procesy[idxNastProc].dajCzPoj() <= czas)
            kolejkaProcesow.add(procesy[idxNastProc++]);           
    }
    
    public Proces[] symuluj(Proces[] procesy) {
        // Kopiowanie tablicy procesow.
        this.procesy = new Proces[procesy.length];
        for (int i = 0; i < procesy.length; i++) 
            this.procesy[i] = new Proces(procesy[i]);
        
        kolejkaProcesow = new LinkedList<>();  
        idxNastProc = 0;
        czas = 0.0;
        
        while (idxNastProc < this.procesy.length || !kolejkaProcesow.isEmpty()) { // Są jeszcze procesy do zrealizowania.
            uzupelnijKolejke();
            if (kolejkaProcesow.isEmpty()) { // Pusta kol. procesow, czekamy aż pojawi się jakiś proces.
                czas = this.procesy[idxNastProc].dajCzPoj();
                uzupelnijKolejke();
            }
            przydzielProcesor();
        }
        return this.procesy;
    }
    
    // Zwraca proces o najmniejszym pozostałym zapotrzebowaniu na procesor.
    // W przypadku remisu wybiera proces o mniejszym id.
    protected Proces wybierzMinPozostZapotProces() {
        Proces minZapProces = kolejkaProcesow.peekFirst();
        for (Proces p: kolejkaProcesow) 
            if ((Planista.czyRowneDouble(minZapProces.dajPozostZapot(), p.dajPozostZapot()) && minZapProces.dajId() > p.dajId())
                    || (minZapProces.dajPozostZapot() > p.dajPozostZapot()))
                minZapProces = p;
        kolejkaProcesow.remove(minZapProces);
        return minZapProces;
    }
    
    public double obliczSrCzasObrotu() {
        double czObrotu = 0.0;
        for (Proces p : procesy) 
            czObrotu += p.dajCzZak() - p.dajCzPoj();
        return czObrotu/procesy.length;
    }
    
    public double obliczSrCzasOczekiwania() {
        double czOczek = 0.0;
        for (Proces p : procesy) 
            czOczek += p.dajCzZak() - p.dajZap() - p.dajCzPoj();
        return czOczek/procesy.length;
    }
}