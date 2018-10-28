package planista;

public class StrRR extends Strategia {
    private int paramQ;

    public Proces[] symuluj(Proces[] procesy, int paramQ) {
        this.paramQ = paramQ;
        return symuluj(procesy);
    }

    @Override
    public void przydzielProcesor() {
        Proces p = kolejkaProcesow.pollFirst();
        int czasPracy = Math.min((int)p.dajPozostZapot(), paramQ);
        p.zwiekszCzPr(czasPracy);
        
        if (Planista.czyRowneDouble(p.dajPozostZapot(), 0.0)) {
            czas += czasPracy;
            p.ustawCzZak(czas);
        }
        else { // Proces nie skończył pracy, wraca do kolejki.
            czas += czasPracy - 1;
            uzupelnijKolejke();
            kolejkaProcesow.add(p);
            czas++;
        }
        uzupelnijKolejke();
    }
    
    @Override
    public String toString() {
        return "RR-" + paramQ;
    }
}
