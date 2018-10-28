package planista;

public class StrFCFS extends Strategia {
    @Override
    public void przydzielProcesor() {
        Proces p = kolejkaProcesow.pollFirst();
        czas += p.dajZap();
        p.ustawCzZak(czas);
    }
    
    @Override
    public String toString() {
        return "FCFS";
    }
}
