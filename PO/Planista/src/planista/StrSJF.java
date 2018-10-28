package planista;

public class StrSJF extends Strategia {  
    @Override
    public void przydzielProcesor() {
        Proces p = wybierzMinPozostZapotProces();
        czas += p.dajZap();
        p.ustawCzZak(czas);
    }
    
    @Override
    public String toString() {
        return "SJF";
    }
}
