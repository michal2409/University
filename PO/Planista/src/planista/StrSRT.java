package planista;

public class StrSRT extends Strategia {  
    @Override
    public void przydzielProcesor() {
        Proces p = wybierzMinPozostZapotProces(); 
        while(Planista.porownajDouble(p.dajPozostZapot(), 0.0) > 0) {
            czas++;
            p.zwiekszCzPr(1);
            
            while (idxNastProc < procesy.length && procesy[idxNastProc].dajCzPoj() <= czas) { // Pobieranie nowych procesów.
                if (Planista.porownajDouble(procesy[idxNastProc].dajPozostZapot(), p.dajPozostZapot()) < 0) { // Przychodzący proces deklaruje mniejsze zapotrzebowanie.
                    kolejkaProcesow.add(p);                                                                   // zabiera procesor obecnemu procesowi.
                    p = procesy[idxNastProc++];
                } else 
                    kolejkaProcesow.add(procesy[idxNastProc++]);
            }
        }
        p.ustawCzZak(czas);
    }
    
    @Override
    public String toString() {
        return "SRT";
    }
}
