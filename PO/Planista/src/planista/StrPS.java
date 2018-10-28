package planista;

import static java.lang.Math.floor;
import java.util.Comparator;
import java.util.ListIterator;

class sortPozostCzasPracy implements Comparator<Proces> {
    @Override
    public int compare(Proces p1, Proces p2) {
        if (Planista.czyRowneDouble(p1.dajPozostZapot(), p2.dajPozostZapot()))
            return 0;
        if (Planista.porownajDouble(p1.dajPozostZapot(), p2.dajPozostZapot()) > 0)
            return 1;
        return -1;
    }
}

public class StrPS extends Strategia {
    private void przepracujOkresCzasu(double pracaNaJednCzasu, double przepracowanyCzas) {
        for (Proces p : kolejkaProcesow)
            p.zwiekszCzPr(pracaNaJednCzasu*przepracowanyCzas);
        czas += przepracowanyCzas;
    }

    @Override
    public void przydzielProcesor() {
        while(!kolejkaProcesow.isEmpty()) {
            kolejkaProcesow.sort(new sortPozostCzasPracy());
            double zuzycieProcNaJednCzasu = 1.0/kolejkaProcesow.size();
            ListIterator<Proces> i = kolejkaProcesow.listIterator();
            Proces p = i.next();
            
            while (Planista.porownajDouble(p.dajPozostZapot(), zuzycieProcNaJednCzasu) <= 0) { // Procesy kończące się w obecnej jedn. czasu.
                double przyrostCzDoZakProc = p.dajPozostZapot()/(zuzycieProcNaJednCzasu);
                if (floor(czas + 1.0) <= czas + przyrostCzDoZakProc) // Czas zakonczenia procesu przekracza nastepna jednostke czasu.
                    break;
                
                p.ustawCzZak(czas + przyrostCzDoZakProc);
                i.remove();
                przepracujOkresCzasu(zuzycieProcNaJednCzasu, przyrostCzDoZakProc);
                zuzycieProcNaJednCzasu = 1.0/kolejkaProcesow.size();
                if (!i.hasNext())
                    break;
                p = i.next();
            }
            // Przepracowanie okresu dopełniającego czas do pełnej jednostki.
            przepracujOkresCzasu(zuzycieProcNaJednCzasu, floor(czas + 1.0) - czas);
            uzupelnijKolejke();
        }  
    }
    
    @Override
    public double obliczSrCzasOczekiwania() {
        return 0.0;
    }
    
    @Override
    public String toString() {
        return "PS";
    }
}
