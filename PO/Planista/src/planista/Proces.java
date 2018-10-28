package planista;

public class Proces {
    private final int id;
    private final int czasPojawienia;
    private final int zapotrzebowanie;
    private double czasPracy;
    private double czasZakonczenia;

    public Proces(int id, int czasPojaw, int zapotrz) {
        this.id = id;
        czasPojawienia = czasPojaw;
        zapotrzebowanie = zapotrz;
    }
    
    public Proces(Proces p) {
        this(p.dajId(), p.dajCzPoj(), p.dajZap());
    }

    public int dajId() {
        return id;
    }
        
    public int dajCzPoj() {
        return czasPojawienia;
    }
    
    public int dajZap() {
        return zapotrzebowanie;
    }
    
    public double dajCzZak() {
        return czasZakonczenia;
    }
    
    public double dajCzPr() {
        return czasPracy;
    }
    
    public void ustawCzZak(double t) {
        czasZakonczenia = t;
    }
    
    public void zwiekszCzPr(double t) {
        czasPracy += t;
    }
    
    public double dajPozostZapot() {
        return zapotrzebowanie - czasPracy;
    }
}
