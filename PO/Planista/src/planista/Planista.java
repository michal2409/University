package planista;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Locale;
import java.util.Scanner;

class sortCzasPojaw implements Comparator<Proces> {
    @Override
    public int compare(Proces p1, Proces p2) {
        if (p1.dajCzPoj() > p2.dajCzPoj())
            return 1;
        if (p1.dajCzPoj() == p2.dajCzPoj() && p1.dajId() > p2.dajId())
            return 1;
        return -1;
    }
}

class sortCzasZak implements Comparator<Proces> {
    @Override
    public int compare(Proces p1, Proces p2) {
        if (Planista.czyRowneDouble(p1.dajCzZak(), p2.dajCzZak()) && p1.dajId() > p2.dajId())
            return 1;
        if (Planista.porownajDouble(p1.dajCzZak(), p2.dajCzZak()) > 0)
            return 1;
        return -1;
    }
}

class DaneExcept extends Exception {
    private final int nrLinii;
    private final String wiadomosc;
    
    public DaneExcept(int nrLinii, String wiadomosc) {
        this.nrLinii = nrLinii;
        this.wiadomosc = wiadomosc;
    }
    
    @Override
    public String toString() {
        return "Błąd w wierszu " + nrLinii + ": " + wiadomosc + ".";
    }
}

public class Planista {
    private static int nrLinii = 0;
    private final static double EPSILON = 0.0000000000001d;
    
    public static boolean czyRowneDouble(final double a, final double b) {
        if (a == b) 
            return true;
        return Math.abs(a - b) < EPSILON;
    }

    public static int porownajDouble(final double a, final double b) {
        return czyRowneDouble(a, b) ? 0 : (a < b) ? -1 : +1;
    }
    
    private static int wczytajLiczbe(Scanner in) throws DaneExcept {
        int[] wczytLiczba = wczytajNLiczb(in, 1);
        return wczytLiczba[0];
    }
    
    private static int[] wczytajNLiczb(Scanner in, int n) throws DaneExcept {
        nrLinii++;
        if (!in.hasNextLine())
            throw new DaneExcept(nrLinii, "brak linii");
        Scanner lineScanner = new Scanner(in.nextLine());
        int [] wczytLiczby = new int[n];
        for (int i = 0; i < n; i++) {
            if (!lineScanner.hasNext()) 
                throw new DaneExcept(nrLinii, "za mało danych");
            if (!lineScanner.hasNextInt()) 
                throw new DaneExcept(nrLinii, "podano wartość nie będącą liczbą całkowitą");
            wczytLiczby[i] = lineScanner.nextInt(); 
        }
        if (lineScanner.hasNext()) 
            throw new DaneExcept(nrLinii, "podano za dużo wartości");
        return wczytLiczby;
    } 
    
    public static void wypiszWynikiSymulacji(Strategia str, Proces[] procesy) {
        Arrays.sort(procesy, new sortCzasZak());
        System.out.println("Strategia: " + str);
        for (Proces p : procesy)
            System.out.format(Locale.US, "[" + p.dajId() + " " + p.dajCzPoj() + " %.2f" + "]", p.dajCzZak());
        System.out.println();
        System.out.format(Locale.US, "Średni czas obrotu: %.2f%n", str.obliczSrCzasObrotu());
        System.out.format(Locale.US, "Średni czas oczekiwania: %.2f%n", str.obliczSrCzasOczekiwania());
        System.out.println();
    }

    public static void main(String[] args) {
        Scanner in = null;
        Proces[] procesy = null;
        int[] parametryQ = null;
        try {
            in = (args.length == 0) ? new Scanner(System.in) : new Scanner(new File(args[0]));
           
            // Wczytywanie liczby procesow.
            int liczbaProcesow = wczytajLiczbe(in);
            if (liczbaProcesow <= 0)
                throw new DaneExcept(nrLinii, "podano liczbę procesów mniejszą lub równą 0");
            
            // Wczytanie procesów.
            procesy = new Proces[liczbaProcesow];
            for (int i = 0; i < liczbaProcesow; i++) {
                int[] wczytLiczby = wczytajNLiczb(in, 2);
                if (wczytLiczby[0] < 0)
                    throw new DaneExcept(nrLinii, "podano czas mniejszy od 0");
                if (wczytLiczby[1] < 1)
                    throw new DaneExcept(nrLinii, "podano zapotrzebowanie mniejsze od 1");
                procesy[i] = new Proces(i+1, wczytLiczby[0], wczytLiczby[1]);
            }

            // Wczytanie liczby procesow RR.
            int liczbaProcesowRR = wczytajLiczbe(in);
            if (liczbaProcesowRR <= 0)
                    throw new DaneExcept(nrLinii, "podano liczbę procesów RR mniejszą lub równą od 0");
            
            // Wczytanie parametrow q dla procesow RR.
            parametryQ = wczytajNLiczb(in, liczbaProcesowRR);
            for (int q : parametryQ)
                if (q <= 0)
                    throw new DaneExcept(nrLinii, "podano parametr q mniejszy lub równy 0");
            if (in.hasNextLine())
                throw new DaneExcept(nrLinii+1, "podano nadmiarową linię");
        } catch (DaneExcept ex) {
            System.out.println(ex);
            System.exit(0);
        } catch (FileNotFoundException e) {
            System.out.println("Plik z danymi nie jest dostępny.");
            System.exit(0);
        } finally {
            if(in != null)
                in.close();
        }

        Strategia[] strategie = {new StrFCFS(), new StrSJF(), new StrSRT(), new StrPS()};
        StrRR strRR = new StrRR();

        Arrays.sort(procesy, new sortCzasPojaw());
        for (Strategia str : strategie) 
            wypiszWynikiSymulacji(str, str.symuluj(procesy));
        for (int q : parametryQ) 
            wypiszWynikiSymulacji(strRR, strRR.symuluj(procesy, q));
    }
}