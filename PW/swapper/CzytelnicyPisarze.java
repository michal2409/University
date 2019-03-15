package swapper;

import java.util.ArrayList;
import java.util.List;

public class CzytelnicyPisarze {

    private static int CZYTELNICY = 200;
    private static int PISARZE = 200;
    private static volatile int iluCzyta = 0, iluPisze = 0, czekaCzyt = 0, czekaPis = 0;
    private static BinarnySemafor ochrona;
    private static Semafor czytelnicy;
    private static Semafor pisarze;

    private static class Czytelnicy implements Runnable {

        private void czytam(int id) {
            System.out.println("Czytam " + id);
        }

        @Override
        public void run() {
            Thread t = Thread.currentThread();
            try {
                for (int i = 0; i < CZYTELNICY; i++) {
                    ochrona.P();
                    if (iluPisze + czekaPis > 0) {
                        czekaCzyt++;
                        ochrona.V();
                        czytelnicy.P(); // dziedziczenie sekcji krytycznej
                        czekaCzyt--;
                    }
                    iluCzyta++;
                    if (czekaCzyt > 0)
                        czytelnicy.V();
                    else
                        ochrona.V();
                    czytam(i);
                    ochrona.P();
                    iluCzyta--;
                    if ((iluCzyta == 0) && (czekaPis > 0))
                        pisarze.V();
                    else
                        ochrona.V();
                }
            } catch (InterruptedException e) {
                t.interrupt();
                System.err.println(t.getName() + " przerwany");
            }
        }
    }

    private static class Pisarze implements Runnable {

        private void pisze(int id) {
            System.out.println("Pisze " + id);
        }

        @Override
        public void run() {
            Thread t = Thread.currentThread();
            try {
                for (int i = 0; i < PISARZE; i++) {
                    ochrona.P();
                    if (iluPisze + iluCzyta > 0) {
                        czekaPis++;
                        ochrona.V();
                        pisarze.P(); // dziedziczenie sekcji krytycznej
                        czekaPis--;
                    }
                    iluPisze++;
                    ochrona.V();
                    pisze(i);
                    ochrona.P();
                    iluPisze--;
                    if (czekaCzyt > 0)
                        czytelnicy.V();
                    else if (czekaPis > 0)
                        pisarze.V();
                    else
                        ochrona.V();
                }
            } catch (InterruptedException e) {
                t.interrupt();
                System.err.println(t.getName() + " przerwany");
            }
        }
    }

    public static void main(String[] args) {
        try {
            ochrona = new BinarnySemafor(1);
            czytelnicy = new Semafor(0);
            pisarze = new Semafor(0);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("Główny przerwany");
        }

        Thread c = new Thread(new Czytelnicy());
        Thread p = new Thread(new Pisarze());

        p.start();
        c.start();

        try {
            c.join();
            p.join();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("Główny przerwany");
        }
    }
}

