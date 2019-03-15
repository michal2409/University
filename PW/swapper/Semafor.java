package swapper;

import java.util.concurrent.atomic.AtomicInteger;

public class Semafor {

	private volatile AtomicInteger licznik;
	private volatile BinarnySemafor ochrona;
	private volatile BinarnySemafor bramka;

	public Semafor(int s) throws InterruptedException {
		licznik = new AtomicInteger(s);
		ochrona = new BinarnySemafor(1);
		bramka = new BinarnySemafor(Math.min(s, 1));
	}

	public void V() throws InterruptedException {
		ochrona.P();
		if (licznik.incrementAndGet() == 1)
			bramka.V();
		ochrona.V();
	}

	public void P() throws InterruptedException {
		bramka.P();
		ochrona.P();
		if (licznik.decrementAndGet() > 0)
			bramka.V();
		ochrona.V();
	}
}

