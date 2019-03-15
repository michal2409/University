package swapper;

import java.util.Collections;
import java.util.Collection;

public class BinarnySemafor {

	private volatile Swapper<Integer> swapper;
	private static Collection<Integer> empty = Collections.emptySet();
	private static Collection<Integer> singletonOne = Collections.singleton(1);

	public BinarnySemafor(int s) throws InterruptedException {
		swapper = new Swapper();
		if (s == 1) // ustawienie semafora na 1
			swapper.swap(empty, singletonOne);
	}

	public void V() throws InterruptedException {
		swapper.swap(empty, singletonOne);
	}

	public void P() throws InterruptedException {
		swapper.swap(singletonOne, empty);
	}
}

