package swapper;

import java.util.Collection;
import java.util.HashSet;

public class Swapper<E> {

	private volatile HashSet<E> swapper;

	public Swapper() {
		swapper = new HashSet<E>();
	}

	public void swap(Collection<E> removed, Collection<E> added) throws InterruptedException {
		HashSet<E> swapperKopia = null;
		try {
			synchronized (this) {
				while (!swapper.containsAll(removed))
					wait();
				swapperKopia = new HashSet<>(swapper);
				swapper.removeAll(removed);
				swapper.addAll(added);
				notifyAll();
			}
		}
		catch (Exception e) {
			if (swapperKopia != null)
				swapper = swapperKopia;
			throw e;
		}
	}
}

