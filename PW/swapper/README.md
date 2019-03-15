<h2 id="wprowadzenie">Wprowadzenie</h2>
<p><em>Swapper</em> to zbiór, na którym można wykonać niepodzielną operację usunięcia a następnie dodania obiektów.</p>
<p>W pakiecie <code>swapper</code>, implementującym w Javie swapper wartości typu <code>E</code>, jest definicja klasy <code>Swapper&lt;E&gt;</code>:</p>
<div class="sourceCode" id="cb1"><pre class="sourceCode java"><code class="sourceCode java"><a class="sourceLine" id="cb1-1" title="1"><span class="kw">package</span><span class="im"> swapper;</span></a>
<a class="sourceLine" id="cb1-2" title="2"></a>
<a class="sourceLine" id="cb1-3" title="3"><span class="kw">import</span><span class="im"> java.util.Collection;</span></a>
<a class="sourceLine" id="cb1-4" title="4">...</a>
<a class="sourceLine" id="cb1-5" title="5"></a>
<a class="sourceLine" id="cb1-6" title="6"><span class="kw">public</span> <span class="kw">class</span> Swapper&lt;E&gt; {</a>
<a class="sourceLine" id="cb1-7" title="7"></a>
<a class="sourceLine" id="cb1-8" title="8">    <span class="kw">public</span> <span class="fu">Swapper</span>() {</a>
<a class="sourceLine" id="cb1-9" title="9">        ...</a>
<a class="sourceLine" id="cb1-10" title="10">    }</a>
<a class="sourceLine" id="cb1-11" title="11"></a>
<a class="sourceLine" id="cb1-12" title="12">    <span class="kw">public</span> <span class="dt">void</span> <span class="fu">swap</span>(<span class="bu">Collection</span>&lt;E&gt; removed, <span class="bu">Collection</span>&lt;E&gt; added) <span class="kw">throws</span> <span class="bu">InterruptedException</span> {</a>
<a class="sourceLine" id="cb1-13" title="13">        ...</a>
<a class="sourceLine" id="cb1-14" title="14">    }</a>
<a class="sourceLine" id="cb1-15" title="15"></a>
<a class="sourceLine" id="cb1-16" title="16">    ...</a>
<a class="sourceLine" id="cb1-17" title="17"></a>
<a class="sourceLine" id="cb1-18" title="18">}</a></code></pre></div>
<p>Bezparametrowy konstruktor tworzy swapper, który w stanie początkowym jest pusty.</p>
<p>Metoda <code>swap(removed, added)</code> wstrzymuje wątek do chwili, gdy w swapperze będą wszystkie elementy kolekcji <code>removed</code>. Następnie, niepodzielnie:</p>
<ol type="1">
<li><p>usuwa ze swappera wszystkie elementy kolekcji <code>removed</code>, po czym</p></li>
<li><p>dodaje do swappera wszystkie elementy kolekcji <code>added</code>.</p></li>
</ol>
<p>Kolekcje <code>removed</code> i <code>added</code> mogą mieć niepuste przecięcie.</p>
<p>Elementy swappera nie powtarzają się. Dodanie do swappera obiektu, który już w nim jest, nie ma żadnego efektu.</p>
<p>Zarówno kolekcja <code>removed</code> jak i <code>added</code> może mieć powtórzenia. Nie wpływają one na działanie metody.</p>
<p>W przypadku przerwania wątku metoda zgłasza wyjątek <code>InterruptedException</code>.</p>
<p>Przerwane wykonanie metody nie zmienia zawartości swappera.</p>
<p>Przerwanie wątku korzystającego ze swappera nie wpływa na poprawność działania pozostałych wątków.</p>
<h2 id="polecenie">Polecenie</h2>
<ul>
<li><p>(8 pkt)</p>
<p>Zaimplementuj w Javie swapper zgodny z powyższą specyfikacją. Do pakietu <code>swapper</code> dołącz wszystkie potrzebne definicje pomocnicze.</p></li>
<li><p>(2 pkt)</p>
<p>Napisz dwa programy przykładowe, demonstrujące zastosowanie swappera do rozwiązania:</p>
<ul>
<li><p>problemu producentów i konsumentów ze skończonym buforem wieloelementowym,</p></li>
<li><p>problemu czytelników i pisarzy.</p></li>
</ul>
<p>W programach przykładowych, oprócz swappera, nie należy używać żadnych innych mechanizmów synchronizacji.</p></li>
</ul>
<h2 id="uwagi">Uwagi</h2>
<p>Program ma być w wersji 8 języka Java. Powinien się kompilować i działać poprawnie na komputerze <code>students</code>.</p>
<p>Wolno korzystać tylko ze standardowych pakietów zainstalowanych na <code>students</code>.</p>
<p>Implementacja nie musi gwarantować, że wątek nie zostanie zagłodzony przez inne wątki korzystające za swappera.</p>
<p>Jako rozwiązanie należy wysłać na moodla plik <code>ab123456.tar.gz</code>, gdzie <code>ab123456</code> to login na <code>students</code>.</p>
<p>W wysłanym pliku <code>.tar.gz</code> ma być katalog <code>swapper</code> z plikami źródłowymi <code>.java</code>.</p>
<h2 id="walidacja">Walidacja</h2>
<p>Rozwiązania zostaną poddane walidacji, wstępnie sprawdzającej zgodność ze specyfikacją.</p>
<p>Na komputerze <code>students</code>, w katalogu walidacji, będzie:</p>
<ul>
<li><p>podkatalog <code>packed</code> z rozwiązaniami,</p></li>
<li><p>plik <a href="https://moodle.mimuw.edu.pl/mod/resource/view.php?id=11676">validate.sh</a>,</p></li>
<li><p>podkatalog <code>validate</code> plikiem z <a href="https://moodle.mimuw.edu.pl/mod/resource/view.php?id=11677">validate/Validate.java</a>.</p></li>
</ul>
<p>Polecenie</p>
<div class="sourceCode" id="cb2"><pre class="sourceCode bash"><code class="sourceCode bash"><a class="sourceLine" id="cb2-1" title="1"><span class="fu">sh</span> validate.sh ab123456</a></code></pre></div>
<p>przeprowadzi walidację rozwiązania studenta o identyfikatorze <code>ab123456</code>. Komunikat <code>OK</code> poinformuje o sukcesie.</p>
<p>Rozwiązania, które pomyślnie przejdą walidację, zostaną dopuszczone do testów poprawności.</p>
<hr>
