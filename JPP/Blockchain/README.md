<h2>A. Drzewa skrótów</h2>

<p>Drzewo skrótów (Merkle tree), są strukturą danych,
która umożliwia zwięzłe reprezentowanie potencjalnie dużych ilości danych
za pomocą ich skrótów, w sposób umożliwiający niepodważalne wykazanie, że pewien element należy do drzewa o danym skrócie.</p>

<p>Popularny ich wariant (stosowany np. przez Bitcoin),
który rozważamy w tym zadaniu,
to zrównoważone drzewo binarne, w którym liście zawierają dane,
natomiast każdy wierzchołek wewnętrzny zawiera skrót skrótów swoich potomków.
Wierzchołki, które mają jednego potomka, dla celów obliczania skrótu
traktujemy tak jakby miały dwóch potomków o identycznym skrócie.</p>

<p>Algorytm budowy drzewa skrótów dla niepustej listy elementów można opisać następująco (por. <a href="https://en.bitcoin.it/wiki/Protocol%5Fdocumentation#Merkle%5FTrees">Bitcoin wiki</a> ):</p>

<ol>
<li>Jeśli lista ma długość 1, to skrót jej jedynego elementu jest korzeniem drzewa.</li>
<li>Pogrupuj elementy w pary (jeśli lista ma długość nieparzystą, ostatni element zduplikuj)</li>
<li>Dla każdej pary oblicz jej skrót</li>
<li>Wynikiem jest lista skrótów</li>
<li>Powtarzaj proces tak długo aż zostanie jeden skrót (korzeń drzewa)</li>
</ol>

<p>W praktycznych zastosowaniach jako funkcji skrótu należy użyć
funkcji bezpiecznej kryptograficznie (np SHA256, SHA3).
W tym zadaniu dla uproszczenia użyjemy 32-bitowej funkcji skrótu
(która zdecydowanie nie jest bezpieczna),
zaimplementowanej w dostarczonym module <code>Hashable32</code>.</p>

<p>Stwórz moduł <code>HashTree</code> realizujący drzewa skrótu,
spełniający poniższe warunki.</p>

<p>Moduł powinien dostarczać co najmniej operacji:</p>

<pre><code class="haskell">leaf :: Hashable a =&gt; a -&gt; Tree a
node :: Hashable a =&gt;  Tree a -&gt; Tree a -&gt; Tree a
twig :: Hashable a =&gt; Tree a -&gt; Tree a
buildTree :: Hashable a =&gt; [a] -&gt; Tree a
treeHash :: Tree a -&gt; Hash
drawTree :: Show a =&gt; Tree a -&gt; String
</code></pre>

<p>Funkcje <code>leaf</code>, <code>twig</code> i <code>node</code> budują węzły drzewa o odpowiednio 0,1 i 2 potomkach.</p>

<p><code>drawTree</code> ma być funkcją produkującą czytelną reprezentacje tekstową drzewa
tak, aby:</p>

<pre><code>&gt;&gt;&gt; putStr $ drawTree $ buildTree "fubar"
0x2e1cc0e4 -
  0xfbfe18ac -
    0x6600a107 -
      0x00000066 'f'
      0x00000075 'u'
    0x62009aa7 -
      0x00000062 'b'
      0x00000061 'a'
  0xd11bea20 +
    0x7200b3e8 +
      0x00000072 'r'
</code></pre>

<p>(<code>+</code> i <code>-</code> oznaczają odpowiednio wierzchołek o jednym i dwóch potomkach);
do wypisywania skrótów mozna użyć funkcji <code>showHash</code> z modułu <code>Hashable32</code>.</p>

<h2>B. Dowody</h2>

<p>Dowodem na przynależność elementu do drzewa o określonym korzeniu jest
ścieżka, której każdy element zawiera informację, który potomek zawiera
interesujący nas element oraz skrót  drugiego z potomków.</p>

<pre><code class="haskell">type MerklePath = [Either Hash Hash]
data MerkleProof a = MerkleProof a MerklePath
</code></pre>

<p>Zdefiniuj (w module HashTree) funkcje</p>

<pre><code class="haskell">buildProof :: Hashable a =&gt; a -&gt; Tree a -&gt; Maybe (MerkleProof a)
merklePaths :: Hashable a =&gt; a -&gt; Tree a -&gt; [MerklePath]
</code></pre>

<p>oraz instancję klasy <code>Show</code> dla <code>MerkleProof</code> i potrzebne funkcje tak, aby</p>

<pre><code>&gt;&gt;&gt; mapM_ print $ map showMerklePath  $ merklePaths 'i' $ buildTree "bitcoin"
"&lt;0x5214666a&lt;0x7400b6ff&gt;0x00000062"
"&gt;0x69f4387c&lt;0x6e00ad98&gt;0x0000006f"

&gt;&gt;&gt; buildProof 'i' $ buildTree "bitcoin"
Just (MerkleProof 'i' &lt;0x5214666a&lt;0x7400b6ff&gt;0x00000062)

&gt;&gt;&gt; buildProof 'e' $ buildTree "bitcoin"
Nothing
</code></pre>

<p>Zdefiniuj funkcję sprawdzającą dowód przynależności
do drzewa skrótu o danym korzeniu:</p>

<pre><code class="haskell">verifyProof :: Hashable a =&gt; Hash -&gt; MerkleProof a -&gt; Bool
</code></pre>

<pre><code>&gt;&gt;&gt; let t = buildTree "bitcoin"
&gt;&gt;&gt; let proof = buildProof 'i' t
&gt;&gt;&gt; verifyProof (treeHash t) &lt;$&gt; proof
Just True
&gt;&gt;&gt; verifyProof 0xbada55bb &lt;$&gt; proof
Just False
</code></pre>

<h2>C. Blockchain</h2>

<p>Stworzymy uproszczony łańcuch bloków (blockchain).
oparty o schemat "Proof of Work". Uzupełnij dostarczony moduł <code>Blockchain</code> zgodnie z poniższym opisem.</p>

<p>Blockchain to łańcuch bloków z mechanizmami zapewniania jego rzetelności.
W schemacie "Proof of Work" gwarancja rzetelności
opiera się na trudności obliczeniowej generowania kolejnych bloków
(zatem zmiana historii po upływie kilku bloków wymaga mocy obliczeniowej, której nikt nie posiada). Jako nagrodę za wykonywanie tej pracy obliczeniowej, ktokolwiek stworzy blok (co wymaga znalezienia wartości uzupełniającej blok tak, aby jego skrót spełniał określone warunki), otrzymuje w nagrodę pewną ilość monet (dla Bitcoina w tym momencie 12.5BTC, u nas 50 monet). Nagroda ta zapisywana jest w specjalnej transakcji wewnątrz bloku, zwanej `coinbase'.</p>

<p>Cytując wiki Bitcoin:</p>

<p><em>``New bitcoins are generated by the network through the process of "mining". In a process that is similar to a continuous raffle draw, mining nodes on the network are awarded bitcoins each time they find the solution to a certain mathematical problem (and thereby create a new block).''</em> --- https://en.bitcoin.it/wiki/Help:FAQ#How_are_new_bitcoins_created.3F</p>

<p><em>``Mining is the process of spending computation power to secure Bitcoin transactions against reversal and introducing new Bitcoins to the system.</em></p>

<p><em>Technically speaking, mining is the calculation of a hash of the a block header, which includes among other things a reference to the previous block, a hash of a set of transactions and a nonce. If the hash value is [correct], a new block is formed and the miner gets the newly generated Bitcoins. [Otherwise], a new nonce is tried, and a new hash is calculated.''</em></p>

<p>*The advantage of using such a mechanism consists of the fact, that it is very easy to check a result: Given the payload and a specific nonce, only a single call of the hashing function is needed to verify that the hash has the required properties. Since there is no known way to find these hashes other than brute force, this can be used as a "proof of work" that someone invested a lot of computing power to find the correct nonce for this payload.</p>

<p><em>This feature is then used in the Bitcoin network to allow the network to come to a consensus on the history of transactions. An attacker that wants to rewrite history will need to do the required proof of work before it will be accepted. And as long as honest miners have more computing power, they can always outpace an attacker.</em>
--- https://en.bitcoin.it/wiki/Help:FAQ#What_is_mining.3F</p>

<p>Nasz (bardzo uproszczony) blockchain opiera się na mieszance rozwiązań z Bitcoin i Ethereum. W naszym schemacie, transakcja to przekazanie środków
między adresami (kwestię podpisywania transakcji, acz bardzo istotną, tutaj ignorujemy).
Kwoty reprezentujemy jako liczby naturalne
w tysięcznych częściach "monety" (coin).</p>

<p>W praktyce zachętą do włączania transakcji do bloku są opłaty od transakcji, które otrzymuje twórca bloku, tutaj tę kwestię również pomijamy.</p>

<pre><code class="haskell">type Address = Hash
type Amount = Word32
coin :: Amount
coin = 1000

data Transaction = Tx
  { txFrom :: Address
  , txTo :: Address
  , txAmount :: Amount
  } deriving Show

tx1 = Tx
  { txFrom = hash "Alice"
  , txTo = hash "Bob"
  , txAmount = 1*coin
  }
</code></pre>

<p>Blok składa się z nagłówka i listy transakcji:</p>

<pre><code>data Block = Block { blockHdr :: BlockHeader, blockTxs :: [Transaction]}
data BlockHeader = BlockHeader
  {
    parent :: Hash
  , coinbase :: Transaction
  , txroot :: Hash -- root of the Merkle tree
  , nonce :: Hash
  } deriving Show
</code></pre>

<p>Nagłówek zawiera:</p>

<ul>
<li><code>parent</code> --- skrót poprzedniego bloku (0 dla pierwszego bloku)</li>
<li><code>txroot</code> --- skrót z korzenia drzewa skrótów transakcji</li>
<li><code>coinbase</code> --- specjalna transakcja zawierająca nagrodę dla twórcy bloku</li>
<li><code>nonce</code> --- dowód pracy, wartość taka, by skrót nagłówka spełniał zadany warunek</li>
</ul>

<p>Skrótem bloku jest skrót jego nagłówka
(skrót drzewa transakcji jest zawarty w nagłówku).
<!-- Stwórz instancje `Hashable` dla typów `Block` i `BlockHeader`. --></p>

<p>Blok jest poprawnym przedłużeniem łańcucha,
którego ostatnim ogniwem jest blok o skrócie <code>parent</code>,
jeśli jego skrót kończy się (binarnie) liczbą zer wyznaczoną przez parametr <code>difficulty</code>,
<code>txroot</code> jest skrótem korzenia drzewa skrótów
utworzonego dla listy transakcji bloku poprzedzonej <code>coinbase</code>,
zaś <code>coinbase</code> spełnia warunki podane poniżej.</p>

<pre><code class="haskell">difficulty = 5
blockReward = 50*coin
validNonce :: BlockHeader -&gt; Bool
validNonce b = (hash b) `mod` (2^difficulty) == 0
</code></pre>

<p><code>coinbase</code> - specjalna transakcja zawierająca nagrodę dla twórcy bloku</p>

<pre><code class="haskell">coinbaseTx miner = Tx  { txFrom = 0, txTo = miner, txAmount = blockReward }
</code></pre>

<p>Uzupełnij funkcję</p>

<pre><code class="haskell">mineBlock :: Miner -&gt; Hash -&gt; [Transaction] -&gt; Block
mineBlock miner parent txs = undefined
</code></pre>

<p>tak aby tworzyła blok stanowiący poprawne przedłużenie łańcucha zakończonego skrótem <code>parentHash</code> i zawierający transakcje <code>txs</code>.</p>

<p>Stwórz funkcje</p>

<pre><code class="haskell">validChain :: [Block] -&gt; Bool
verifyChain :: [Block] - &gt; Maybe Hash
verifyBlock :: Block -&gt; Hash -&gt; Maybe Hash
</code></pre>

<ul>
<li><code>verifyBlock block parentHash</code> daje w wyniku <code>Just h</code>,
o ile <code>block</code> ma skrót <code>h</code> i jest poprawnym następcą bloku
o skrócie <code>parentHash</code></li>
<li><code>verifyChain blocks</code> daje w wyniku <code>Just h</code>,
o ile na liście <code>blocks</code> każdy blok jest poprawnym przedłużeniem
nastepujących po nim (argument jest listą bloków od ostatniego do pierwszego),
i pierwszy blok ma skrót <code>h</code> (zachowanie dla listy pustej może być inne).</li>
<li><code>validChain blocks</code> daje <code>True</code> o ile <code>blocks</code> jest poprawnym łańcuchem
zgodnie z opisem powyżej.</li>
</ul>

<pre><code>&gt;&gt;&gt; verifyChain [block1, block2]
Nothing
&gt;&gt;&gt; VH &lt;$&gt; verifyChain [block2,block1,block0]
Just 0x0dbea380
</code></pre>

<p>NB transakcje przechowujemy w naturalnej kolejności, bloki w kolejności odwróconej, tj. najnowszy blok dołączamy na początku listy.</p>

<h2>D. Poświadczenia transakcji</h2>

<p>Zdefiniuj poświadczenia transakcji, ich generowanie i kontrolę:</p>

<pre><code class="haskell">data TransactionReceipt = TxReceipt
  {  txrBlock :: Hash, txrProof :: MerkleProof Transaction } deriving Show

validateReceipt :: TransactionReceipt -&gt; BlockHeader -&gt; Bool
validateReceipt r hdr = txrBlock r == hash hdr
                        &amp;&amp; verifyProof (txroot hdr) (txrProof r)

mineTransactions :: Miner -&gt; Hash -&gt; [Transaction]
                 -&gt; (Block, [TransactionReceipt])
</code></pre>

<p>tak, aby</p>

<pre><code>&gt;&gt;&gt; let charlie = hash "Charlie"
&gt;&gt;&gt; let (block, [receipt]) = mineTransactions charlie (hash block1) [tx1]
&gt;&gt;&gt; block
BlockHeader {
  parent = 797158976,
  coinbase = Tx {
    txFrom = 0,
    txTo = 1392748814,
    txAmount = 50000},
  txroot = 2327748117, nonce = 3}
Tx {txFrom = 2030195168, txTo = 2969638661, txAmount = 1000}
&lt;BLANKLINE&gt;

&gt;&gt;&gt; receipt
TxReceipt {
  txrBlock = 230597504,
  txrProof = MerkleProof (Tx {
    txFrom = 2030195168,
    txTo = 2969638661,
    txAmount = 1000})
  &gt;0xbcc3e45a}
&gt;&gt;&gt; validateReceipt receipt (blockHdr block)
True
</code></pre>

<p>(niektóre linie zostały złamane dla czytelności)</p>

<h2>E. Drukowanie</h2>

<p>Uzupełnij dostarczony moduł <code>PPrint</code> tak, aby:</p>

<pre><code>&gt;&gt;&gt; runShows $ pprBlock block2
hash: 0x0dbea380
parent: 0x2f83ae40
miner: 0x5303a90e
root: 0x8abe9e15
nonce: 3
Tx# 0xbcc3e45a from: 0000000000 to: 0x5303a90e amount: 50000
Tx# 0x085e2467 from: 0x790251e0 to: 0xb1011705 amount: 1000

&gt;&gt;&gt; runShows $ pprListWith pprBlock [block0, block1, block2]
hash: 0x70b432e0
parent: 0000000000
miner: 0x7203d9df
root: 0x5b10bd5d
nonce: 18
Tx# 0x5b10bd5d from: 0000000000 to: 0x7203d9df amount: 50000
hash: 0x2f83ae40
parent: 0x70b432e0
miner: 0x790251e0
root: 0x5ea7a6f0
nonce: 0
Tx# 0x5ea7a6f0 from: 0000000000 to: 0x790251e0 amount: 50000
hash: 0x0dbea380
parent: 0x2f83ae40
miner: 0x5303a90e
root: 0x8abe9e15
nonce: 3
Tx# 0xbcc3e45a from: 0000000000 to: 0x5303a90e amount: 50000
Tx# 0x085e2467 from: 0x790251e0 to: 0xb1011705 amount: 1000
</code></pre>

<p>(funkcja <code>pprBlock</code> jest zdefiniowana w dostarczonym szkielecie modułu <code>Blockchain</code>)</p>

<h2>Testy</h2>

<p>Komentarze rozpoczynające się od sekwencji <code>&gt;&gt;&gt;</code> są wykonywalnymi testami.
Można je sprawdzić używając <code>doctest</code>:</p>

<pre><code>$ doctest HashTree.hs
Examples: 8  Tried: 8  Errors: 0  Failures: 0: 0
$ doctest Blockchain.hs
Examples: 16  Tried: 16  Errors: 0  Failures: 0
</code></pre>