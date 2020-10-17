module HashTree where
import Hashable32

-- Part A

data Tree x = Leaf x Hash  
            | Twig (Tree x) Hash
            | Node (Tree x) Hash (Tree x)

leaf :: Hashable a => a -> Tree a
leaf x = Leaf x (hash x)

twig :: Hashable a => Tree a -> Tree a
twig t = Twig t (hash (h, h))
  where h = treeHash t

node :: Hashable a => Tree a -> Tree a -> Tree a
node l r = Node l (hash (treeHash l, treeHash r)) r

treeHash :: Tree a -> Hash
treeHash (Leaf _ h)   = h
treeHash (Twig _ h)   = h
treeHash (Node _ h _) = h

buildTree :: Hashable a => [a] -> Tree a
buildTree xs = auxBuild (map leaf xs)
  where
    auxBuild []  = error "Tried to build tree from an empty list."
    auxBuild [x] = x
    auxBuild xs  = buildList xs []

    buildList [] s         = auxBuild $ reverse s
    buildList [x] s        = auxBuild $ reverse ((twig x):s)
    buildList (x1:x2:xs) s = buildList xs ((node x1 x2):s)

drawTree :: Show a => Tree a -> String
drawTree t = auxDraw t 0
  where
    drawNode t lvl = (replicate (2*lvl) ' ') ++ (auxDrawNode t) 
      where 
        auxDrawNode (Leaf x h)   = showHash h ++ " " ++ show x ++ "\n"
        auxDrawNode (Twig _ h)   = showHash h ++ " +\n"
        auxDrawNode (Node _ h _) = showHash h ++ " -\n"

    auxDraw x@(Leaf _ _) lvl   = drawNode x lvl
    auxDraw x@(Twig t _) lvl   = drawNode x lvl ++ auxDraw t (lvl+1)
    auxDraw x@(Node l _ r) lvl = drawNode x lvl ++ auxDraw l (lvl+1) ++ auxDraw r (lvl+1)

{- | Pretty printing
>>> putStr $ drawTree $ buildTree "fubar"
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
-}

-- Part B

type MerklePath = [Either Hash Hash]
data MerkleProof a = MerkleProof a MerklePath

-- example: addFront 1 [[2,3,4], [5,6]] = [[1,2,3,4], [1,5,6]]
addFront x xs = [x:l | l <- xs]

merklePaths :: Hashable a => a -> Tree a -> [MerklePath]
merklePaths x (Leaf _ h)
  | hash x == h = [[]]
  | otherwise   = []

merklePaths x (Twig t _) = addFront (Left (treeHash t)) (merklePaths x t)

merklePaths x (Node l _ r) = addFront (Left  (treeHash r)) (merklePaths x l)
                             ++
                             addFront (Right (treeHash l)) (merklePaths x r)

-- Based on https://hackage.haskell.org/package/base-4.12.0.0/docs/Text-Show.html
instance (Show a) => Show (MerkleProof a) where
  showsPrec d (MerkleProof t mpath) = showParen (d > app_prec) $
            showString "MerkleProof " .
            showsPrec (app_prec+1) t .
            showString (" " ++ (showMerklePath mpath))
    where app_prec = 10

showMerklePath :: MerklePath -> String
showMerklePath xs = auxShow xs [[]]
  where
    auxShow [] s = concat $ reverse s
    auxShow ((Right x):xs) s = auxShow xs ((">"++(showHash x)):s)
    auxShow ((Left x):xs)  s = auxShow xs (("<"++(showHash x)):s)

buildProof :: Hashable a => a -> Tree a -> Maybe (MerkleProof a)
buildProof x t
    | null xs   = Nothing
    | otherwise = Just (MerkleProof x (head xs))
  where xs = merklePaths x t

verifyProof :: Hashable a => Hash -> MerkleProof a -> Bool
verifyProof h (MerkleProof a mpath) = (foldr (\x acc -> applyHash x acc) (hash a) mpath) == h
  where
    applyHash (Left x) acc  = hash (acc, x)
    applyHash (Right x) acc = hash (x, acc)

{- | Pretty printing
>>> mapM_ print $ map showMerklePath  $ merklePaths 'i' $ buildTree "bitcoin"
"<0x5214666a<0x7400b6ff>0x00000062"
">0x69f4387c<0x6e00ad98>0x0000006f"

>>> buildProof 'i' $ buildTree "bitcoin"
Just (MerkleProof 'i' <0x5214666a<0x7400b6ff>0x00000062)

>>> buildProof 'e' $ buildTree "bitcoin"
Nothing

>>> let t = buildTree "bitcoin"
>>> let proof = buildProof 'i' t
>>> verifyProof (treeHash t) <$> proof
Just True
>>> verifyProof 0xbada55bb <$> proof
Just False
-}
 