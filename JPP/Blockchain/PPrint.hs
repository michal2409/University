module PPrint where

writeln :: String -> IO ()
writeln = putStrLn

showsPair :: Show a => (String, a) -> ShowS
showsPair (k,v) = showString (k ++ ": " ++ (show v))

pprH, pprV :: [ShowS] -> ShowS
pprV xs = intercalateS (showString "\n") xs
pprH xs = intercalateS (showString " ")  xs

intercalateS :: ShowS -> [ShowS] -> ShowS
intercalateS sep list = foldr (.) (showString "") (map1 (.sep) list)
  where
    -- Map all except last element in list.
    map1 f []  = []
    map1 f [x] = [x]
    map1 f (x:xs) = f x : map1 f xs

pprListWith :: (a -> ShowS) -> [a] -> ShowS
pprListWith f xs = pprV (map f xs)

runShows :: ShowS -> IO ()
runShows = putStrLn . ($"")
