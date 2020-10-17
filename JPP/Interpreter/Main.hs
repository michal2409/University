{-# LANGUAGE LambdaCase #-}

import System.IO
import System.Environment
import LexGrammar
import ParGrammar
import AbsGrammar
import ErrM
import Interpreter
import StaticCheck
import Data.Maybe

newtype ErrPos = ErrPos (Int, Int)

instance Show ErrPos where
    show (ErrPos (x,y)) = "[" ++ show x ++ ":" ++ show y ++ "]"

convert :: Maybe (Int, Int) -> ErrPos
convert (Just (a, b)) = ErrPos (a, b)

run :: String -> IO ()
run text = case pProgram $ myLexer text of
    Bad err -> showErr err
    Ok p -> do
        let (Program _ s) = fmap convert p
        runStaticCheck s >>= \case
            Left err -> showErr err
            Right _ -> do
                runInterpreter s >>= \case
                    Left err -> showErr err
                    Right _ -> return ()
    where
        showErr err = hPutStrLn stderr $ show err

main :: IO ()
main = do
    args <- getArgs
    case args of
        [file] -> readFile file >>= run
        _ -> hPutStrLn stderr "usage: interpreter file"
