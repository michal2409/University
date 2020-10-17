{-# LANGUAGE LambdaCase #-}

module Interpreter where

    import Data.Map (Map)
    import qualified Data.Map as Map

    import Control.Monad.Reader
    import Control.Monad.State
    import Control.Monad.Except

    import AbsGrammar
    import Environment

    import Data.Maybe


    runInterpreter stmts = runExceptT $ runStateT
                                            (runReaderT (evalStmts stmts) emptyEnv)
                                        emptyStore

    evalStmts :: [(Stmt a)] -> (Eval a) ()
    evalStmts ((FnDef a t ident args (Block b stmts)):xs) = do
        env <- declVar a ident VVoid  -- To allow recursion declare function ident
        local (const env) (setVar a ident (VFun (stmts, env, args)))  -- Update store with func def
        local (const env) (evalStmts xs)  -- Computing rest of statemts in changed env

    evalStmts ((Decl a t (Init b ident e)):xs) = do
        env <- evalExpr e >>= declVar a ident
        local (const env) (evalStmts xs)
        
    evalStmts ((Decl a t (NoInit b ident)):xs) =
        evalStmts ((Decl a t (Init b ident (defaultVal t))):xs)

    evalStmts []                   = return ()
    evalStmts ((Empty a):xs)       = evalStmts xs
    evalStmts ((Ret a e):xs)       = evalExpr e >>= setRetV
    evalStmts ((VRet a):xs)        = setBranch $ Just $ V_RETURN VVoid
    evalStmts ((Break a):xs)       = setBranch $ Just BREAK
    evalStmts ((Continue a):xs)    = setBranch $ Just CONTINUE
    evalStmts ((SExp a e):xs)      = evalExpr e >>= (\_ -> evalStmts xs)
    evalStmts ((Print a e):xs)     = evalExpr e >>= msg >>= (\_ -> evalStmts xs)
    evalStmts ((Ass a ident e):xs) = evalExpr e >>= setVar a ident >>= (\_ -> evalStmts xs)

    evalStmts ((BStmt a (Block b stmts)):xs) = do
        env <- ask
        local (const env) (evalStmts stmts)
        handleBranch xs
        
    evalStmts ((CondElse a e (Block b s1) (Block c s2)):xs) = do
        env <- ask
        evalExpr e >>= \case
            VBool True  -> local (const env) (evalStmts s1)
            VBool False -> local (const env) (evalStmts s2)
        handleBranch xs

    evalStmts ((Cond a e (Block b s)):xs) = do
        VBool r <- evalExpr e
        when (r) $ ask >>= (\env -> (local (const env) (evalStmts s)))
        handleBranch xs

    evalStmts ((While a e (Block b s)):xs) = do
        evalExpr e >>= \case
            VBool True -> do
                ask >>= (\env -> local (const env) (evalStmts s))
                getBranch >>= \case
                    Nothing           -> do evalStmts ((While a e (Block b s)):xs)
                    Just CONTINUE     -> do resetBranch; evalStmts ((While a e (Block b s)):xs)
                    Just BREAK        -> do resetBranch; evalStmts xs        
                    Just (V_RETURN _) -> do evalStmts xs
            VBool False -> evalStmts xs

    -- Evaluation of expressions
    evalExpr :: (Expr a) -> (Eval a) (Value a)
    evalExpr (ELitInt a x)     = return (VInt x)
    evalExpr (ELitFalse a)     = return (VBool False)
    evalExpr (ELitTrue  a)     = return (VBool True)
    evalExpr (EString a x)     = return (VStr x)
    evalExpr (EVar a ident)    = readIdent a ident
    evalExpr (EAdd a e1 op e2) = liftM2 (evalAdd op) (evalExpr e1) (evalExpr e2)
    evalExpr (ERel a e1 op e2) = liftM2 (evalRel op) (evalExpr e1) (evalExpr e2)
    evalExpr (EAnd a e1 e2)    = liftM2 evalAnd (evalExpr e1) (evalExpr e2)
    evalExpr (EOr a e1 e2)     = liftM2 evalOr (evalExpr e1) (evalExpr e2)
    evalExpr (Not a e)         = liftM evalNot (evalExpr e)
    evalExpr (Neg a e)         = liftM evalNeq (evalExpr e)

    evalExpr (EMul a e1 op e2) = do
        VInt r1 <- evalExpr e1
        VInt r2 <- evalExpr e2
        when ((isDivModOp op) && r2 == 0) $ throwError $ DivisionByZero a
        return $ VInt (evalMul op r1 r2)

    evalExpr (EApp a ident es) = do
        VFun (s, fenv, args) <- evalExpr (EVar a ident)
        fenv <- ask >>= (\env -> local (const fenv) (evalArgs args es env))
        local (const fenv) (evalStmts s) >>= (\_ -> getReturnVal)

    evalExpr (EList a exprs) = mapM evalExpr exprs >>= (\es -> return $ VList es)
    
    evalExpr (ELen a ident) = do
        VList l <- getVal a ident
        return $ VInt $ fromIntegral $ length l

    evalExpr (LAppend a ident@(Ident name) e) = 
        evalExpr e >>=
            (\v -> getVal a ident >>=
                (\(VList l) -> return $ VList (l ++ [v])))

    evalExpr (EListIdx a ident e) = do
        VInt idx <- evalExpr e
        VList l  <- getVal a ident
        when (idx < 0 || idx >= fromIntegral (length l))
            $ throwError $ OutOfRangeIdx a idx
        return $ l !! (fromIntegral idx)

    -- Evaluation of operators
    evalAdd :: (AddOp a) -> (Value a) -> (Value a) -> (Value a)
    evalAdd (Plus  _) (VInt a) (VInt b) = VInt $ a + b
    evalAdd (Minus _) (VInt a) (VInt b) = VInt $ a - b
    
    evalMul :: Integral b => (MulOp a) -> b -> b -> b
    evalMul (Times _) = (*)
    evalMul (Div _)   = (div)
    evalMul (Mod _)   = (mod)

    evalRel :: (RelOp a) -> (Value a) -> (Value a) -> (Value a)
    evalRel (LTH _) (VInt a)  (VInt b)  = VBool $ a < b
    evalRel (LE  _) (VInt a)  (VInt b)  = VBool $ a <= b
    evalRel (GTH _) (VInt a)  (VInt b)  = VBool $ a > b
    evalRel (GE  _) (VInt a)  (VInt b)  = VBool $ a >= b
    evalRel (EQU _) (VInt a)  (VInt b)  = VBool $ a == b
    evalRel (EQU _) (VBool a) (VBool b) = VBool $ a == b
    evalRel (EQU _) (VStr a)  (VStr b)  = VBool $ a == b
    evalRel (NE  _) (VInt a)  (VInt b)  = VBool $ a /= b
    evalRel (NE  _) (VBool a) (VBool b) = VBool $ a /= b
    evalRel (NE  _) (VStr a)  (VStr b)  = VBool $ a /= b

    evalAnd :: (Value a) -> (Value a) -> (Value a)
    evalAnd (VBool a) (VBool b) = VBool $ a && b

    evalOr :: (Value a) -> (Value a) -> (Value a)
    evalOr  (VBool a) (VBool b) = VBool $ a || b

    evalNot :: (Value a) -> (Value a)
    evalNot (VBool a) = VBool $ not a

    evalNeq :: (Value a) -> (Value a)
    evalNeq (VInt a) = VInt $ negate a

    -- Utils function
    getReturnVal :: (Eval a) (Value a)
    getReturnVal = do
        getBranch >>= \case
            Just (V_RETURN v) -> do resetBranch; return v
            Nothing -> do resetBranch; return VVoid

    handleBranch :: [(Stmt a)] -> (Eval a) ()
    handleBranch xs = do
        getBranch >>= \case
            Nothing -> evalStmts xs
            _       -> return ()

    isDivModOp :: (MulOp a) -> Bool
    isDivModOp op =
        case op of
            Div _ -> True
            Mod _ -> True
            _     -> False

    -- For unintialized expressions default value is set
    defaultVal :: (Type a) -> (Expr a)
    defaultVal (TInt  a)   = ELitInt   a 0
    defaultVal (TBool a)   = ELitFalse a
    defaultVal (TStr  a)   = EString   a ""
    defaultVal (TList a _) = EList     a []


    evalArgs :: [(Arg a)] -> [(Expr a)] -> Env -> (Eval a) Env
    evalArgs [] [] env = ask

    evalArgs (Arg a1 (TRef _ t) (Ident name):args) ((EVar _ ident):es) env = 
        evalArgs args es env >>=
            (\env' -> local (const env) (getLoc a1 ident) >>=
                (\loc -> return (Map.insert name loc env')))
        
    evalArgs (Arg a t ident:args) (e:es) env =
        evalArgs args es env >>=
            (\env' -> local (const env) (evalExpr e) >>=
                (\v -> local (const  env') (declVar a ident v)))
