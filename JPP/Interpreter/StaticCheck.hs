{-# LANGUAGE LambdaCase #-}

module StaticCheck where

    import Environment
    import AbsGrammar

    import Data.Map (Map)
    import qualified Data.Map as Map

    import Control.Monad.Reader
    import Control.Monad.Except
    import Control.Monad.State


    runStaticCheck stmts = runExceptT $ runStateT
                                            (runReaderT (checkStmts stmts) emptyTEnv)
                                        emptyTStore

    checkStmts :: [(Stmt a)] -> (TEval a) ()
    checkStmts []               = return ()
    checkStmts ((Empty a)  :xs) = checkStmts xs
    checkStmts ((SExp a e) :xs) = exprType e >>= (\_  -> checkStmts xs)
    checkStmts ((Print a e):xs) = exprType e >>= (\_  -> checkStmts xs)

    checkStmts ((Break a):xs) = do
        assertWhileDepth a "Break"
        checkStmts xs

    checkStmts ((Continue a):xs) = do
        assertWhileDepth a "Continue"
        checkStmts xs

    checkStmts ((Ass a ident e):xs) =
        getType a ident >>= \t -> assertExprType a e t >>= \_ -> checkStmts xs

    checkStmts ((VRet a):xs) = do
        assertFuncDepth a
        setTBranch $ Just $ T_RETURN (TVoid a) 
        checkStmts xs

    checkStmts ((Ret a e):xs) = do
        assertFuncDepth a
        exprType e >>= (\t -> setTBranch $ Just $ T_RETURN t)
        checkStmts xs

    checkStmts ((BStmt a1 (Block a2 s)):xs) = do
        blockCheck s
        checkStmts xs

    checkStmts ((Cond a1 e (Block a2 s)):xs) = do
        assertExprType a1 e (TBool a1)
        incIfDepth
        blockCheck s
        resetTBranch
        decIfDepth
        checkStmts xs

    checkStmts ((CondElse a1 e (Block a2 s1) (Block a3 s2)):xs) = do
        assertExprType a1 e (TBool a1)
        incIfDepth
        blockCheck s1
        getTBranch >>= \case
            Just (T_RETURN t1) -> do
                resetTBranch
                blockCheck s2
                getTBranch >>= \case
                    Just (T_RETURN t2) -> assertTypes a1 t1 t2
                    _ -> do
                        blockCheck xs
                        getTBranch >>= \case
                            Just (T_RETURN t3) -> assertTypes a1 t1 t3
                            _ -> assertIfDepth a1 t1
            _ -> blockCheck s2
        decIfDepth
        checkStmts xs

    checkStmts ((While a1 e (Block a2 s)):xs) = do
        assertExprType a1 e (TBool a1)
        incWhileDepth
        blockCheck s
        resetTBranch
        decWhileDepth
        checkStmts xs

    checkStmts ((Decl a1 t (NoInit a2 ident)):xs) = do
        assertDeclType t
        localScope a1 ident t xs
        
    checkStmts ((Decl a1 t (Init a2 ident e)):xs) = do
        assertExprType a2 e t
        checkStmts ((Decl a1 t (NoInit a2 ident)):xs)

    checkStmts ((FnDef a1 rt ident@(Ident n) args (Block a2 s)):xs) = do
        incFuncDepth
        incBlockDepth
        fenv <- argsTEnv args >>= (\env -> addTEnv a1 n (TFun a1 rt ident args) env)
        local (const fenv) (checkStmts s)
        getTBranch >>= \case
            Just (T_RETURN t) -> do assertTypes a1 t rt
            _ -> unless (isVoid rt) $ throwError $ NoReturnInFunc a1
        resetTBranch
        decFuncDepth
        decBlockDepth
        localScope a1 ident (TFun a1 rt ident args) xs
        
    exprType :: (Expr a) -> (TEval a) (Type a)
    exprType (EVar a ident) = getType a ident
    exprType (ELitInt a _)  = return (TInt a)
    exprType (ELitFalse a)  = return (TBool a)
    exprType (ELitTrue a)   = return (TBool a)
    exprType (EString a _)  = return (TStr a)

    exprType (EApp a ident es) = do
        assertFunction a ident
        TFun a rt _ args <- getType a ident
        assertArgs a args es
        return rt

    exprType (EList a es) = do
        ts <- mapM exprType es
        checkHomog a ts
        return $ TList a (head ts)

    exprType (ELen a ident) = do
        assertList a ident
        return (TInt a)

    exprType (EListIdx a l e) = do
        assertList a l
        assertExprType a e (TInt a)
        getType a l >>= return

    exprType (LAppend a l e) = do
        assertList a l
        TList _ t1 <- getType a l
        t2 <- exprType e
        assertTypes a t1 t2
        return (TList a t1)

    exprType (Neg a e) = assertExprType a e (TInt a) >>= (\_ -> return (TInt a))
    exprType (Not a e) = assertExprType a e (TBool a) >>= (\_ -> return (TBool a))
    exprType (EMul a e1 op e2) = assertIII a e1 e2
    exprType (EAdd a e1 op e2) = assertIII a e1 e2
    exprType (EAnd a e1 e2)    = assertBBB a e1 e2
    exprType (EOr a e1 e2)     = assertBBB a e1 e2
    exprType (ERel a e1 op e2) = do
        case op of
            EQU _ -> do assertAtomicType a e1
                        assertAtomicType a e2
            NE _  -> do assertAtomicType a e1 
                        assertAtomicType a e2
            _     -> do assertExprType a e1 (TInt a)
                        assertExprType a e2 (TInt a)
        return (TBool a)

-- Utils functions

    assertTypes :: a -> (Type a) -> (Type a) -> (TEval a) ()
    assertTypes a t1 t2 = when (not $ typesEq t1 t2) $ throwError $ TypeError a t1 t2

    typesEq :: (Type a) -> (Type a) -> Bool
    typesEq (TInt _) (TInt _) = True
    typesEq (TStr _) (TStr _) = True
    typesEq (TBool _) (TBool _) = True
    typesEq (TVoid _) (TVoid _) = True
    typesEq (TRef _ _) (TRef _ _) = True
    typesEq (TList _ _) (TList _ _) = True
    typesEq (TFun _ _ _ _) (TFun _ _ _ _) = True
    typesEq _ _ = False

    checkHomog :: a -> [(Type a)] -> (TEval a) ()
    checkHomog a [] = return ()
    checkHomog a [x] = return ()
    checkHomog a (x:y:xs) = do
        when (not $ typesEq x y) $ throwError $ ListNotHomogeneous a
        checkHomog a (y:xs)

    assertExprType :: a -> (Expr a) -> (Type a) -> (TEval a) ()
    assertExprType a e t = exprType e >>= (\et -> assertTypes a t et)

    assertRetBranches :: a -> Maybe (Branch a) -> Maybe (Branch a) -> (TEval a) (Bool)
    assertRetBranches a b1 b2 = do
        case (b1, b2) of
            (Just (T_RETURN t1), Just (T_RETURN t2)) -> do 
                assertTypes a t1 t2
                return True
            (Nothing, Nothing) -> return True
            (_, _) -> return False

    assertAtomicType a e = do
        exprType e >>= \case
            TInt  _ -> return ()
            TBool _ -> return ()
            TStr  _ -> return ()
            t       -> throwError $ NoCmpType a t

    assertList a ident = do
        getType a ident >>= \case
            TList _ _ -> return ()
            t -> throwError $ ListExpected a t

    assertFunction a ident = do
        getType a ident >>= \case
            TFun _ _ _ _ -> return ()
            t -> throwError $ FunExpected a t

    assertIII :: a -> (Expr a) -> (Expr a) -> (TEval a) (Type a)
    assertIII a e1 e2 = do
        assertExprType a e1 (TInt a)
        assertExprType a e2 (TInt a)
        return (TInt a)

    assertBBB :: a -> (Expr a) -> (Expr a) -> (TEval a) (Type a)
    assertBBB a e1 e2 = do
        assertExprType a e1 (TBool a)
        assertExprType a e2 (TBool a)
        return (TBool a)

    assertWhileDepth :: a -> String -> (TEval a) ()
    assertWhileDepth a s = do
        d <- getWhileDepth
        when(d <= 0) $ throwError $ BrkContOutWhile a s

    assertIfDepth :: a -> (Type a) -> (TEval a) ()
    assertIfDepth a t = do
        d <- getIfDepth 
        when(d == 1 && not (isVoid t)) $ throwError $ IfElseBranch a
        
    assertFuncDepth :: a -> (TEval a) ()
    assertFuncDepth a = do
        d <- getFuncDepth
        when (d == 0) $ throwError $ ReturnOutsideFunc a

    assertDeclType :: (Type a) -> (TEval a) ()
    assertDeclType (TRef a t)     = throwError $ InvalidDeclType a "reference"
    assertDeclType (TVoid a)      = throwError $ InvalidDeclType a "void"
    assertDeclType (TFun a _ _ _) = throwError $ InvalidDeclType a "function"
    assertDeclType _              = return ()

    assertArgs :: a -> [(Arg a)] -> [(Expr a)] -> (TEval a) ()
    assertArgs a [] [] = return ()
    assertArgs a [] _ = throwError $ FuncApp a
    assertArgs a _ [] = throwError $ FuncApp a

    assertArgs a ((Arg a1 (TRef a2 t) ident):ts) (e:es) = 
        assertExprType a e t >>= (\_ -> assertArgs a ts es)

    assertArgs a ((Arg a1 t ident):ts) (e:es) =
        assertExprType a e t >>= (\_ -> assertArgs a ts es)

    argsTEnv :: [(Arg a)] -> (TEval a) (TEnv a)
    argsTEnv [] = ask
    argsTEnv ((Arg a (TRef _ t) (Ident n)):args) = argsTEnv ((Arg a t (Ident n)):args)
    argsTEnv ((Arg a t (Ident n)):args) = argsTEnv args >>= (\env -> addTEnv a n t env)

    localScope :: a -> Ident -> (Type a) -> [(Stmt a)] -> (TEval a) ()
    localScope a (Ident n) t xs = do
        env <- ask >>= (\env -> addTEnv a n t env)
        local (const env) (checkStmts xs)

    blockCheck :: [(Stmt a)] -> (TEval a) ()
    blockCheck s = do
        env <- ask
        incBlockDepth
        local (const env) (checkStmts s)
        decBlockDepth
    
    addTEnv :: a -> String -> (Type a) -> (TEnv a) -> (TEval a) (TEnv a)
    addTEnv a n t env = do 
        d1 <- getBlockDepth
        d2 <- getVarDepth a n env
        when (d1 == d2) $ throwError $ VarRedefinition a n
        return $ Map.insert n (t, d1) env
        
    getVarDepth :: a -> String -> (TEnv a) -> (TEval a) (Integer)
    getVarDepth a n env = do
        case Map.lookup n env of
            Nothing     -> return (-1)
            Just (_, d) -> return d

    isVoid :: Type a -> Bool
    isVoid t = case t of
        TVoid _ -> True
        _       -> False
