module Environment where

    import AbsGrammar

    import Control.Monad.Reader
    import Control.Monad.State
    import Control.Monad.Except

    import Data.Maybe
    import Data.Map (Map)
    import qualified Data.Map as Map


    type Name     = String
    type Loc      = Integer
    type Env      = Map Name Loc
    type TEnv   a = Map Name ((Type a), Integer)
    type FunDef a = ([Stmt a], Env, [Arg a])
    type Eval   a = ReaderT Env (StateT (Store a) (ExceptT (Err a) IO))
    type TEval  a = ReaderT (TEnv a) (StateT (TStore a) (ExceptT (Err a) IO))

    data Store a = Store { nextLoc  :: Loc
                         , branch   :: Maybe (Branch a)
                         , stateMap :: Map Loc (Value a)}

    data TStore a = TStore { branchT    :: Maybe (Branch a)
                           , whileDepth :: Integer
                           , ifDepth    :: Integer
                           , funcDepth  :: Integer
                           , blockDepth :: Integer }

    data Branch a = V_RETURN (Value a)
                  | T_RETURN (Type a)
                  | BREAK
                  | CONTINUE

    data Value a = VInt  Integer
                 | VBool Bool
                 | VStr  String
                 | VFun  (FunDef a)
                 | VList [(Value a)]
                 | VVoid

    instance Show (Value a) where
        show (VInt  x) = show x
        show (VBool x) = show x
        show (VStr  x) = x
        show (VList x) = show x
        show VVoid     = "Void"
        show (VFun _)  = "Function"

    data (Err a) = DivisionByZero        a
                 | UnboundVariable       a String
                 | SegmentationFault     a
                 | OutOfRangeIdx         a Integer
                 | TypeError             a (Type a) (Type a)
                 | ListNotHomogeneous    a
                 | FuncApp               a
                 | ReturnOutsideFunction a
                 | NoReturnInFunc        a
                 | InvalidDeclType       a String
                 | ListExpected          a (Type a)
                 | FunExpected           a (Type a)
                 | NoCmpType             a (Type a)
                 | IfElseBranch          a
                 | BrkContOutWhile       a String
                 | ReturnOutsideFunc     a
                 | VarRedefinition       a String

    instance Show a => Show (Err a) where
        show (TypeError a x y)         = show a ++ " Couldn't match type " ++ show x ++ " with " ++ show y
        show (ReturnOutsideFunction a) = show a ++ " Return outside function"
        show (NoReturnInFunc a)        = show a ++ " Function without return statement"
        show (OutOfRangeIdx a idx)     = show a ++ " Index " ++ show idx ++ " out of range"
        show (ListNotHomogeneous a)    = show a ++ " List not homogenous"
        show (InvalidDeclType a s)     = show a ++ " Tried to declare invalid type " ++ s
        show (FuncApp a)               = show a ++ " Function called with wrong argument numbers"
        show (DivisionByZero a)        = show a ++ " Division by zero"
        show (UnboundVariable a s)     = show a ++ " Variable " ++ s ++ " not declared"
        show (SegmentationFault a)     = show a ++ " Segmentation fault"
        show (ListExpected a t)        = show a ++ " Expected List type but got " ++ show t
        show (FunExpected a t)         = show a ++ " Cannot call type " ++ show t
        show (NoCmpType a t)           = show a ++ " Type " ++ show t ++ " is not comparable"
        show (IfElseBranch a)          = show a ++ " In if-else statements one branch is missing return"
        show (BrkContOutWhile a s)     = show a ++ " " ++ s ++ " outside while statement"
        show (ReturnOutsideFunc a)     = show a ++ " Return outside function"
        show (VarRedefinition a s)     = show a ++ " Variable " ++ s ++ " redefinition"

    emptyStore :: Store a
    emptyStore = Store 0 Nothing Map.empty

    emptyEnv :: Env
    emptyEnv = Map.empty

    emptyTStore :: TStore a
    emptyTStore = TStore Nothing 0 0 0 0

    emptyTEnv :: TEnv a
    emptyTEnv = Map.empty

    getLoc :: a -> Ident -> Eval a Loc
    getLoc a (Ident x) = do
        env <- ask
        case Map.lookup x env of
            Nothing  -> throwError $ UnboundVariable a x
            Just loc -> return loc

    setLoc :: Ident -> Loc -> (Eval a) Env
    setLoc (Ident x) loc = ask >>= (\env -> return $ Map.insert x loc env)

    setRetV :: Value a -> (Eval a) ()
    setRetV v = get >>= (\store -> put (store {branch = Just (V_RETURN v)}))

    setBranch :: Maybe (Branch a) -> (Eval a) ()
    setBranch b = get >>= (\store -> put (store {branch = b}))

    setTBranch :: Maybe (Branch a) -> (TEval a) ()
    setTBranch b = get >>= (\store -> put (store {branchT = b}))

    incWhileDepth :: (TEval a) ()
    incWhileDepth = get >>= (\store -> put (store {whileDepth = (whileDepth store)+1}))

    resetBranch :: (Eval a) ()
    resetBranch = get >>= (\store -> put (store {branch = Nothing}))

    resetTBranch :: (TEval a) ()
    resetTBranch = get >>= (\store -> put (store {branchT = Nothing}))

    getBranch :: (Eval a) (Maybe (Branch a))
    getBranch = get >>= (\store -> return $ branch store)

    getTBranch :: (TEval a) (Maybe (Branch a))
    getTBranch = get >>= (\store -> return $ branchT store)

    getVal :: a -> Ident -> (Eval a) (Value a)
    getVal a ident = getLoc a ident >>= readLoc a

    getType :: a -> Ident -> (TEval a) (Type a)
    getType a (Ident n) = do
        env <- ask
        case Map.lookup n env of
            Nothing     -> throwError $ UnboundVariable a n
            Just (t, _) -> return t

    setVar :: a -> Ident -> (Value a) -> (Eval a) ()
    setVar a ident v = get >>= (\store -> getLoc a ident >>= \loc ->
        put (store {stateMap = (Map.adjust (const v) loc (stateMap store))}))

    writeStore :: (Value a) -> (Eval a) Loc
    writeStore v = do
        store <- get
        let loc      = nextLoc store
        let b        = branch store
            next     = 1 + loc
            newState = Map.insert loc v (stateMap store)
        put (Store next b newState)
        return loc

    readLoc :: a -> Loc -> (Eval a) (Value a)
    readLoc a loc = do
        store <- get
        case Map.lookup loc (stateMap store) of
            Nothing -> throwError $ SegmentationFault a
            Just v  -> return v

    readIdent :: a -> Ident -> (Eval a) (Value a)
    readIdent a ident = getLoc a ident >>= readLoc a

    declVar :: a -> Ident -> (Value a) -> (Eval a) Env
    declVar a ident v = writeStore v >>= setLoc ident

    msg :: (Value a) -> (Eval a) ()
    msg = liftIO . putStrLn . show

    decWhileDepth :: (TEval a) ()
    decWhileDepth = get >>= (\store -> put (store {whileDepth = (whileDepth store)-1}))
    
    getWhileDepth :: (TEval a) (Integer)
    getWhileDepth = get >>= (\store -> return $ whileDepth store)

    incIfDepth :: (TEval a) ()
    incIfDepth = get >>= (\store -> put (store {ifDepth = (ifDepth store)+1}))

    decIfDepth :: (TEval a) ()
    decIfDepth = get >>= (\store -> put (store {ifDepth = (ifDepth store)-1}))

    getIfDepth :: (TEval a) (Integer)
    getIfDepth = get >>= (\store -> return $ ifDepth store)

    incFuncDepth :: (TEval a) ()
    incFuncDepth = get >>= (\store -> put (store {funcDepth = (funcDepth store)+1}))

    decFuncDepth :: (TEval a) ()
    decFuncDepth = get >>= (\store -> put (store {funcDepth = (funcDepth store)-1}))

    getFuncDepth :: (TEval a) (Integer)
    getFuncDepth = get >>= (\store -> return $ funcDepth store)

    incBlockDepth :: (TEval a) ()
    incBlockDepth = get >>= (\store -> put (store {blockDepth = (blockDepth store)+1}))

    decBlockDepth :: (TEval a) ()
    decBlockDepth = get >>= (\store -> put (store {blockDepth = (blockDepth store)-1}))

    getBlockDepth :: (TEval a) (Integer)
    getBlockDepth = get >>= (\store -> return $ blockDepth store)
