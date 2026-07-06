import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, MultiLabelBinarizer
import category_encoders as ce
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


#--------------------- Constant --------------------------
#defind multilable col (split by ';')
multiLable_cols=['LanguageHaveWorkedWith','LanguageWantToWorkWith','LanguageAdmired',
            'DatabaseHaveWorkedWith','DatabaseWantToWorkWith','DatabaseAdmired',
            'PlatformHaveWorkedWith','PlatformWantToWorkWith','PlatformAdmired',
            'WebframeHaveWorkedWith','WebframeWantToWorkWith','WebframeAdmired',
            'DevEnvsHaveWorkedWith','DevEnvsWantToWorkWith', 'DevEnvsAdmired',
            'SOTagsHaveWorkedWith','SOTagsWantToWorkWith','SOTagsAdmired',
            'OpSysPersonal use','OpSysProfessional use',
            'OfficeStackAsyncHaveWorkedWith','OfficeStackAsyncWantToWorkWith','OfficeStackAsyncAdmired',
            'CommPlatformHaveWorkedWith','CommPlatformWantToWorkWith','CommPlatformAdmired',
            'AIModelsHaveWorkedWith','AIModelsWantToWorkWith','AIModelsAdmired',
            'SO_Dev_Content',
            'AIToolCurrently','AIToolDon\'t plan to use AI for this task','AIToolPlan to partially use AI','AIToolPlan to mostly use AI',
            'AIToolCurrently mostly AI','AIFrustration','AIAgent_Uses',
            'AIAgentImpactSomewhat agree','AIAgentChallengesNeutral','AIAgentChallengesStrongly agree',
            'AIAgentChallengesSomewhat disagree','AIAgentChallengesSomewhat agree','AIHuman',
            'EmploymentAddl','LearnCode','AILearnHow',
            'AIToolCurrently partially AI']

grouped_mode_cols = [
    ('SOFriction', 'AIModelsChoice'), ('AIAgentChange', 'AIModelsChoice'),
    ('PlatformChoice', 'DatabaseChoice'), ('SOAccount', 'SOPartFreq'),
    ('AISelect', 'AIModelsChoice'), ('ICorPM', 'DevType'),
    ('AISent', 'AIModelsChoice'), ('AIAcc', 'AIComplex'),
    ('TechEndorseIntro', 'Employment'), ('OrgSize', 'Employment'),
    ('AIAgents', 'AIModelsChoice'), ('SOPartFreq', 'SOAccount'),
    ('AIComplex', 'AIModelsChoice'), ('Employment', 'DevType'),
    ('LearnCodeAI', 'AIModelsChoice'), ('DatabaseChoice', 'DevType'),
    ('DevEnvsChoice', 'AISelect'), ('AIModelsChoice', 'AISelect'),
    ('WebframeChoice', 'DevType'), ('DevType', 'ICorPM')
]

continuous_cols = ['salary_log', 'WorkExp', 'YearsCode','ToolCountWork', 'ToolCountPersonal']



# ---------------- Functions ------------------------
def classify_columns(df):
    global multiLable_cols
    df = df.drop(columns=['ResponseId'])
    # deleted col
    high_missing_col=df.columns[df.isnull().mean()*100>=70].sort_values(ascending=False).tolist()
    # dealing with it in the null processing
    mid_missing_col=df.columns[df.isnull().mean()*100>5].sort_values(ascending=False).tolist()
    #deleted row
    low_missing_col=df.columns[(df.isnull().mean()*100>0)&(df.isnull().mean()*100<=5)].sort_values(ascending=False)
    low_missing_col = low_missing_col[low_missing_col != 'ConvertedCompYearly'].tolist()   
    #defind multilable col (split by ';')
    #category
    categorical_cols=df.select_dtypes(include='object').columns.tolist()
    #numaric
    numeric_cols=df.select_dtypes(include='number').columns.tolist()

    single_label_cols = [
        c for c in categorical_cols 
        if c not in multiLable_cols 
        and df[c].nunique() <= 500
    ]

    return categorical_cols, numeric_cols, high_missing_col, mid_missing_col, low_missing_col, single_label_cols


def primary_cleaning(df, categorical_cols, numeric_cols, high_missing_col, low_missing_col):
    Cleaned_df = df
    target_col = 'JobSat'
    safe_high_missing = [col for col in high_missing_col if col != target_col]
    Cleaned_df = Cleaned_df.drop(columns=safe_high_missing, errors='ignore')
    
    existing_low_missing = [c for c in low_missing_col if c in Cleaned_df.columns]
    Cleaned_df = Cleaned_df.dropna(subset=existing_low_missing)
    Cleaned_df = Cleaned_df.dropna(subset=['JobSat'])
    Cleaned_df = Cleaned_df.drop(columns=['CompTotal', 'Currency'])

    for col in list(set(categorical_cols) & set(Cleaned_df)):
        Cleaned_df[col] = Cleaned_df[col].str.lower().str.strip()

    for col in list(set(categorical_cols) & set(Cleaned_df)):
        if col in multiLable_cols:
            continue
        counts = Cleaned_df[col].value_counts(normalize=True)
        rare_values = counts[counts < 0.002].index
        Cleaned_df[col] = Cleaned_df[col].replace(rare_values, 'other')
        Cleaned_df[col] = Cleaned_df[col].replace(r'(?i).*other.*', 'other', regex=True)

    # Clippings
    Cleaned_df.loc[Cleaned_df['YearsCode'] > 60, 'YearsCode'] = 60
    Cleaned_df.loc[Cleaned_df['WorkExp'] > 60, 'WorkExp'] = 60
    
    # Salary log
    Cleaned_df['salary_log'] = np.log1p(Cleaned_df['ConvertedCompYearly'])
    if 'ConvertedCompYearly' not in high_missing_col :
        if 'ConvertedCompYearly' not in low_missing_col:
            Cleaned_df = Cleaned_df.drop(columns=['ConvertedCompYearly'])

    Cleaned_df.loc[Cleaned_df['ToolCountWork'] > 50, 'ToolCountWork'] = 50
    Cleaned_df.loc[Cleaned_df['ToolCountPersonal'] > 50, 'ToolCountPersonal'] = 50
    numeric_cols = Cleaned_df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols=df.select_dtypes(include='object').columns.tolist()

    
    return Cleaned_df, categorical_cols, numeric_cols




def split_df(df,mid_missing_col, categorical_cols, numeric_cols, high_missing_col, low_missing_col):
    global multiLable_cols
    # Primary cleaning
    cleaned_df, categorical_cols, numeric_cols = primary_cleaning(df, categorical_cols, numeric_cols, high_missing_col, low_missing_col)
    y = cleaned_df['JobSat']
    X = cleaned_df.drop(columns='JobSat')

    #update predefine lists
    mid_missing_col = list(set(mid_missing_col) & set(X))
    numeric_cols = list(set(numeric_cols) & set(X.columns))
    multiLable_cols = list(set(multiLable_cols) & set(X.columns))
    categorical_cols = [col for col in categorical_cols if col in X.columns and col not in multiLable_cols]


    # 1. Split data into Train_Full and Test (Test = 15% of total data)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)

    # 2. Split Train_Full into Train and Validation (Val = 15% of total data, which is ~17.6% of train_full)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.176, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, mid_missing_col, numeric_cols, multiLable_cols, categorical_cols



# ____________ Helper Functions _____________________________

def cramers_v(col1, col2):
    mask = col1.notna() & col2.notna()
    if mask.sum() < 10:
        return 0
        
    ct = pd.crosstab(col1[mask], col2[mask])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0
        
    chi2 = chi2_contingency(ct)[0]
    n    = ct.sum().sum()
    r, k = ct.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

def fill_with_mode(train, val, test, col):
    fill_value = train[col].mode()[0]
    for df in [train, val, test]:
        df[col] = df[col].fillna(fill_value)
    return train, val, test

def fill_with_grouped_mode(train, val, test, target_col, group_col):

    grouped_mode = (
        train.dropna(subset=[target_col])
            .groupby(group_col)[target_col]
            .agg(lambda x: x.mode()[0]))
    
    global_mode = train[target_col].mode()[0]
    for df in [train, val, test]:
        missing_mask = df[target_col].isna()
        if missing_mask.sum() == 0:
            continue
        df.loc[missing_mask, target_col] = (
            df.loc[missing_mask, group_col]
            .map(grouped_mode)
            .fillna(global_mode))
        
    return train, val, test

def count_selections(train, val, test, col, new_col_name):
    for df in [train, val, test]:
        df[new_col_name] = df[col].astype(str).str.count(';') + 1
        df.loc[df[col].isna(), new_col_name] = 0       
    return train, val, test
    

def RO_Questions_Handling(X_train, X_val, X_test, new_col_name, question_options):
    # Train
    X_train[new_col_name] = X_train[question_options].isnull().all(axis=1).astype(int)
    X_train[question_options] = X_train[question_options].fillna(0)
    
    # Validation
    X_val[new_col_name] = X_val[question_options].isnull().all(axis=1).astype(int)
    X_val[question_options] = X_val[question_options].fillna(0)

    # Test
    X_test[new_col_name] = X_test[question_options].isnull().all(axis=1).astype(int)
    X_test[question_options] = X_test[question_options].fillna(0)

    return X_train, X_val, X_test

def Quick_Imputer(X_train, X_val, X_test, cols, num_or_cat): # 1 -> num , 0 -> cat
    if num_or_cat == 0:
        for col in cols:
            if col in X_train.columns:
                fill_value = X_train[col].mode()[0]
                X_train[col] = X_train[col].fillna(fill_value)
                X_val[col]   = X_val[col].fillna(fill_value)
                X_test[col]  = X_test[col].fillna(fill_value)
    else:
        for col in cols:
            if col in X_train.columns:
                fill_value = X_train[col].median()
                X_train[col] = X_train[col].fillna(fill_value)
                X_val[col]   = X_val[col].fillna(fill_value)
                X_test[col]  = X_test[col].fillna(fill_value)
    return X_train, X_val, X_test

def Full_Cleaning_Pipeline(X_train, X_val, X_test, multiLable_cols,single_label_cols):
    global grouped_mode_cols
    # ── 1. YearsCode & WorkExp ────────────────────────────────────────────────
    years_code_median = X_train['YearsCode'].median()
    for df_temp in [X_train, X_val, X_test]:
        df_temp['YearsCode'] = df_temp['YearsCode'].fillna(years_code_median)

        student_logic = df_temp['Employment'].eq('student') & df_temp['Age'].eq('18-24 years old')
        df_temp.loc[df_temp['WorkExp'].isna() & student_logic, 'WorkExp'] = 0

    impute_cols_workexp = ['WorkExp', 'YearsCode']
    it_imput = IterativeImputer(random_state=42)
    X_train[impute_cols_workexp] = it_imput.fit_transform(X_train[impute_cols_workexp])
    X_val[impute_cols_workexp]   = it_imput.transform(X_val[impute_cols_workexp])
    X_test[impute_cols_workexp]  = it_imput.transform(X_test[impute_cols_workexp])

    for df_temp in [X_train, X_val, X_test]:
        df_temp.loc[df_temp['WorkExp'] > df_temp['YearsCode'], 'WorkExp'] = df_temp['YearsCode']
        df_temp['WorkExp'] = df_temp['WorkExp'].clip(lower=0).round(0)

    # ── 2. Survey Ranking Questions ──────────────────────────────
    so_actions_cols   = [col for col in X_train.columns if col.startswith('SO_Actions')]
    tech_endorse_cols = [col for col in X_train.columns if col.startswith('TechEndorse_')]
    tech_oppose_cols  = [col for col in X_train.columns if col.startswith('TechOppose_')]

    X_train, X_val, X_test = RO_Questions_Handling(X_train, X_val, X_test, 'SO_Actions_Skipped', so_actions_cols)
    X_train, X_val, X_test = RO_Questions_Handling(X_train, X_val, X_test, 'TechEndorse_Skipped', tech_endorse_cols)
    X_train, X_val, X_test = RO_Questions_Handling(X_train, X_val, X_test, 'TechOppose_Skipped', tech_oppose_cols)

    for df_temp in [X_train, X_val, X_test]:
        df_temp['Both_tech_skipped'] = ((df_temp['TechOppose_Skipped'] == 1) & (df_temp['TechEndorse_Skipped'] == 1)).astype(int)
        df_temp.drop(columns='TechEndorse_Skipped', inplace=True, errors='ignore')


    # ── 3. Salary ─────────────────────────────────────────────────────────────
    country_salary_map = X_train.groupby('Country')['salary_log'].median()
    general_salary_median = X_train['salary_log'].median()

    for df_temp in [X_train, X_val, X_test]:
        df_temp['salary_log'] = df_temp['salary_log'].fillna(df_temp['Country'].map(country_salary_map))
        df_temp['salary_log'] = df_temp['salary_log'].fillna(general_salary_median)

    # ── 4. Tool Count Columns ─────────────────────────────────────────────────
    tool_cols = ['ToolCountWork', 'ToolCountPersonal']
    for col in tool_cols:
        train_median = X_train[col].median()
        for df_temp in [X_train, X_val, X_test]:
            df_temp[col] = df_temp[col].fillna(train_median)

    # ── 5. Categorical Features ──────────────────────────────────────────────
    for col in multiLable_cols:
        if col in X_train.columns:
            X_train, X_val, X_test = count_selections(X_train, X_val, X_test, col, col + '_count')


    grouped_done = {col for col, _ in grouped_mode_cols}
    for target_col, group_col in grouped_mode_cols:
        if target_col in X_train.columns and group_col in X_train.columns:
            X_train, X_val, X_test = fill_with_grouped_mode(X_train, X_val, X_test, target_col, group_col)

    for col in single_label_cols:
        if col in grouped_done or col not in X_train.columns:
            continue
        X_train, X_val, X_test = fill_with_mode(X_train, X_val, X_test, col)

    # ── 6. Final Touch (Quick Imputer) ────────────────────────────────────────
    current_num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    current_cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    X_train, X_val, X_test = Quick_Imputer(X_train, X_val, X_test, current_num_cols, 1)
    X_train, X_val, X_test = Quick_Imputer(X_train, X_val, X_test, current_cat_cols, 0)

    return X_train, X_val, X_test, current_cat_cols, current_num_cols , it_imput

def handle_outliers_iqr(X_train, X_val, X_test, cols):
    for col in cols:
        if col in X_train.columns:
            Q1 = X_train[col].quantile(0.25)
            Q3 = X_train[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            X_train[col] = X_train[col].clip(lower=lower_bound, upper=upper_bound)
            X_val[col]   = X_val[col].clip(lower=lower_bound, upper=upper_bound)
            X_test[col]  = X_test[col].clip(lower=lower_bound, upper=upper_bound)

    return X_train, X_val, X_test

def calc_rate_2col(row, col1, col2):
    worked=set([e.strip() for e in str(row[col1]).split(';') if e.strip()!='nan'])
    want=set([e.strip() for e in str(row[col2]).split(';') if e.strip()!='nan'])
    if len(worked)==0:
        return 0
    return len(worked.intersection(want))/len(worked)

def feature_engineering(df_cleaned):
    df_feature_eng=df_cleaned.copy()
    df_feature_eng['salary_par_workExp'] = df_feature_eng['salary_log'] / (df_feature_eng['WorkExp']+1)
    df_feature_eng['over_Tools'] =  df_feature_eng['ToolCountWork'] / (df_feature_eng['WorkExp']+1)
    tech_cat=['Language','Database','Platform','Webframe','DevEnvs','SOTags','OfficeStackAsync','CommPlatform','AIModels']
    for cat in tech_cat:
        have_col=cat+'HaveWorkedWith'
        want_col=cat+'WantToWorkWith'
        admired_col=cat+'Admired'
        if have_col in df_feature_eng.columns and want_col in df_feature_eng.columns:
                df_feature_eng[cat+'_want_rate']=df_feature_eng.apply(lambda r: calc_rate_2col(r,have_col,want_col),axis=1)
        if have_col in df_feature_eng.columns and admired_col in df_feature_eng.columns:
                df_feature_eng[cat+'_admir_rate']=df_feature_eng.apply(lambda r: calc_rate_2col(r,have_col,admired_col),axis=1)
        col_to_drop=[c for c in [have_col,want_col,admired_col]
                    if c in df_feature_eng.columns]
        df_feature_eng = df_feature_eng.drop(columns=col_to_drop,errors='ignore')

    w_col=[col for col in df_feature_eng.columns if '_want_rate' in col ]
    if len(w_col)>0:
        df_feature_eng['avg_want_rate']=df_feature_eng[w_col].mean(axis=1)

    ad_col=[col for col in df_feature_eng.columns if '_admir_rate' in col ]
    if len(ad_col)>0:
        df_feature_eng['avg_admir_rate']=df_feature_eng[ad_col].mean(axis=1)

    return df_feature_eng


def apply_ordinal_encoding(X_train, X_val, X_test):
    age_order = ['other', 'under 18 years old', '18-24 years old', '25-34 years old',
                '35-44 years old', '45-54 years old', '55-64 years old', '65 years or older']

    edlevel_order = ['other', 'something else', 'primary/elementary school',
                    'secondary school (e.g. american high school, german gymnasium, etc.)',
                    'some college/university study without earning a degree',
                    'associate degree (a.a., a.s., etc.)',
                    'bachelor’s degree (b.a., b.s., b.eng., etc.)',
                    'master’s degree (m.a., m.s., m.eng., mba, etc.)',
                    'professional degree (jd, md, ph.d, ed.d, etc.)']

    ordinal_cols = ['Age', 'EdLevel']
    enc = OrdinalEncoder(categories=[age_order, edlevel_order],
                        handle_unknown='use_encoded_value', unknown_value=-1)

    X_train[ordinal_cols] = enc.fit_transform(X_train[ordinal_cols])
    X_val[ordinal_cols]   = enc.transform(X_val[ordinal_cols])
    X_test[ordinal_cols]  = enc.transform(X_test[ordinal_cols])

    return X_train, X_val, X_test, enc

def apply_onehot_encoding(X_train, X_val, X_test):
    low_card_cols = [
        'MainBranch', 'RemoteWork', 'Employment', 'OrgSize', 'LearnCodeChoose',
        'LearnCodeAI', 'ICorPM', 'PurchaseInfluence', 'TechEndorseIntro',
        'AIThreat', 'NewRole', 'LanguageChoice', 'DatabaseChoice', 'PlatformChoice',
        'WebframeChoice', 'DevEnvsChoice', 'AIModelsChoice', 'SOAccount',
        'SOVisitFreq', 'SODuration', 'SOPartFreq', 'SOComm', 'SOFriction',
        'AISelect', 'AISent', 'AIAcc', 'AIComplex', 'AIAgents', 'AIAgentChange',
        'AIOpen', 'Industry'
    ]
    low_card_cols = [c for c in low_card_cols if c in X_train.columns]

    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    enc.fit(X_train[low_card_cols])

    def get_ohe_df(df, encoder, cols):
        encoded = encoder.transform(df[cols])
        names = encoder.get_feature_names_out(cols)
        return pd.DataFrame(encoded, columns=names, index=df.index)

    train_ohe = get_ohe_df(X_train, enc, low_card_cols)
    val_ohe   = get_ohe_df(X_val, enc, low_card_cols)
    test_ohe  = get_ohe_df(X_test, enc, low_card_cols)

    X_train = pd.concat([X_train.drop(columns=low_card_cols), train_ohe], axis=1)
    X_val   = pd.concat([X_val.drop(columns=low_card_cols),   val_ohe],   axis=1)
    X_test  = pd.concat([X_test.drop(columns=low_card_cols),  test_ohe],  axis=1)

    return X_train, X_val, X_test, enc

def apply_multihot_encoding(X_train, X_val, X_test, col):
    if col not in X_train.columns:
        return X_train, X_val, X_test, None

    def split_col(series):
        return series.fillna('').apply(lambda x: [v.strip() for v in str(x).split(';') if v.strip() != ''])

    X_train[col] = split_col(X_train[col])
    X_val[col]   = split_col(X_val[col])
    X_test[col]  = split_col(X_test[col])

    mlb = MultiLabelBinarizer()
    mlb.fit(X_train[col])

    prefix = col[:6]
    new_cols = [f"{prefix}_{c}" for c in mlb.classes_]

    train_mlb = pd.DataFrame(mlb.transform(X_train[col]), columns=new_cols, index=X_train.index)
    val_mlb   = pd.DataFrame(mlb.transform(X_val[col]),   columns=new_cols, index=X_val.index)
    test_mlb  = pd.DataFrame(mlb.transform(X_test[col]),  columns=new_cols, index=X_test.index)

    X_train = pd.concat([X_train.drop(columns=[col]), train_mlb], axis=1)
    X_val   = pd.concat([X_val.drop(columns=[col]),   val_mlb],   axis=1)
    X_test  = pd.concat([X_test.drop(columns=[col]),  test_mlb],  axis=1)

    return X_train, X_val, X_test, mlb

def apply_bayesian_encoding(X_train, X_val, X_test, y_train):
    high_card_cols = ['Country', 'DevType', 'AIExplain']
    high_card_cols = [c for c in high_card_cols if c in X_train.columns]

    enc = ce.TargetEncoder(cols=high_card_cols, smoothing=30)
    X_train[high_card_cols] = enc.fit_transform(X_train[high_card_cols], y_train)
    X_val[high_card_cols]   = enc.transform(X_val[high_card_cols])
    X_test[high_card_cols]  = enc.transform(X_test[high_card_cols])

    return X_train, X_val, X_test, enc




# 1. Apply VarianceThreshold to remove noise
def selecting_variance(X_train, X_val, X_test):
    num_features_before = X_train.shape[1]
    train_index = X_train.index
    val_index = X_val.index
    test_index = X_test.index
    print("Number of features before Variance Threshold: ", num_features_before)

    selector = VarianceThreshold(threshold=0.01)

    X_train_filtered = selector.fit_transform(X_train)
    X_val_filtered = selector.transform(X_val)
    X_test_filtered = selector.transform(X_test)

    surviving_cols = selector.get_feature_names_out()
    X_train = pd.DataFrame(X_train_filtered, columns=surviving_cols, index=train_index)
    X_val = pd.DataFrame(X_val_filtered, columns=surviving_cols, index=val_index)
    X_test = pd.DataFrame(X_test_filtered, columns=surviving_cols, index=test_index)

    print("Number of features after Variance Threshold: ", X_train.shape[1])

    return X_train, X_val, X_test, selector

def select_randomForst(X_train, X_val, X_test, y_train):
    rf_selection=RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=-1)
    rf_filter=SelectFromModel(rf_selection,threshold='median')
    rf_filter.fit(X_train,y_train)
    selected_cols=X_train.columns[rf_filter.get_support()]

    importances=rf_filter.estimator_.feature_importances_
    import_df=pd.DataFrame({'Feature':X_train.columns,'Importance':importances}).sort_values('Importance',ascending=False)
    print(import_df.head(20))

    X_train=X_train[selected_cols]
    X_val=X_val[selected_cols]
    X_test=X_test[selected_cols]

    print(f"Number of features after Tree selection: {X_train.shape[1]}")

    return X_train, X_val, X_test



def scale(X_train, X_val, X_test, numeric_cols):
    # 3. Scale only numerical features (avoid breaking encoded categories)
    existing_num_cols = [col for col in numeric_cols if col in X_train.columns]   
    scaler = StandardScaler()
    X_train[existing_num_cols] = scaler.fit_transform(X_train[existing_num_cols])
    X_val[existing_num_cols] = scaler.transform(X_val[existing_num_cols])
    X_test[existing_num_cols] = scaler.transform(X_test[existing_num_cols])

    return X_train, X_val, X_test, scaler