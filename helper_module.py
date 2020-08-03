
#function for one hot encoding
def one_hot_enc(df,cols):
    df = pd.get_dummies(df,prefix_sep="_",columns=cols)
    return df


#Function for frequency encoding
def freq_enc(df,cols):
    for col in cols:
        df[col] = df[col].map(df[col].value_counts().to_dict())/len(df)
    return df


#Function for label encoding
def label_enc(df,cols):
    for col in cols:
        df[col] = df[col].factorize()[0]
    return df


#Target encoding with KFold
def mean_target(data, cols):
    kf = KFold(5)
    a = pd.DataFrame()
    for tr_ind, val_ind in kf.split(data):
        X_tr, X_val= data.iloc[tr_ind].copy(), data.iloc[val_ind].copy()
        for col in cols:
            means = X_val[col].map(X_tr.groupby(col).target.mean())
            X_val[col + '_mean_target'] = means + 0.0001
        a = pd.concat((a, X_val))
    prior = target.mean()
    a.fillna(prior, inplace=True)
    return a

# Fuction for automatic feature interaction
def feature_interact(df, col):
    for i in s_col:
        for j in s_col: df[i + '/' + j] = df[i] / df[j]

    for i in s_col:
        for j in s_col: df[i + '+' + j] = df[i] + df[j]

    for i in s_col:
        for j in s_col: df[i + '-' + j] = df[i] - df[j]

    for i in s_col:
        for j in s_col: df[i + '*' + j] = df[i] * df[j]
    return df


#Function for manhattan distance when two latitudes and two longitudes are given
def manhattan_distance(lat1, lng1, lat2, lng2):
    a = np.abs(lat2 -lat1)
    b = np.abs(lng1 - lng2)
    return a + b

# Fuction for haversine array
def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

# Function for bearing array
def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

