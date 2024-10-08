import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('allergyPredModel/data.csv')
df['HAS_ALLERGY'] = df[['SHELLFISH_ALG_START', 'FISH_ALG_START', 'MILK_ALG_START', 
                        'SOY_ALG_START', 'EGG_ALG_START', 'WHEAT_ALG_START', 
                        'PEANUT_ALG_START', 'SESAME_ALG_START', 'TREENUT_ALG_START', 
                        'WALNUT_ALG_START', 'PECAN_ALG_START', 'PISTACH_ALG_START', 
                        'ALMOND_ALG_START', 'BRAZIL_ALG_START', 'HAZELNUT_ALG_START', 
                        'CASHEW_ALG_START']].notna().any(axis=1).astype(int)
df.drop(columns=['SHELLFISH_ALG_START', 'FISH_ALG_START', 'MILK_ALG_START', 
                 'SOY_ALG_START', 'EGG_ALG_START', 'WHEAT_ALG_START', 
                 'PEANUT_ALG_START', 'SESAME_ALG_START', 'TREENUT_ALG_START', 
                 'WALNUT_ALG_START', 'PECAN_ALG_START', 'PISTACH_ALG_START', 
                 'ALMOND_ALG_START', 'BRAZIL_ALG_START', 'HAZELNUT_ALG_START', 
                 'CASHEW_ALG_START'], inplace=True)

df = pd.get_dummies(df, columns=['GENDER_FACTOR', 'RACE_FACTOR', 'ETHNICITY_FACTOR', 'PAYER_FACTOR'], drop_first=True)

X = df.drop(columns=['HAS_ALLERGY'])
y = df['HAS_ALLERGY']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)


y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))


model.save('allergy_model.h5')