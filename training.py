import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from itertools import product
import json
import joblib

np.random.seed(42)
tf.random.set_seed(42)

df = pd.read_csv("tess_preprocessed_scaled.csv")
X = df.drop(columns=["tfopwg_disp"]).values
y = df["tfopwg_disp"].values.astype(int)

feature_names = df.drop(columns=["tfopwg_disp"]).columns.tolist()

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)

X_test_df = pd.DataFrame(X_test, columns=feature_names)
X_test_df["tfopwg_disp"] = y_test
X_test_df.to_csv("x_test_y_test.csv", index=False)
print("Saved test set as 'x_test_y_test.csv'")

param_grid = {
    'learning_rate': [0.0001],
    'batch_size': [64],
    'dropout_rate': [0.2],
    'l2_reg': [0.0005],
    'architecture': [
        [256, 128, 64, 32]
    ]
}

def create_model_grid(input_dim, architecture, dropout_rate, l2_reg, num_classes=3):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    
    for i, units in enumerate(architecture):
        model.add(layers.Dense(units, kernel_regularizer=keras.regularizers.l2(l2_reg)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        current_dropout = dropout_rate * (1 - i * 0.1)
        model.add(layers.Dropout(max(current_dropout, 0.1)))
    
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def grid_search(X_train, y_train, X_val, y_val, param_grid, max_combinations=50):
    results = []
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = list(product(*values))
    
    if len(all_combinations) > max_combinations:
        print(f"Total combinations: {len(all_combinations)}, limiting to {max_combinations}")
        np.random.shuffle(all_combinations)
        all_combinations = all_combinations[:max_combinations]
    else:
        print(f"Total combinations to try: {len(all_combinations)}")
    
    for idx, combination in enumerate(all_combinations):
        params = dict(zip(keys, combination))
        print(f"\n{'='*60}")
        print(f"Testing combination {idx + 1}/{len(all_combinations)}")
        print(f"Parameters: {params}")
        print(f"{'='*60}")
        
        try:
            model = create_model_grid(
                input_dim=X_train.shape[1],
                architecture=params['architecture'],
                dropout_rate=params['dropout_rate'],
                l2_reg=params['l2_reg']
            )
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=0
            )
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=params['batch_size'],
                callbacks=[early_stop],
                verbose=0
            )
            
            val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
            val_f1 = f1_score(y_val, y_pred, average='weighted')
            
            result = {
                'params': params.copy(),
                'val_loss': float(val_loss),
                'val_accuracy': float(val_accuracy),
                'val_f1_score': float(val_f1),
                'epochs_trained': len(history.history['loss'])
            }
            results.append(result)
            
            print(f"Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val Loss: {val_loss:.4f}")
            
            del model
            keras.backend.clear_session()
            
        except Exception as e:
            print(f"Error with combination: {e}")
            continue
    
    return results

print("Starting grid search...")
results = grid_search(X_train, y_train, X_val, y_val, param_grid, max_combinations=50)

results_sorted = sorted(results, key=lambda x: x['val_f1_score'], reverse=True)

print("\n" + "="*80)
print("TOP 5 PARAMETER COMBINATIONS")
print("="*80)

for i, result in enumerate(results_sorted[:5]):
    print(f"\nRank {i+1}:")
    print(f"  Parameters: {result['params']}")
    print(f"  Val Accuracy: {result['val_accuracy']:.4f}")
    print(f"  Val F1 Score: {result['val_f1_score']:.4f}")
    print(f"  Val Loss: {result['val_loss']:.4f}")
    print(f"  Epochs: {result['epochs_trained']}")

with open('grid_search_results.json', 'w') as f:
    json.dump(results_sorted, f, indent=2)
print("\nGrid search results saved to 'grid_search_results.json'")

best_params = results_sorted[0]['params']
print(f"\n{'='*60}")
print("Training final model with best parameters...")
print(f"Best params: {best_params}")
print(f"{'='*60}\n")

final_model = create_model_grid(
    input_dim=X_train.shape[1],
    architecture=best_params['architecture'],
    dropout_rate=best_params['dropout_rate'],
    l2_reg=best_params['l2_reg']
)

final_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=best_params['learning_rate']),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1),
    ModelCheckpoint('best_model_grid_search.keras', monitor='val_loss', save_best_only=True, verbose=1)
]

history = final_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=best_params['batch_size'],
    callbacks=callbacks,
    verbose=1
)

test_loss, test_accuracy = final_model.evaluate(X_test, y_test)
y_pred = np.argmax(final_model.predict(X_test), axis=1)
test_f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n{'='*60}")
print(f"FINAL MODEL PERFORMANCE")
print(f"{'='*60}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Saving final model
final_model.save('the_model.keras')

