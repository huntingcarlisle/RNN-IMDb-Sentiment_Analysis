def train_wordembed_lstm_dense_model(Optimizer, X_train, y_train, X_val, y_val):
    """ 
    Takes as input an optimizer function, 
    returns the fit score and model
    """
    # Initialize model
    model = Sequential()
    
    # Add layers
    model.add(Embedding(input_dim = 10000, output_dim = 128))
    model.add(LSTM(units=128))
    model.add(Dense(units=1, activation='sigmoid'))
    
    # Compile Model
    model.compile(loss='binary_crossentropy', 
                  optimizer = Optimizer, 
                  metrics=['accuracy'])
    
    # Train Model
    scores = model.fit(X_train, 
                       y_train, 
                       batch_size=128, 
                       epochs=10, 
                       validation_data=(X_val, y_val), 
                       verbose=0)
    
    return scores, model