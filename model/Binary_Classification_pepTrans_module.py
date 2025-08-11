from imports import *
def wd_schedule(epoch):
    wd = 1e-4
    if epoch >= 30:
        wd *= 1e-2
    elif epoch >= 20:
        wd *= 1e-1
    #print('Weight decay: ', wd)
    return wd
        
def pepTrans_Binary_Classification(input_dim,X_train):
        initial_learning_rate = 0.001
        total_steps = 100 * (len(X_train) // 128) 

    # Set up a cosine decay schedule
        lr_schedule = CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=total_steps,
            alpha=0.01)  
        
        input_layer = Input(shape=(input_dim,))
        first_half = Lambda(lambda x: x[:, :1024])(input_layer)
        second_half = Lambda(lambda x: x[:, 1024:])(input_layer)

        first_half = Reshape((1024, 1))(first_half)
        second_half = Reshape((1024, 1))(second_half)
        #print(first_half.shape)

        cnn1 = Conv1D(3, 3, strides=1, activation='relu', padding='same',dilation_rate=2, kernel_regularizer=l2(0.00003), kernel_initializer=he_normal())(first_half)
        cnn1 = BatchNormalization()(cnn1)
        cnn1 = Dropout(0.3)(cnn1)
        cnn1 = Flatten()(cnn1)

        cnn2 = Conv1D(3, 3, strides=1, activation='relu', padding='same',dilation_rate=2, kernel_regularizer=l2(0.00003), kernel_initializer=he_normal())(second_half)
        cnn2 = BatchNormalization()(cnn2)
        cnn2 = Dropout(0.3)(cnn2)
        cnn2 = Flatten()(cnn2)

        concatenated = concatenate([cnn1, cnn2])

        combined_output = Dense(1500, activation='relu', kernel_regularizer=l2(0.0003))(concatenated)
        combined_output = BatchNormalization()(combined_output)
        combined_output = Dropout(0.4)(combined_output)

        output = Dense(1, activation='sigmoid')(combined_output)
        optimizer = AdamW(learning_rate=lr_schedule, weight_decay=wd_schedule(0))
        
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])

        return model