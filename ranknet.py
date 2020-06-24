import numpy as np
from keras import backend
from keras.layers import Activation, Dense, Input, Subtract
from keras.models import Model
from keras import regularizers
import keras

def accuracy(get_score, pairwise_object, coarse = True):
    acc = 0
    total = 0
    scores = get_score([pairwise_object.standardized_features.T])
    scores = scores[0].reshape(1,-1)[0]
    for i in range(pairwise_object.num_items):
        for j in range(i+1, pairwise_object.num_items):
            if (pairwise_object.training_data[i,j] == 1 and coarse) or (pairwise_object.training_data[i,j] == 0 and not coarse):
                if pairwise_object.comparison_data[i,j] + pairwise_object.comparison_data[j, i] > 0:
                    total +=1
                    score_i = scores[i]
                    score_j = scores[j]
                    # print(score_i - score_j)
                    # print(pairwise_object.preference_matrix[i,j] - .5)
                    if (pairwise_object.preference_matrix[i,j] - .5)*(score_i - score_j)>0:
                        acc +=1
    return acc / total

'''
pairwise_data_train/pairwise_data_validate/pairwise_data_test: objects of type fit_model_class
nodes: number of nodes in first hidden layer
c: l2 regularization strength
X_1: feature vectors of all the winning items that is num_comparisons x dim
X_2: feature vectors of all the losing items that is num_comparisons x dim
y: vector of all ones of length num_comparisons
'''
def run_ranknet(pairwise_data_train, pairwise_data_validate, pairwise_data_test, nodes, c, X_1, X_2, y, NUM_EPOCHS = 800, BATCH_SIZE = 250):
#def run_ranknet(nodes, c, ranking_object, X_1, X_2, y, NUM_EPOCHS = 800, BATCH_SIZE = 250):
    _, INPUT_DIM = X_1.shape
    # np.random.seed(0)

    # Model.
    h_1 = Dense(nodes, activation="relu", kernel_regularizer=regularizers.l2(c))
    # h_2 = Dense(4, activation="relu")
    # h_3 = Dense(2, activation="relu")
    # s = Dense(1)
    s = Dense(1, kernel_regularizer=regularizers.l2(c))

    # Relevant document score.
    rel_doc = Input(shape=(INPUT_DIM,), dtype="float32")
    h_1_rel = h_1(rel_doc)
    # h_2_rel = h_2(h_1_rel)
    # h_3_rel = h_3(h_2_rel)
    rel_score = s(h_1_rel)

    # Irrelevant document score.
    irr_doc = Input(shape=(INPUT_DIM,), dtype="float32")
    h_1_irr = h_1(irr_doc)
    # h_2_irr = h_2(h_1_irr)
    # h_3_irr = h_3(h_2_irr)
    irr_score = s(h_1_irr)

    # Subtract scores.
    diff = Subtract()([rel_score, irr_score])

    # Pass difference through sigmoid function.
    prob = Activation("sigmoid")(diff)

    # Build model.
    model = Model(inputs=[rel_doc, irr_doc], outputs=prob)
    keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    keras.layers.Dropout(1, noise_shape=None, seed=None)

    # Train model.
    history = model.fit([X_1, X_2], y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1)

    # Generate scores from document/query features.
    get_score = backend.function([rel_doc], [rel_score])

    train_coarse_acc = accuracy(get_score, pairwise_data_train, coarse = True)
    #the split for coarse train ddin't work right
    train_fine_acc = 0

    validate_coarse_acc =  accuracy(get_score, pairwise_data_validate, coarse = True)
    validate_fine_acc = accuracy(get_score, pairwise_data_validate, coarse = False)

    test_coarse_acc = accuracy(get_score, pairwise_data_test, coarse = True)
    test_fine_acc = accuracy(get_score, pairwise_data_test, coarse = False)

    return train_coarse_acc, train_fine_acc, validate_coarse_acc, validate_fine_acc, test_coarse_acc, test_fine_acc
