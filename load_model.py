from keras.models import model_from_json
def load_model(images, resultados):
    # load json and create model
    json_file = open("modelos_treinados/canhoto-model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("modelos_treinados/canhoto-model.h5")
    print("Loaded model from disk")
     
    # evaluate loaded model on test data
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = model.evaluate(images, resultados)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
    return score[1]*100