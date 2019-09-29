from deeppavlov import build_model, configs

document = ['The rain in Spain falls mainly on the plain.']
question = ['Where does the rain in Spain fall?']

model = build_model(configs.squad.squad, download=False)
model(document, question)
