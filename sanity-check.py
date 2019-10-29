import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from deeppavlov import build_model, configs

document = ['The rain in Spain falls mainly on the plain.']
question = ['Where does the rain in Spain fall?']

try:
    model = build_model(configs.squad.squad, download=False)
except Exception:
    model = build_model(configs.squad.squad, download=True)
print(document[0])
print()
print(f"{question[0]} {model(document, question)[0][0]}")
