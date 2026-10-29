from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Fire doors on escape routes require EI60 rating.",
    "Internal office doors do not require a fire rating.",
    "Door marks must follow D-Level-Number.",
    "Emergency exit doors must not be locked during occupancy.",
    "Air terminals require a unique Mark.",
    "Revit models must use approved project coordinates.",
    "Zigbee routers extend the Zigbee mesh network.",
    "Battery-powered Zigbee devices normally do not act as routers.",
    "Aqara motion sensors communicate using Zigbee.",
    "Coffee machines must be cleaned every Friday."
]

#1. Create representation of our documents

vectorizer = TfidfVectorizer()

document_vector = vectorizer.fit_transform(documents)

#breakpoint
vectorizer.get_feature_names_out()
document_vector.shape


#print("Docuemnt vector: ", document_vector)
question = "What requirements exist for emergency doors?" #Test 1
#question = "Tell me about Zigbee network extenders" #Test 2
#question = "Which devices improve my wireless smart home network?" #Test 3
#question = "How often should I clean the coffee machine?" #Test 4

question_vector = vectorizer.transform([question])
#print("Question vector: ", question_vector)


scores = cosine_similarity(
    question_vector,
    document_vector
    )[0]

#print("Scores: ", scores)

results = sorted(
    zip(scores, documents),
    reverse=True
)

for score, document in results:
    print(f"{score:.3f} | {document}")

