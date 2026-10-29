from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')



#text = "Zigbee routers extend the Zigbee mesh network."
#print(embedding)
#print("Dimensions: ", embedding.shape)

# texts = [
#     "Zigbee routers extend the Zigbee mesh network.",
#     "Devices that improve wireless smart home networks.",
#     "Coffee machines must be cleaned every Friday."
# ]

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

document_vectors = model.encode(documents)

#question = "What requirements exist for emergency doors?" #Test 1
#question = "Tell me about Zigbee network extenders" #Test 2
question = "Which devices improve my wireless smart home network?" #Test 3
#question = "How often should I clean the coffee machine?" #Test 4

question_vector = model.encode([question])

#print("Shape: ", embeddings.shape)


scores = cosine_similarity(
    question_vector,
    document_vectors    
)[0]


results = sorted(
    zip(scores, documents),
    reverse=True
)

top_k = 3
top_results = results[:top_k]

# for score, document in top_results:
#     print("{:.3f} | {}".format(score, document))

context = "\n\n".join(document for score, document in top_results)

# print(context)

prompt = """
You are a technical assiistant.
Aswer the QUESTION using only the information in the CONTEXT.
If the CONTEXT does not contain enough information to asnwer the question, say: "I don't have enough information."
Do not use information that is not contained in the CONTEXT.

CONTEXT:
{context}

QUESTION:
{question}
""".format(context=context, question=question)

print(prompt)