from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    {
        "text": "Fire doors on escape routes require EI60 rating.",
        "country": "Ireland",
        "discipline": "Fire",
        "status": "Approved"
    },
    {
        "text": "Internal office doors do not require a fire rating.",
        "country": "UK",
        "discipline": "Fire",
        "status": "Approved"
    }
]

country_filter = "UK"
filtered_documents = [doc for doc in documents if doc["country"] == country_filter]
filtered_texts = [doc["text"] for doc in filtered_documents]

#1. Create representation of our documents

vectorizer = TfidfVectorizer()

document_vector = vectorizer.fit_transform(filtered_texts)

#breakpoint
vectorizer.get_feature_names_out()
document_vector.shape


#print("Docuemnt vector: ", document_vector)
#question = "What requirements exist for emergency doors?" #Test 1
#question = "Tell me about Zigbee network extenders" #Test 2
question = "Which devices improve my wireless smart home network?" #Test 3
#question = "How often should I clean the coffee machine?" #Test 4

question_vector = vectorizer.transform([question])
#print("Question vector: ", question_vector)


scores = cosine_similarity(
    question_vector,
    document_vector
    )[0]

#print("Scores: ", scores)

results = sorted(
    zip(scores, filtered_documents),
    key=lambda result: result[0],
    reverse=True
)

for score, document in results:
    print(
        f"{score:.3f} | "
        f"{document['text']} | "
        f"{document['country']} | "
        f"{document['discipline']} | "
        f"{document['status']}"
    )
