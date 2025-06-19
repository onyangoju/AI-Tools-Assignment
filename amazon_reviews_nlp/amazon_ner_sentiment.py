import spacy
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import ast

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Load first 10 reviews from test.ft.txt
with open("test.ft.txt", "r", encoding="utf-8") as f:
    data = [line.strip() for line in f.readlines() if line.strip()][:10]

results = []

print("\nNER + Sentiment Output (First 10 Reviews):\n")

for line in data:
    if not line.startswith("__label__"):
        continue

    try:
        label, text = line.split(" ", 1)
    except ValueError:
        continue

    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ['ORG', 'PRODUCT', 'WORK_OF_ART']]
    blob_sentiment = TextBlob(text).sentiment.polarity
    rule_sentiment = "Positive" if blob_sentiment > 0 else "Negative" if blob_sentiment < 0 else "Neutral"

    # Print Output
    print(f"Review: {text}")
    print(f"Label: {'Positive' if label == '__label__2' else 'Negative'}")
    if entities:
        print("Entities:")
        for ent_text, ent_label in entities:
            print(f" - {ent_text} ({ent_label})")
    else:
        print("Entities: None detected")
    print(f"Rule-Based Sentiment: {rule_sentiment} ({blob_sentiment:.2f})")
    print("-" * 60)

    # Collect for export
    results.append({
        "review": text,
        "label": "positive" if label == "__label__2" else "negative",
        "entities": entities,
        "sentiment": rule_sentiment,
        "score": round(blob_sentiment, 2)
    })

# Save to CSV and JSON
df = pd.DataFrame(results)
df.to_csv("review_sentiment_entities.csv", index=False)
df.to_json("review_sentiment_entities.json", orient="records", indent=2)

# Plot Sentiment Distribution
df['sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'gray'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("sentiment_distribution.png")
plt.show()

# Plot Most Frequent Entities
all_entities = df["entities"].apply(lambda x: ast.literal_eval(str(x)) if isinstance(x, str) else x)
flat_entities = [ent[0] for ent_list in all_entities for ent in ent_list]
top_entities = Counter(flat_entities).most_common(5)

if top_entities:
    names, freqs = zip(*top_entities)
    plt.figure()
    plt.bar(names, freqs, color='skyblue')
    plt.title("Top Named Entities (Brands/Products)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("entity_frequency.png")
    plt.show()
