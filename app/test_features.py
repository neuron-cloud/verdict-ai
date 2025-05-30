from features import extract_features

samples = [
    ("HUMAN", """
As a senior medical student transitioning into neurosurgery, the challenges and ethical complexities of the field have been vividly illustrated...
[ FULL TEXT HERE ]
"""),
    ("HUMAN", """
Background: Epidermoid tumors (ETs), though benign, present unique surgical challenges due to their intricate involvement with brain structures...
[ FULL TEXT HERE ]
"""),
    ("HUMAN", """
Dear Class of 2024,

First off, Happy New Year! I hope you are all doing well...
[ FULL TEXT HERE ]
"""),
    ("AI", """
🧠 So How Could This Work?
Here’s where your idea becomes unique—because you have both clinical and technical insight...
[ FULL TEXT HERE ]
"""),
    ("AI", """
This is phenomenal, Myke. You just built the first working prototype of Pathway AI—and it’s clean, polished, and genuinely functional...
[ FULL TEXT HERE ]
"""),
    ("AI", """
Perfect. Claude is a great choice—especially for this type of structured reasoning and multi-step flow...
[ FULL TEXT HERE ]
"""),
]

print("label,avg_sentence_length,sentence_variance,lexical_diversity,readability_score,entropy,word_count")

for label, text in samples:
    features = extract_features(text)
    print(f"{label},{features['avg_sentence_length']},{features['sentence_variance']},{features['lexical_diversity']},{features['readability_score']},{features['entropy']},{features['word_count']}")
