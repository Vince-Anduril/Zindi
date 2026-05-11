"""
Template phrases extracted from training data analysis.

These are high-frequency opening/closing phrases that should appear in our predictions
to boost ROUGE. The model should be biased toward these patterns at inference.

Derived from outputs/analysis/phrases.json.
"""

# Most distinctive: Eng_Eth uses a fixed template "This is a question about, X."
# in 57% of answers (2215/3915). Detecting topic and applying template
# could give near-perfect ROUGE for that fraction.
ENG_ETH_TEMPLATE = "This is a question about, {topic}."

# STI topic keywords commonly appearing in Eng_Eth template
ENG_ETH_TOPICS = [
    "HIV/AIDS", "Herpes", "Syphilis", "HPV", "Gonorrhea",
    "Trichomoniasis", "Chlamydia", "Bacterial Vaginosis",
]

# High-frequency opening phrases per language (use as priors in fine-tuning)
# Format: subset -> list of (phrase, frequency)
TOP_OPENINGS = {
    "Eng_Eth": [
        ("This is a question about,", 2215),
        ("Gonorrhea is treated with antibiotics.", 10),
        ("HIV cannot be cured, but", 10),
        ("Herpes cannot be cured but", 9),
        ("No, that is a myth.", 8),
    ],
    "Eng_Uga": [
        ("Chlamydia also known as Chlamydial", 247),
        ("Genital Herpes or Herpes Simplex", 200),
        ("The only way to completely", 178),
        ("Gonorrhea can be traced back", 96),
        ("A chlamydia also known as", 82),
    ],
    "Eng_Ken": [
        ("HIV (Human Immunodeficiency Virus) is", 44),
        ("Yes, it is possible for", 34),
        ("Engaging in sexual activity with", 20),
    ],
    "Eng_Gha": [
        ("your school nurse or healthcare", 44),
        ("Healthcare providers play a crucial", 13),
        ("Adolescents can advocate for their", 11),
    ],
    "Aka_Gha": [
        ("wo sukuu ɔyarehwɛfo anaa akwahosan", 26),
        ("Mmabun betumi de wɔn ho", 21),
        ("Mmabun betumi ahwɛ ahu sɛ", 15),
    ],
    "Lug_Uga": [
        ("Kiramidiya era amanyiddwa nga Akawuka", 42),
        ("Eddagala eritta bbakitiiriya erya Penicillin,", 18),
        ("Enziku bulwadde bwa kikaba obusaasaanyizibwa", 17),
    ],
    "Swa_Ken": [
        ("virusi vya ukimwi (Virusi vya", 47),
        ("Ukimwi (Virusi vya Upungufu wa", 29),
        ("Tiba ya kurefusha maisha (ART),", 10),
    ],
    "Amh_Eth": [
        ("ቂጥኝ በፔኒሲሊን ይድናል። ካልታከመ አእምሮን፣", 5),
        ("ኤችአይቪ ሊድን አይችልም፣ ነገር ግን", 5),
        ("ኤች አይ ቪ ሊድን አይችልም", 4),
    ],
}


# Critical medical terms that MUST appear in answers per language
# (high-frequency vocabulary from trigram analysis - boost via logit bias)
KEY_MEDICAL_TERMS = {
    "Lug_Uga": [
        "Akawuka akaleeta Siriimu",      # HIV - 2362 occurrences
        "Kawuka akaleeta Siriimu",
        "Obuwuka obuleeta Obulwadde",    # STI
        "Obulwadde bw'Ekikaba",
    ],
    "Swa_Ken": [
        "virusi vya ukimwi",              # HIV virus - 1755 occurrences
        "Upungufu wa Kinga",              # Immunodeficiency
        "magonjwa ya zinaa",              # STIs
    ],
    "Amh_Eth": [
        "ኤች አይ ቪ",                          # HIV - 54 occurrences
        "ረጅም እና ጤናማ",                     # long and healthy
        "ኮንዶም መጠቀም",                      # condom use
        "የግብረ ሥጋ ግንኙነት",                  # sexual relations
    ],
    "Aka_Gha": [
        "nna ne awo",                     # sexual and reproductive
        "awo akwahosan ho",               # reproductive health
        "a wɔde wɔn",
    ],
    "Eng_Uga": [
        "also known as",
        "Herpes Simplex Virus (HSV)",
        "Chlamydial Genitourinary",
    ],
    "Eng_Gha": [
        "sexual and reproductive",
        "sexual and reproductive health",
        "Here are some",
    ],
    "Eng_Ken": [
        "It's important to",
        "HIV (Human Immunodeficiency Virus)",
        "the risk of HIV",
        "important to note",
    ],
}


def detect_eng_eth_topic(question: str) -> str | None:
    """For Eng_Eth questions, try to detect the STI topic for template application."""
    q = question.lower()
    topic_keywords = {
        "HIV/AIDS": ["hiv", "aids", "immunodeficiency"],
        "Herpes": ["herpes", "hsv"],
        "Syphilis": ["syphilis"],
        "HPV": ["hpv", "papilloma"],
        "Gonorrhea": ["gonorrhea", "gonorrhoea"],
        "Trichomoniasis": ["trichomon"],
        "Chlamydia": ["chlamydia"],
        "Bacterial Vaginosis": ["bacterial vaginosis", "bv"],
    }
    for topic, keywords in topic_keywords.items():
        if any(k in q for k in keywords):
            return topic
    return None


if __name__ == "__main__":
    # Verify template detection on real Eng_Eth questions
    import sys
    sys.path.insert(0, ".")
    from src.data import load_train
    df = load_train()
    eth = df[df["subset"] == "Eng_Eth"].head(20)
    for _, row in eth.iterrows():
        topic = detect_eng_eth_topic(row["input"])
        print(f"  topic={topic!s:25s} | Q: {row['input'][:80]}")
