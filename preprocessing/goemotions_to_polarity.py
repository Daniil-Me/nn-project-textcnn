import pandas as pd

INPUT_CSV = "go_emotions_dataset.csv"
POS_FILE = "rt-polarity.pos"
NEG_FILE = "rt-polarity.neg"

POSITIVE_LABELS = [
    "admiration",
    "amusement",
    "approval",
    "caring",
    "desire",
    "excitement",
    "gratitude",
    "joy",
    "love",
    "optimism",
    "pride",
    "relief",
]

NEGATIVE_LABELS = [
    "anger",
    "annoyance",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "fear",
    "grief",
    "nervousness",
    "remorse",
    "sadness",
]

DISCARD_LABELS = [
    "neutral",
    "confusion",
    "curiosity",
    "realization",
    "surprise",
]

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).replace("\r", " ").replace("\n", " ").strip()
    return " ".join(text.split())

def main():
    df = pd.read_csv(INPUT_CSV)

    needed = ["text"] + POSITIVE_LABELS + NEGATIVE_LABELS + DISCARD_LABELS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError("В CSV отсутствуют колонки: {}".format(missing))

    pos_lines = []
    neg_lines = []

    skipped_mixed = 0
    skipped_neutral_or_ambiguous = 0
    skipped_empty = 0

    for _, row in df.iterrows():
        text = clean_text(row["text"])
        if not text:
            skipped_empty += 1
            continue

        pos_count = int(row[POSITIVE_LABELS].sum())
        neg_count = int(row[NEGATIVE_LABELS].sum())
        discard_count = int(row[DISCARD_LABELS].sum())

        if pos_count > 0 and neg_count == 0 and discard_count == 0:
            pos_lines.append(text)
        elif neg_count > 0 and pos_count == 0 and discard_count == 0:
            neg_lines.append(text)
        elif pos_count > 0 and neg_count > 0:
            skipped_mixed += 1
        else:
            skipped_neutral_or_ambiguous += 1

    pos_lines = list(dict.fromkeys(pos_lines))
    neg_lines = list(dict.fromkeys(neg_lines))

    with open(POS_FILE, "w", encoding="utf-8") as f:
        for line in pos_lines:
            f.write(line + "\n")

    with open(NEG_FILE, "w", encoding="utf-8") as f:
        for line in neg_lines:
            f.write(line + "\n")

    print("Done.")
    print("Positive:", len(pos_lines))
    print("Negative:", len(neg_lines))
    print("Skipped mixed:", skipped_mixed)
    print("Skipped neutral/ambiguous:", skipped_neutral_or_ambiguous)
    print("Skipped empty:", skipped_empty)
    print("Created files:", POS_FILE, NEG_FILE)

if __name__ == "__main__":
    main()
