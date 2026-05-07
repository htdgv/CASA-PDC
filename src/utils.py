import pandas as pd 
import os 
print(f"Current working directory: {os.getcwd()}")
import re
import torch
from collections import defaultdict

def get_phatic_flag(text):
    """
    Identifies Level 0 (Phatic) communication.
    Criteria: Short length AND contains common social fillers.
    """
    # 1. Standardize and tokenize
    text_clean = text.lower().strip()
    tokens = text_clean.split()
    
    # 2. Dictionary of common phatic expressions
    PHATIC_WORDS = {
        'hi', 'hello', 'hey', 'thanks', 'thank', 'you', 'okay', 'ok', 'yeah', 
        'yes', 'no', 'right', 'mhm', 'uh-huh', 'sure', 'fine', 'sorry', 
        'i see', 'got it', 'understand', 'good'
    }
    
    # 3. Logic: If it's very short (e.g., <= 3 words) and consists 
    # mostly of these words, it's likely phatic.
    is_short = len(tokens) <= 3
    is_filler = all(token in PHATIC_WORDS or token in [',', '.', '!', '?'] for token in tokens)
    
    # Special case: If the seeker just says "I don't know" - often a phatic stalling tactic
    if text_clean in ["i don't know", "not sure", "maybe"]:
        return 1

    return 1 if (is_short and is_filler) else 0

def get_feeling_flag(text):
    """
    Checks if the turn is a direct disclosure of emotion.
    """
    feeling_keywords = {'feel', 'feeling', 'felt', 'upset', 'sad', 'happy', 'angry', 'hurt'}
    tokens = text.lower().split()
    
    # If the sentence starts with "I feel" or "I am feeling"
    if "i feel" in text.lower() or "i'm feeling" in text.lower():
        return 1
    
    # Or if a significant number of words are feeling-based
    feeling_overlap = [w for w in tokens if w in feeling_keywords]
    if len(feeling_overlap) >= 1:
        return 1
        
    return 0

def get_mature_flag(row):
    """
    Identifies Level 1 (Mature) markers.
    Criteria: Presence of insight words, decent length, 
    and reasonable emotional intensity.
    """
    # 1. Thresholds (as a researcher, you can tune these)
    # Mature defenses are usually more verbose than immature ones.
    is_verbose = row['feat_length'] > 12 
    # Significant insight density
    has_insight = row['feat_insight'] > 0.05 
    # Mature people tend to use 'I' to take ownership
    uses_ownership = row['feat_i_density'] > 0.05
    
    # 2. Logic: High insight + ownership + length
    if has_insight and is_verbose and uses_ownership:
        return 1
    
    return 0

def calculate_speed_of_change(df):
    speed_of_change = []
    for diag_id, group in df.groupby("dialogue_id"):
        group = group.sort_values(by="id").reset_index(drop=True)
        label_diff = group["label"].diff().fillna(0)  # calculate the difference in labels between turns
        speed_of_change.append({
            "dialogue_id": diag_id,
            "speed_of_change": label_diff.abs().mean()  # average absolute change in labels per turn
        })
    return pd.DataFrame(speed_of_change)

def calculate_rising_falling(df, diag_start_label, diag_end_label):
    rising_count = 0
    falling_count = 0
    stable = 0
    for diag_id, group in df.groupby("dialogue_id"):
        group = group.sort_values(by="id").reset_index(drop=True)
        if diag_start_label.loc[diag_start_label['dialogue_id'] == diag_id, 'label'].values[0] > diag_end_label.loc[diag_end_label['dialogue_id'] == diag_id, 'label'].values[0]:
            falling_count += 1
        elif diag_start_label.loc[diag_start_label['dialogue_id'] == diag_id, 'label'].values[0] < diag_end_label.loc[diag_end_label['dialogue_id'] == diag_id, 'label'].values[0]:
            rising_count += 1
        else: 
            stable += 1
    return rising_count, falling_count, stable


# Q3: where in the dialogue do seekers tend to open up more? do turn position analysis
def get_utterance_length(text):
    """
    Measures word count as a proxy for narrative disclosure.
    Clinically: Short turns = possible Level 1 (Action) or 0 (Functional). 
    Long turns = possible Level 6 (Obsessional) or 7 (Adaptive).
    """
    if not isinstance(text, str) or text.strip() == "":
        return 0
    words = text.split()
    return len(words)

def get_i_pronoun_density(text):
    """
    Calculates the ratio of 1st-person singular pronouns.
    Clinically: High density = 'Self-Observation' (Level 7)
    or 'Subjective Distortion' (Level 2/3).
    """
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
    
    words = text.lower().split()
    total_words = len(words)
    if total_words == 0: return 0.0
    
    # Matching: I, me, my, mine, myself
    i_pattern = r'\b(i|me|my|mine|myself)\b'
    i_matches = re.findall(i_pattern, text.lower())
    
    return len(i_matches) / total_words

def get_embedding_intensity(text, tokenizer, model):
    """
    Measures the 'Norm' of the hidden states.
    In high-level NLP research, a higher vector norm often correlates 
    with more complex/intense semantic content.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use the [CLS] token representation (index 0)
    embeddings = outputs.last_hidden_state[:, 0, :]
    
    # Calculate the L2 Norm (magnitude) of the vector
    intensity_score = torch.norm(embeddings, p=2).item()
    
    return intensity_score

def get_mental_emotion_intensity(text, emotion_pipe):
    """
    Uses Mental-RoBERTa logic to determine how 'charged' an utterance is.
    Clinically: If a seeker uses clinical terms (e.g., 'empty', 'numb'), 
    this model recognizes the gravity better than a general model.
    """
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
    
    result = emotion_pipe(text[:512])[0]
    
    label = result['label']
    score = result['score']
    
    if label.lower() == 'neutral':
        return 1 - score  # If it's 90% neutral, intensity is 0.1
    else:
        return score      # If it's 90% negative, intensity is 0.9

def get_emotion_intensity(text, emotion_pipe):
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
    # The latest model uses labels like 'positive', 'negative', 'neutral'
    res = emotion_pipe(text[:512])[0]
    
    if res['label'].lower() == 'neutral':
        return 1 - res['score']
    else:
        return res['score']


def get_insight_density(text):
    """
    Matches keywords associated with mentalization and insight.
    Keywords based on LIWC 'Insight' and 'Cognitive Process' categories.
    """
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
    
    words = text.lower().split()
    total_words = len(words)
    if total_words == 0: return 0.0
    
    # Clinical Insight Keywords
    # TODO: review this as it seems to be hard coded and may not capture all relevant terms. Consider expanding with synonyms or related concepts.
    insight_keywords = [
        'realize', 'understand', 'think', 'believe', 'know', 
        'notice', 'decide', 'feel', 'reason', 'conclude', 
        'aware', 'insight', 'meaning', 'because', 'why'
    ]
    
    # Count occurrences
    count = sum(1 for word in words if any(k in word for k in insight_keywords))
    
    return count / total_words


def calculate_openup_ratio(group):
    # Sort by turn index to ensure temporal order
    group = group.sort_values('turn_index')
    n = len(group)
    if n < 3: return None # Need enough turns to compare
    
    # Split into Early and Late phases
    cut = int(n * 0.3)
    early_phase = group.head(cut)['CDI'].mean()
    late_phase = group.tail(cut)['CDI'].mean()
    
    return late_phase / early_phase if early_phase > 0 else 0

def calculate_refined_openup_ratio(group):
    group = group.sort_values('turn_index')
    n = len(group)
    if n < 4: return None
    
    # 1. Early Phase: First 30%
    early_cut = int(n * 0.3)
    early_cdi = group.head(early_cut)['CDI'].mean()
    
    # 2. Late Phase: 60% to 90% (This avoids the "Goodbye" turns at 90-100%)
    late_start = int(n * 0.6)
    late_end = int(n * 0.9)
    late_cdi = group.iloc[late_start:late_end]['CDI'].mean()
    
    return late_cdi / early_cdi if early_cdi > 0 else 0

def get_opening_up_turn(group):
    # 1. Safety check: Handle empty groups if they exist
    if group.empty:
        return pd.Series({'opening_up_turn': None, 'opening_up_cdi': None, 'defense_level': None})

    # 2. Find the index of the highest CDI
    # idxmax() finds the index of the row with the maximum value
    idx = group['CDI'].idxmax()
    max_row = group.loc[idx]

    # 3. Return a Series (dialogue_id will be the index of opening_up_turns)
    return pd.Series({
        'opening_up_turn': max_row['turn_index'],
        'opening_up_cdi': max_row['CDI'],
        'defense_level': max_row['label'], # Very useful for Phase 2 analysis!
        'text_snippet': str(max_row['text'])[:50] + "..." # To verify it looks like 'opening up'
    })

def calculate_dmrs_mechanism(item_weights):
    """Post-processing logic to group scores by defense mechanism."""
    grouped = defaultdict(list)
    for key, value in item_weights.items():
        prefix = key.rsplit("_", 1)[0]
        grouped[prefix].append(value)

    return {
        defense: (sum(values) - 5) * 100 / 234
        for defense, values in grouped.items()
    }