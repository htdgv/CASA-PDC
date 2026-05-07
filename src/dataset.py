import torch
from torch.utils.data import Dataset
import numpy as np
from ollama import chat
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util


class PsychologicalDefenseDataset1(Dataset):
    def __init__(self, texts, meta_features, dmrs_features, labels, tokenizer, max_len=256):
        self.texts = texts
        self.meta = torch.tensor(meta_features, dtype=torch.float)
        self.dmrs = torch.tensor(dmrs_features, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'meta_features': self.meta[idx],
            'dmrs_features': self.dmrs[idx],
            'labels': self.labels[idx]
            }


class PsychologicalDefenseDataset(Dataset):
        def __init__(self, texts, meta_features, labels, tokenizer, max_len=128):
            self.texts = texts
            self.meta_features = meta_features
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self): return len(self.labels)

        def __getitem__(self, item):
            encoding = self.tokenizer(
                self.texts[item],
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'meta_features': torch.tensor(self.meta_features[item], dtype=torch.float),
                'labels': torch.tensor(self.labels[item], dtype=torch.long)
            }




class DefenseDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.texts = df["input_text"].values
        self.labels = df["label"].values
        
        # Convert the column of lists into a standard 2D float array
        self.features = np.stack(df["clinical_features"].values).astype(np.float32)
        
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        encoding = self.tokenizer(
            str(self.texts[item]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'extra_features': torch.tensor(self.features[item], dtype=torch.float),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }
    
# =============================
def get_inference_input(data, level, n=3):
    level_data = data[data['label'] == level]
    return level_data.sample(n=n).reset_index(drop=True)

def format_dialogue_history(dialogue_turns):
    formatted_text = ""
    for turn in dialogue_turns:
        # Assuming your data saves speaker as 'supporter' or 'seeker'
        speaker = "Supporter" if turn['speaker'] == 'SUP' else "Seeker"
        text = turn['text']
        formatted_text += f"{speaker}: {text}\n"
    return formatted_text.strip()

def generate_stressor(history, target_turn, model='llama3'):
    if model=='qwen3.5:9b':
        prompt = f"""
            <|im_start|>system
            You are an expert clinical psychologist specializing in linguistic analysis. Your task is to identify the "Salient Stressor" — the core psychological conflict or external pressure — present in the user's last statement.
            <|im_end|>
            <|im_start|>user
            ### DIALOGUE CONTEXT:
            {history}

            ### TARGET UTTERANCE TO ANALYZE:
            "{target_turn}"

            ### INSTRUCTIONS:
            1. Analyze the psychological threat or conflict in the target utterance based on the provided context.
            2. Identify the primary Stressor Category (e.g., Interpersonal Conflict, Self-Esteem Threat, External Crisis, Health Anxiety, Role Strain).
            3. Provide a concise, one-sentence clinical description of the threat.

            ### OUTPUT FORMAT:
            1. **Stressor Category:** [Category Name]
            2. **Description:** [One sentence explanation]
            <|im_end|>
            <|im_start|>assistant
            """
    elif model == 'llama3':
        prompt = f"""
            ### TASK: Clinical Stressor Identification
            Identify the "Salient Stressor" causing psychological conflict in the Target Utterance.

            ### DIALOGUE CONTEXT:
            {history}

            ### TARGET UTTERANCE:
            "{target_turn}"

            ### OUTPUT FORMAT:
            1. Stressor Category: (e.g., Interpersonal Conflict, Self-Esteem Threat, External Crisis)
            2. Description: (One sentence explaining the threat)
            """
    response = chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
    )
    
    return response.message.content.strip().replace('"', '')

def generate_synthetic(mechanism_name, level, definition, pattern_description, 
                       example_1, example_2, example_3, 
                       history, stressor, model='llama3'):
    
    if model == 'qwen3.5:9b':
        prompt = f"""
            <|im_start|>system
            You are a clinical data synthesis engine. Your goal is to generate high-fidelity synthetic utterances for mental health research. You must strictly adhere to the provided defense mechanism definitions and dialogue history.
            <|im_end|>
            <|im_start|>user
            ### TASK
            Generate 5 NEW seeker utterances for the NEXT TURN that demonstrate the psychological defense mechanism: {mechanism_name}.

            ### CONTEXTUAL GROUNDING
            - **STRESSOR:** {stressor}
            - **DIALOGUE HISTORY:** 
            {history}

            ### DEFENSE SPECIFICATIONS
            - **Mechanism:** {mechanism_name} (Level {level})
            - **Definition:** {definition}
            - **Pattern:** {pattern_description}

            ### REFERENCE STYLE (FEW-SHOT)
            1. "{example_1}"
            2. "{example_2}"
            3. "{example_3}"

            ### CONSTRAINTS
            - The utterances must be natural and fit the current dialogue flow.
            - Output exactly 5 examples.
            - Use plain text only.
            - NO explanation.
            - NO markdown (no bolding, no italics).
            - NO code fences (no ```).
            - NO introductory or concluding remarks.

            ### OUTPUT
            <|im_end|>
            <|im_start|>assistant
            """
    
    elif model == 'llama3':
        prompt = f"""
            ### TASK: Generate Synthetic Psychological Defense Examples
            You are simulating a seeker in a mental health support chat.
            
            ### CONTEXTUAL GROUNDING:
            STRESSOR: {stressor}
            DIALOGUE HISTORY:
            {history}

            ### DEFENSE TO SIMULATE:
            Mechanism: {mechanism_name} (Level {level})
            Definition: {definition}
            Pattern: {pattern_description}

            ### REFERENCE STYLE (Few-Shot):
            1. "{example_1}"
            2. "{example_2}"
            3. "{example_3}"

            ### GOAL:
            Generate 5 NEW seeker utterances for the NEXT TURN using the {mechanism_name} defense.
            Ensure they follow the history and react to the stressor.

            ### OUTPUT FORMAT:
            1 string.
            No explanation, no markdown, no code fences.
        """
    
    response = chat(model=model, messages=[{'role': 'user', 'content': prompt}])
    content = response.message.content.strip()

    # Strip markdown code fences if the model wraps output in ```json ... ```
    content = re.sub(r'^```(?:json)?\s*', '', content)
    content = re.sub(r'\s*```$', '', content)
    content = content.strip()
    content = content.split(':\n\n')[-1]  # Get the part after "Return ONLY a strings. No explanation, no markdown, no code fences."
    return content

    # try:
    #     # Try parsing the whole response as JSON first (cleanest case)
    #     utterances = json.loads(content)
    #     if isinstance(utterances, list):
    #         utterances = [u for u in utterances if isinstance(u, str) and len(u.strip()) > 5]
    #         return utterances[:5]
    # except json.JSONDecodeError:
    #     pass

    # try:
    #     # Fallback: extract the first [...] block and parse it
    #     bracket_match = re.search(r'\[.*?\]', content, re.DOTALL)
    #     if bracket_match:
    #         utterances = json.loads(bracket_match.group())
    #         if isinstance(utterances, list):
    #             utterances = [u for u in utterances if isinstance(u, str) and len(u.strip()) > 5]
    #             return utterances[:5]
    # except (json.JSONDecodeError, AttributeError):
    #     pass

    # print(f"Parsing failed. Raw response:\n{content}")
    # return []



def calculate_self_bleu(sentences):
    """
    sentences: List of strings (synthetic utterances for ONE specific class)
    """
    if len(sentences) < 2:
        return 0.0
    
    bleu_scores = []
    smoothing = SmoothingFunction().method1
    
    # Tokenize all sentences
    tokenized_sentences = [s.lower().split() for s in sentences]
    
    for i in range(len(tokenized_sentences)):
        # The 'hypothesis' is the current sentence
        hypothesis = tokenized_sentences[i]
        # The 'references' are all other sentences in the group
        references = tokenized_sentences[:i] + tokenized_sentences[i+1:]
        
        # Calculate BLEU-4 (standard for diversity)
        score = sentence_bleu(references, hypothesis, smoothing_function=smoothing)
        bleu_scores.append(score)
    
    return sum(bleu_scores) / len(bleu_scores)


def calculate_semantic_adherence(utterances, stressors, embedder=None):
    """
    utterances: List of synthetic seeker responses
    stressors: List of the stressors used to generate those responses (1:1 mapping)
    """
    if embedder is None:
        embedder = SentenceTransformer('BAAI/bge-large-en-v1.5')
    # 1. Compute embeddings for both
    utt_embeddings = embedder.encode(utterances, convert_to_tensor=True)
    stressor_embeddings = embedder.encode(stressors, convert_to_tensor=True)
    
    # 2. Calculate Cosine Similarity for each pair
    # util.cos_sim returns a matrix; we want the diagonal (pair-wise)
    cosine_scores = util.cos_sim(utt_embeddings, stressor_embeddings)
    
    # Extract the diagonal (the score for each specific pair)
    adherence_scores = cosine_scores.diag().tolist()
    
    avg_adherence = sum(adherence_scores) / len(adherence_scores)
    return avg_adherence, adherence_scores


def quality_report(class_label, synthetic_sentences, stressors):
    diversity = calculate_self_bleu(synthetic_sentences)
    adherence, _ = calculate_semantic_adherence(synthetic_sentences, stressors)
    
    print(f"--- Quality Report for Class {class_label} ---")
    print(f"Sample Count: {len(synthetic_sentences)}")
    print(f"Diversity (Self-BLEU): {diversity:.4f} (Lower is more diverse)")
    print(f"Semantic Adherence: {adherence:.4f} (Relevance to Stressor)")
    
    if diversity > 0.7:
        print("Warning: High redundancy! Try increasing Temperature.")
    if adherence < 0.2:
        print("Warning: Low relevance! Check your prompt context.")



