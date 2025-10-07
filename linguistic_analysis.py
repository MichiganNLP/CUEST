import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import textstat
from collections import Counter
from tqdm import tqdm
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import os

embedder = SentenceTransformer('all-MiniLM-L6-v2')


class CuriosityMetrics:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.answerability_model = pipeline('text-classification', model='facebook/bart-large-mnli')
    
    def semantic_ambiguity(self, text, context=None):
        """Estimate semantic ambiguity with polysemy, POS, and context embedding."""
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha() and w not in self.stop_words]
        if not words:
            return 0
        
        ambiguous_words = ['bank', 'bark', 'bat', 'bow', 'fair', 'kind', 'match', 'mean', 
                          'park', 'play', 'right', 'rock', 'scale', 'spring', 'stick', 
                          'strike', 'table', 'tank', 'tie', 'watch']
        ambiguous_count = sum(1 for w in words if w in ambiguous_words)
        
        pos_tags = pos_tag(words)
        pos_variety = {}
        for word, pos in pos_tags:
            pos_variety.setdefault(word, set()).add(pos)
        multi_pos_words = sum(1 for pos_set in pos_variety.values() if len(pos_set) > 1)
        base_score = (ambiguous_count + multi_pos_words) / len(words)
        
        if context:
            text_emb = embedder.encode([text])
            context_emb = embedder.encode([context])
            similarity = np.dot(text_emb, context_emb.T) / (np.linalg.norm(text_emb) * np.linalg.norm(context_emb))
            embed_score = 1 - similarity[0][0]
        else:
            embed_score = 0
        
        return (base_score + embed_score) / 2

    def rhetorical_devices(self, text):
        """Detect rhetorical devices with expanded set."""
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())

        device_count = 0
        word_freq = Counter(words)
        repeated_words = sum(1 for w, freq in word_freq.items() if freq > 2 and w.isalpha())
        device_count += repeated_words
        
        questions = sum(1 for s in sentences if s.strip().endswith('?'))
        device_count += questions
        
        for sentence in sentences:
            ws = [w for w in word_tokenize(sentence.lower()) if w.isalpha()]
            if len(ws) > 1:
                first_letters = [w[0] for w in ws]
                letter_freq = Counter(first_letters)
                alliteration = sum(1 for l, freq in letter_freq.items() if freq > 2)
                device_count += alliteration
        
        sentence_lengths = [len(word_tokenize(s)) for s in sentences]
        length_groups = Counter(sentence_lengths)
        parallel = sum(1 for l, f in length_groups.items() if f > 1)
        device_count += parallel
        
        analogy_markers = ['like', 'as if', 'similar to']
        metaphors = sum(any(marker in s.lower() for marker in analogy_markers) for s in sentences)
        device_count += metaphors
        
        return device_count / len(sentences) if sentences else 0

    def open_ended_questions(self, text):
        """Proportion of questions that are open-ended and not directly answerable."""
        sentences = sent_tokenize(text)
        questions = [s for s in sentences if s.strip().endswith('?')]
        if not questions:
            return 0
        
        open_markers = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'whose', 'whom']
        open_ended_count = 0
        
        for question in questions:
            tokens = word_tokenize(question.lower())
            marker = any(m in tokens for m in open_markers)
            
            try:
                pred = self.answerability_model(question)
                is_answerable = (pred[0]['label'] == 'LABEL_1')
            except:
                is_answerable = False
            
            if marker and not is_answerable:
                open_ended_count += 1
        
        return open_ended_count / len(questions)

    def cohesion_score(self, text):
        """Lexical & semantic cohesion using overlap and embeddings."""
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 1.0
        
        sentence_words = []
        for s in sentences:
            ws = set(word_tokenize(s.lower()))
            ws = {w for w in ws if w.isalpha() and w not in self.stop_words}
            sentence_words.append(ws)
        
        overlaps = []
        for i in range(len(sentence_words) - 1):
            inter = sentence_words[i] & sentence_words[i + 1]
            union = sentence_words[i] | sentence_words[i + 1]
            overlaps.append(len(inter) / len(union) if union else 0)
        
        lexical_cohesion = np.mean(overlaps) if overlaps else 0
        
        transition_words = ['however', 'therefore', 'moreover', 'furthermore', 'consequently', 
                           'nevertheless', 'meanwhile', 'subsequently', 'additionally', 
                           'similarly', 'conversely', 'thus', 'hence']
        transition_count = sum(1 for s in sentences for w in word_tokenize(s.lower()) if w in transition_words)
        transition_cohesion = transition_count / len(sentences)
        
        sent_embs = embedder.encode(sentences)
        sem_sims = []
        for i in range(len(sentences)-1):
            sim = np.dot(sent_embs[i], sent_embs[i+1]) / (np.linalg.norm(sent_embs[i]) * np.linalg.norm(sent_embs[i+1]))
            sem_sims.append(sim)
        semantic_cohesion = np.mean(sem_sims) if sem_sims else 0
        
        cohesion = (lexical_cohesion + min(transition_cohesion, 1.0) + semantic_cohesion) / 3
        return cohesion


def analyze_text_metrics_by_columns(df, country_columns=None):
    """
    Analyze text metrics for countries where each country is a separate column
    """
    analyzer = CuriosityMetrics()
    results = []
    
    if country_columns is None:
        country_columns = [col for col in df.columns if col.lower() != 'topic']
    
    for country in tqdm(country_columns, desc="Processing countries"):
        if country in df.columns:
            country_texts = df[country].dropna().astype(str)
            combined_text = ' '.join(country_texts)
            if not combined_text.strip():
                continue
            
            metrics = {
                'Country': country,
                'Semantic_Ambiguity': analyzer.semantic_ambiguity(combined_text),
                'Rhetorical_Devices': analyzer.rhetorical_devices(combined_text),
                'Open_Ended_Questions': analyzer.open_ended_questions(combined_text),
                'Cohesion_Score': analyzer.cohesion_score(combined_text),
                'Text_Length': len(combined_text),
                'Sentence_Count': len(sent_tokenize(combined_text)),
                'Entry_Count': len(country_texts)
            }
            results.append(metrics)
    
    return pd.DataFrame(results)


def analyze_single_country_file(filepath, country_name):
    """
    Analyze text metrics for a single country file (country-specific files)
    Assumes the file has a 'Topic' column and a country column
    """
    analyzer = CuriosityMetrics()
    
    try:
        df = pd.read_csv(filepath)
        
        # Find the country column (not 'Topic')
        country_col = [col for col in df.columns if col.lower() != 'topic'][0]
        
        country_texts = df[country_col].dropna().astype(str)
        combined_text = ' '.join(country_texts)
        
        if not combined_text.strip():
            return None
        
        metrics = {
            'Country': country_name,
            'Semantic_Ambiguity': analyzer.semantic_ambiguity(combined_text),
            'Rhetorical_Devices': analyzer.rhetorical_devices(combined_text),
            'Open_Ended_Questions': analyzer.open_ended_questions(combined_text),
            'Cohesion_Score': analyzer.cohesion_score(combined_text),
            'Text_Length': len(combined_text),
            'Sentence_Count': len(sent_tokenize(combined_text)),
            'Entry_Count': len(country_texts)
        }
        
        return metrics
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return None


def process_all_files(base_path='.'):
    """
    Process all file combinations based on your actual file structure:
    - llama3_8b_[source]_country_all_obj[1/2].csv
    - llama3_8b_[source]_country_[brazil/philippines/uk]_obj[1/2].csv
    """
    
    objectives = ['obj1', 'obj2']
    sources = ['yahoo', 'reddit', 'yahooreddit']
    countries = ['brazil', 'philippines', 'uk']
    
    all_results = {}
    
    for obj in objectives:
        for source in sources:
            # Process country_all file
            print(f"\n{'='*60}")
            filename = f"llama3_8b_{source}_country_all_{obj}.csv"
            filepath = os.path.join(base_path, filename)
            
            if os.path.exists(filepath):
                print(f"Processing: {filename}")
                try:
                    df = pd.read_csv(filepath)
                    results_df = analyze_text_metrics_by_columns(df)
                    
                    key = f"{obj}_{source}_country_all"
                    all_results[key] = results_df
                    
                    output_filename = f"text_analysis_{obj}_{source}_country_all.csv"
                    results_df.to_csv(output_filename, index=False)
                    print(f"✓ Results saved to: {output_filename}")
                    
                    print("\nResults:")
                    print(results_df.round(3))
                    
                    print("\nCountries by Semantic Ambiguity:")
                    print(results_df.sort_values('Semantic_Ambiguity', ascending=False)[['Country', 'Semantic_Ambiguity']].round(3))
                    
                except Exception as e:
                    print(f"✗ Error processing {filename}: {str(e)}")
            else:
                print(f"Warning: File not found - {filename}")
            
            # Process country_specific files (brazil, philippines, uk)
            print(f"\n{'='*60}")
            print(f"Processing country-specific files for {source} - {obj}")
            
            country_results = []
            for country in countries:
                filename = f"llama3_8b_{source}_country_{country}_{obj}.csv"
                filepath = os.path.join(base_path, filename)
                
                if os.path.exists(filepath):
                    print(f"Processing: {filename}")
                    try:
                        result = analyze_single_country_file(filepath, country.capitalize())
                        if result:
                            country_results.append(result)
                            print(f"✓ Processed {country}")
                    except Exception as e:
                        print(f"✗ Error processing {filename}: {str(e)}")
                else:
                    print(f"Warning: File not found - {filename}")
            
            if country_results:
                results_df = pd.DataFrame(country_results)
                key = f"{obj}_{source}_country_specific"
                all_results[key] = results_df
                
                output_filename = f"text_analysis_{obj}_{source}_country_specific.csv"
                results_df.to_csv(output_filename, index=False)
                print(f"\n✓ Country-specific results saved to: {output_filename}")
                
                print("\nResults:")
                print(results_df.round(3))
                
                print("\nCountries by Semantic Ambiguity:")
                print(results_df.sort_values('Semantic_Ambiguity', ascending=False)[['Country', 'Semantic_Ambiguity']].round(3))
                
                print("\nCountries by Rhetorical Devices:")
                print(results_df.sort_values('Rhetorical_Devices', ascending=False)[['Country', 'Rhetorical_Devices']].round(3))
                
                print("\nCountries by Cohesion Score:")
                print(results_df.sort_values('Cohesion_Score', ascending=False)[['Country', 'Cohesion_Score']].round(3))
                
                print("\nCountries by Open-Ended Questions:")
                print(results_df.sort_values('Open_Ended_Questions', ascending=False)[['Country', 'Open_Ended_Questions']].round(3))
    
    print(f"\n{'='*60}")
    print("Creating Summary Comparison")
    print(f"{'='*60}")
    
    summary_data = []
    for key, df in all_results.items():
        parts = key.split('_')
        obj = parts[0]
        source = parts[1]
        country_type = '_'.join(parts[2:])
        
        for _, row in df.iterrows():
            summary_data.append({
                'Objective': obj,
                'Source': source,
                'Country_Type': country_type,
                'Country': row['Country'],
                'Semantic_Ambiguity': row['Semantic_Ambiguity'],
                'Rhetorical_Devices': row['Rhetorical_Devices'],
                'Open_Ended_Questions': row['Open_Ended_Questions'],
                'Cohesion_Score': row['Cohesion_Score'],
                'Text_Length': row['Text_Length'],
                'Sentence_Count': row['Sentence_Count'],
                'Entry_Count': row['Entry_Count']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('text_analysis_summary_all_combinations.csv', index=False)
    print("\nSummary saved to: text_analysis_summary_all_combinations.csv")
    print(f"\nSummary statistics:")
    print(summary_df.groupby(['Objective', 'Source', 'Country_Type']).size())
    
    return all_results, summary_df


if __name__ == "__main__":
    base_path = './' 
    
    results, summary = process_all_files(base_path)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total combinations processed: {len(results)}")
    print("\nAll individual results saved with naming pattern:")
    print("  text_analysis_[obj]_[source]_country_all.csv")
    print("  text_analysis_[obj]_[source]_country_specific.csv")
    print("\nSummary file saved as:")
    print("  text_analysis_summary_all_combinations_ft.csv")
